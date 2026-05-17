import os
import sys
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

# 配置 Hugging Face 国内镜像加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

app = FastAPI()

# 项目根目录：scripts/ 的上一级即 naga-agent/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
MODEL_PATH = os.path.join(project_root, "models", "bge-base-zh-v1.5")
MODEL_NAME = "BAAI/bge-base-zh-v1.5"

# 模型文件列表（fname, expected_size），供 _model_exists 和 _download_model 共用
MODEL_FILES = [
    ("config.json", 998),
    ("tokenizer_config.json", 366),
    ("tokenizer.json", 439124),
    ("vocab.txt", 109540),
    ("special_tokens_map.json", 125),
    ("pytorch_model.bin", 409138989),
    ("modules.json", 349),
    ("sentence_bert_config.json", 52),
    ("config_sentence_transformers.json", 124),
    ("1_Pooling/config.json", 190),
]

# 国内镜像源（ModelScope 优先，hf-mirror 备选）
MIRRORS = [
    "https://modelscope.cn/models/BAAI/bge-base-zh-v1.5/resolve/master",
    "https://hf-mirror.com/BAAI/bge-base-zh-v1.5/resolve/main",
]

model = None


def _model_exists() -> bool:
    """检查本地模型目录是否存在且所有文件大小匹配。"""
    if not os.path.isdir(MODEL_PATH):
        return False
    for fname, expected_size in MODEL_FILES:
        target = os.path.join(MODEL_PATH, fname)
        if not os.path.isfile(target) or os.path.getsize(target) != expected_size:
            return False
    return True


def _download_file(url: str, target: str, fname: str, expected_size: int):
    """用 requests 下载文件，实时打印进度。"""
    import requests

    size_str = f"{expected_size / 1024 / 1024:.1f}MB" if expected_size > 1024 * 1024 else f"{expected_size / 1024:.0f}KB"
    print(f"  [下载] {fname} ({size_str}) from {url.split('/')[2]}...", flush=True)
    print(f"  [连接] 正在连接服务器...", flush=True)

    try:
        resp = requests.get(url, stream=True, timeout=(10, 30), headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        print(f"  [失败] {fname} 连接失败: {e}", flush=True)
        if os.path.isfile(target):
            os.remove(target)
        return False

    total = int(resp.headers.get("Content-Length", expected_size))
    print(f"  [接收] 已连接，开始接收数据 ({total / 1024 / 1024:.1f}MB)...", flush=True)
    downloaded = 0
    chunk_size = 1024 * 64  # 64KB
    last_pct = -1

    try:
        with open(target, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                pct = int(downloaded * 100 / total) if total > 0 else 0
                # 每 5% 打印一次进度
                if pct != last_pct and pct % 5 == 0:
                    dl_str = f"{downloaded / 1024 / 1024:.1f}MB" if downloaded > 1024 * 1024 else f"{downloaded / 1024:.0f}KB"
                    total_str = f"{total / 1024 / 1024:.1f}MB" if total > 1024 * 1024 else f"{total / 1024:.0f}KB"
                    print(f"  [进度] {fname} {pct}% ({dl_str}/{total_str})", flush=True)
                    last_pct = pct
    except Exception as e:
        print(f"  [失败] {fname} 下载异常: {e}", flush=True)
        if os.path.isfile(target):
            os.remove(target)
        return False
    finally:
        resp.close()

    print(f"  [完成] {fname}", flush=True)
    return True


def _download_model():
    """下载模型文件，支持多镜像源自动回退。"""
    print(f"[下载] 本地模型不存在，开始下载...", flush=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(os.path.join(MODEL_PATH, "1_Pooling"), exist_ok=True)

    for fname, expected_size in MODEL_FILES:
        target = os.path.join(MODEL_PATH, fname)
        # 跳过已完整下载的文件（大小与预期匹配）
        if os.path.isfile(target) and os.path.getsize(target) == expected_size:
            print(f"  [跳过] {fname} (已存在)", flush=True)
            continue
        # 清理不完整文件
        if os.path.isfile(target) and os.path.getsize(target) > 0:
            print(f"  [续传] {fname} (不完整，重新下载)", flush=True)
            os.remove(target)

        # 逐个镜像尝试
        downloaded = False
        for mirror in MIRRORS:
            url = f"{mirror}/{fname}"
            if _download_file(url, target, fname, expected_size):
                downloaded = True
                break
            else:
                # 清理不完整文件，换下一个镜像重试
                if os.path.isfile(target):
                    os.remove(target)
                print(f"  [换源] {fname} 切换到下一个镜像...", flush=True)

        if not downloaded:
            print(f"[下载] 所有镜像均失败，将回退到 Python 自动下载。", flush=True)
            return False

    print("[下载] 所有文件下载完成。", flush=True)
    return True


def load_model():
    global model
    if model is not None:
        return

    # 1. 检查本地模型是否存在
    if _model_exists():
        print(f"Loading model from local path: {MODEL_PATH}...", flush=True)
        model = SentenceTransformer(MODEL_PATH)
    else:
        # 2. 尝试下载模型文件
        print(f"本地模型不存在: {MODEL_PATH}", flush=True)
        downloaded = _download_model()

        if downloaded and _model_exists():
            print(f"离线下载成功，从本地加载: {MODEL_PATH}...", flush=True)
            model = SentenceTransformer(MODEL_PATH)
        else:
            # 3. 回退到 Python (sentence-transformers) 自动下载
            print(f"离线下载失败，回退到 Python 自动下载 (HF_ENDPOINT={os.environ.get('HF_ENDPOINT')})...", flush=True)
            model = SentenceTransformer(MODEL_NAME)
            # 下载后保存到本地供后续使用
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            model.save(MODEL_PATH)
            print(f"Model saved to: {MODEL_PATH}", flush=True)

    print(f"Model loaded successfully. Dimension: {model.get_sentence_embedding_dimension()}", flush=True)


class EmbedRequest(BaseModel):
    text: str


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/embed")
def embed_text(request: EmbedRequest):
    if model is None:
        load_model()
    vector = model.encode(request.text, normalize_embeddings=True).tolist()
    return {"vector": vector, "dim": len(vector)}


# ── OpenAI 兼容端点（供 mem0 等库调用）────────────────────────────────────

class OpenAIEmbedRequest(BaseModel):
    input: str | list[str]
    model: str = "bge-base-zh-v1.5"


@app.post("/v1/embeddings")
def openai_embeddings(request: OpenAIEmbedRequest):
    """OpenAI /v1/embeddings 兼容接口，供 mem0 的 openai embedder 调用。"""
    if model is None:
        load_model()
    texts = request.input if isinstance(request.input, list) else [request.input]
    data = []
    for text in texts:
        vec = model.encode(text, normalize_embeddings=True).tolist()
        data.append({"object": "embedding", "embedding": vec, "index": len(data)})
    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"prompt_tokens": sum(len(t) for t in texts), "total_tokens": sum(len(t) for t in texts)},
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="bge-base-zh-v1.5 Embedding Daemon")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--cpu-threads", type=int, default=0, help="CPU 计算线程数（0=自动检测全部核心）")
    args = parser.parse_args()

    # 设置 PyTorch 计算线程数
    import torch
    thread_count = int(args.cpu_threads if args.cpu_threads > 0 else (os.cpu_count() or 12)/2)
    torch.set_num_threads(thread_count)
    print(f"[启动] CPU 线程数: {thread_count}", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
