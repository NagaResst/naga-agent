#!/usr/bin/env python3
"""离线下载 bge-base-zh-v1.5 模型文件（Python 原生，无需 wget/curl）。

用法：python3 scripts/download_model.py
      python3 scripts/download_model.py --check   # 仅检查是否已下载
"""

import os
import sys

# 项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bge-base-zh-v1.5")

# 国内镜像源（ModelScope 优先，hf-mirror 备选）
MIRRORS = [
    "https://modelscope.cn/models/BAAI/bge-base-zh-v1.5/resolve/master",
    "https://hf-mirror.com/BAAI/bge-base-zh-v1.5/resolve/main",
]

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


def check_model() -> bool:
    """检查模型是否已完整下载。"""
    if not os.path.isdir(MODEL_PATH):
        return False
    for fname, _ in MODEL_FILES:
        target = os.path.join(MODEL_PATH, fname)
        if not os.path.isfile(target) or os.path.getsize(target) == 0:
            return False
    return True


def download_file(url: str, target: str, fname: str, expected_size: int) -> bool:
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


def download_model():
    """下载模型文件，支持多镜像源自动回退。"""
    print("=== bge-base-zh-v1.5 模型下载工具 ===", flush=True)
    print(f"目标目录: {MODEL_PATH}", flush=True)
    print(f"镜像源:   {', '.join(MIRRORS)}", flush=True)
    print()

    # 检查是否已完整
    if check_model():
        print(f"模型已完整存在于 {MODEL_PATH}，无需下载。")
        return

    # 统计已有文件
    existing = sum(1 for fname, _ in MODEL_FILES
                   if os.path.isfile(os.path.join(MODEL_PATH, fname)) and os.path.getsize(os.path.join(MODEL_PATH, fname)) > 0)
    print(f"已有 {existing}/{len(MODEL_FILES)} 个文件，开始下载缺失文件...", flush=True)
    print()

    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(os.path.join(MODEL_PATH, "1_Pooling"), exist_ok=True)

    for fname, expected_size in MODEL_FILES:
        target = os.path.join(MODEL_PATH, fname)
        # 跳过已完整下载的文件
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
            if download_file(url, target, fname, expected_size):
                downloaded = True
                break
            else:
                # 清理不完整文件，换下一个镜像重试
                if os.path.isfile(target):
                    os.remove(target)
                print(f"  [换源] {fname} 切换到下一个镜像...", flush=True)

        if not downloaded:
            print(f"\n所有镜像均失败！可重新运行本脚本继续（已下载的文件会自动跳过）。", flush=True)
            sys.exit(1)

    print()
    print("=== 下载完成 ===", flush=True)
    print(f"模型保存在: {MODEL_PATH}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        if check_model():
            print(f"模型已完整: {MODEL_PATH}")
        else:
            print(f"模型不完整或不存在: {MODEL_PATH}")
            sys.exit(1)
    else:
        download_model()
