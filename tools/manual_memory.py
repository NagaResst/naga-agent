"""手动记忆管理模块：双层存储（full_document + section），直写 Qdrant memories 集合。"""

import os
import re
import json
from hashlib import md5
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

# 模块级配置（由 configure() 注入，或使用默认值）
_config = {
    "qdrant_host": "localhost",
    "qdrant_port": 6333,
    "daemon_url": "http://127.0.0.1:8000",
}

COLLECTION_NAME = "memories"
VECTOR_DIM = 768  # bge-base-zh-v1.5


def configure(qdrant_host: str = None, qdrant_port: int = None, daemon_url: str = None):
    """注入配置（由 core.py 在启动时调用）。"""
    global _client
    if qdrant_host is not None:
        _config["qdrant_host"] = qdrant_host
    if qdrant_port is not None:
        _config["qdrant_port"] = qdrant_port
    if daemon_url is not None:
        _config["daemon_url"] = daemon_url
    _client = None  # 配置变更后重建连接


_client = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=_config["qdrant_host"], port=_config["qdrant_port"])
    return _client


def _ensure_collection(client: QdrantClient):
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def get_embedding(text: str) -> list:
    """通过本地 Daemon 获取 768 维向量。"""
    import requests
    daemon_url = _config["daemon_url"]
    try:
        resp = requests.post(f"{daemon_url}/embed", json={"text": text}, timeout=30)
        if resp.status_code == 200:
            return resp.json()["vector"]
        raise Exception(f"Daemon error: {resp.text}")
    except Exception as e:
        raise ConnectionError(
            f"无法连接到语义引擎 ({daemon_url})。\n"
            f"请确保语义引擎已启动。原始错误: {e}"
        )


def parse_markdown(content: str, source_path: str) -> list:
    """按标题拆分 Markdown 为 section 列表，复用 skill 的逻辑。"""
    memories = []
    lines = content.split('\n')
    current_title = ""
    current_content = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            current_content.append(line)
            continue
        if in_code_block:
            current_content.append(line)
            continue

        title_match = re.match(r'^(#{1,6})\s+(.+)', line)
        if title_match:
            if current_title and current_content:
                memories.append({
                    "title": f"[{source_path}] {current_title}",
                    "content": '\n'.join(current_content).strip(),
                    "source": source_path,
                })
            current_title = title_match.group(2).strip()
            current_content = [line]
        else:
            current_content.append(line)

    if current_title and current_content:
        memories.append({
            "title": f"[{source_path}] {current_title}",
            "content": '\n'.join(current_content).strip(),
            "source": source_path,
        })

    if not memories and content.strip():
        filename = os.path.basename(source_path)
        memories.append({
            "title": f"[{source_path}] {filename}",
            "content": content.strip(),
            "source": source_path,
        })

    return memories


def check_duplicate(client, content, memory_type="section", source=None, threshold=0.80):
    """语义查重。section 同源查重阈值 0.80，full_document 全库查重阈值 0.95。"""
    if memory_type == "full_document":
        threshold = 0.95

    vector = get_embedding(content)
    must_conditions = [
        models.FieldCondition(key="memory_type", match=models.MatchValue(value=memory_type))
    ]
    if memory_type == "section" and source:
        must_conditions.append(
            models.FieldCondition(key="source", match=models.MatchValue(value=source))
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=models.Filter(must=must_conditions),
        limit=1,
    ).points

    if results and results[0].score > threshold:
        return results[0].score, results[0].payload
    return None, None


def add_document(content: str, source: str) -> str:
    """导入一个文档：双层存储 + 查重 + 引用机制。"""
    client = _get_client()
    _ensure_collection(client)

    if not content.strip():
        return "错误：文档内容为空。"

    results = []
    filename = os.path.basename(source)

    # ── full_document ──
    full_doc_title = f"[完整文档] {source}"
    score, existing = check_duplicate(client, content, memory_type="full_document")

    if existing:
        results.append(f"完整文档已存在 (相似度: {score:.3f})，已跳过")
    else:
        vector = get_embedding(content)
        point_id = md5(f"{full_doc_title}{content}".encode()).hexdigest()
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(
                id=point_id, vector=vector,
                payload={
                    "title": full_doc_title,
                    "content": content,
                    "source": source,
                    "memory_type": "full_document",
                    "filename": filename,
                },
            )],
        )
        results.append("✓ 完整文档已存储")

    # ── section 分段 ──
    sections = parse_markdown(content, source)
    section_results = []
    for sec in sections:
        vector = get_embedding(sec["content"])
        score, existing = check_duplicate(client, sec["content"], memory_type="section", source=sec["source"])

        payload = {
            "title": sec["title"],
            "content": sec["content"],
            "source": sec["source"],
            "memory_type": "section",
        }

        if existing:
            existing_title = existing.get("title", "")
            payload["content"] = f"内容类似 {existing_title}"
            payload["duplicate_of"] = existing_title
            section_results.append(f"  ✓ {sec['title'][:50]}... (标记重复, 相似度 {score:.3f})")
        else:
            section_results.append(f"  ✓ {sec['title'][:50]}...")

        point_id = md5(f"{sec['title']}{sec['content']}".encode()).hexdigest()
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    results.append(f"解析出 {len(sections)} 个分段记忆：")
    results.extend(section_results)
    return "\n".join(results)


def forget(title_or_source: str) -> str:
    """按 title 或 source 删除记忆。"""
    client = _get_client()
    _ensure_collection(client)

    # 滚动查找匹配的点
    deleted = 0
    offset = None
    to_delete = []
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=["title", "source"],
        )
        for p in points:
            payload = p.payload or {}
            if title_or_source in payload.get("title", "") or title_or_source in payload.get("source", ""):
                to_delete.append(p.id)
        if offset is None:
            break

    if not to_delete:
        return f"未找到匹配 '{title_or_source}' 的记忆。"

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(points=to_delete),
    )
    return f"已删除 {len(to_delete)} 条记忆（匹配 '{title_or_source}'）。"


def search_manual(query: str, top_k: int = 5) -> str:
    """语义搜索手动知识库（memories 集合）。"""
    client = _get_client()
    _ensure_collection(client)

    if not query.strip():
        return "错误：查询不能为空。"

    vector = get_embedding(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
    ).points

    if not results:
        return "未找到相关记忆。"

    lines = [f"搜索 '{query}' 返回 {len(results)} 条结果："]
    for i, point in enumerate(results, 1):
        payload = point.payload or {}
        title = payload.get("title", "(无标题)")
        content = payload.get("content", "")
        source = payload.get("source", "")
        mem_type = payload.get("memory_type", "")
        dup_of = payload.get("duplicate_of", "")
        score = point.score

        # 截断显示
        content_preview = content[:200] + ("..." if len(content) > 200 else "")
        lines.append(f"\n[{i}] {title}")
        lines.append(f"    来源: {source}  类型: {mem_type}  相似度: {score:.3f}")
        if dup_of:
            lines.append(f"    引用: 内容类似 {dup_of}")
        lines.append(f"    内容: {content_preview}")

    return "\n".join(lines)
