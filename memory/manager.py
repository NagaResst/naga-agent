import os
import uuid
import threading
import warnings
from datetime import datetime
from urllib.parse import urljoin

import requests

os.environ["MEM0_TELEMETRY"] = "False"


def _build_mem0_config(memory_cfg: dict, api_key: str, base_url: str | None = None):
    """构建 mem0 MemoryConfig，使用可配置的 OpenAI 兼容嵌入接口。"""
    from mem0.configs.base import MemoryConfig, VectorStoreConfig, EmbedderConfig, LlmConfig

    embedder_cfg = memory_cfg.get("embedder", {})
    embedder = EmbedderConfig(
        provider="openai",
        config={
            "model": embedder_cfg.get("model", "bge-base-zh-v1.5"),
            "api_key": os.environ.get("VECTOR_API_KEY") or embedder_cfg.get("api_key") or api_key,
            "openai_base_url": embedder_cfg.get("base_url", "http://127.0.0.1:8000/v1"),
        },
    )

    llm_model = memory_cfg.get("llm_model", "")
    llm_kwargs = {"model": llm_model, "api_key": api_key}
    if base_url:
        llm_kwargs["openai_base_url"] = base_url
    llm = LlmConfig(provider="openai", config=llm_kwargs) if llm_model else None

    qdrant_cfg = memory_cfg.get("qdrant", {})
    vector_store = VectorStoreConfig(
        provider="qdrant",
        config={
            "host": qdrant_cfg.get("host", "localhost"),
            "port": qdrant_cfg.get("port", 6333),
            "collection_name": qdrant_cfg.get("collection_name", "mem0"),
            "embedding_model_dims": qdrant_cfg.get("embedding_dims", 768),
        },
    )

    return MemoryConfig(vector_store=vector_store, embedder=embedder, llm=llm)


class MemoryManager:
    """记忆管理器。

    Layer1 核心记忆（core）:
      - SQLite 持久化，进程内 _core_cache 加速
      - 全量注入 system prompt，上限 core_max_items 条，单条 core_max_chars
      - 支持 save / list / delete（按 ID）
      - 不碰 mem0

    Layer2 情景记忆（episodic）:
      - mem0 SDK 管理，Qdrant 持久化
      - 自动提取(每 N 轮) + 手动保存 + 语义检索 + 按 ID 删除
    """

    def __init__(self, memory_cfg: dict, api_key: str = "", base_url: str | None = None, storage=None, activity=None):
        self._cfg = memory_cfg
        self._storage = storage
        self._api_key = api_key
        # 核心记忆缓存 {id: {"content": str, "created_at": str}}
        self._core_cache: dict[str, dict] = {}
        self._mem0 = None
        self._turn_count = 0
        self._activity = activity

        # mem0 初始化
        try:
            from mem0 import Memory
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=r"^Qdrant client version .* is incompatible with server version .*",
                    category=UserWarning,
                )
                self._mem0 = Memory(config=_build_mem0_config(memory_cfg, api_key, base_url))
        except Exception as e:
            print(f"[MemoryManager] mem0 初始化失败，情景记忆不可用。错误：{e}")

        # 预热核心记忆
        self._load_core_cache()

    @property
    def available(self) -> bool:
        return self._mem0 is not None

    # ── 内部 ────────────────────────────────────────────────

    def _load_core_cache(self):
        if self._storage is None:
            return
        try:
            for row in self._storage.list_memories():
                self._core_cache[row["id"]] = {
                    "content": row["content"],
                    "created_at": row.get("created_at", ""),
                }
        except Exception as e:
            print(f"[MemoryManager] 预热核心记忆缓存失败：{e}")

    def increment_turn(self):
        self._turn_count += 1

    # ── Layer1: 核心记忆 ────────────────────────────────────

    def save_core(self, content: str) -> str:
        """返回成功消息或错误消息。"""
        content = content.strip()
        if not content:
            return "错误：内容不能为空"

        max_chars = self._cfg.get("core_max_chars", 200)
        if len(content) > max_chars:
            return f"核心记忆单条内容不可超过 {max_chars} 字符，当前 {len(content)} 字符"

        max_items = self._cfg.get("core_max_items", 20)
        if len(self._core_cache) >= max_items:
            return f"核心记忆已达上限（{max_items}条），请先 delete 后再保存"

        memory_id = uuid.uuid4().hex[:8]
        now = datetime.now().isoformat()

        if self._storage is not None:
            self._storage.save_memory(memory_id, content)

        self._core_cache[memory_id] = {"content": content, "created_at": now}
        return f"已保存，ID: {memory_id}"

    def list_core(self) -> list[dict]:
        items = []
        for mid, info in self._core_cache.items():
            items.append({"id": mid, "content": info["content"][:200], "created_at": info.get("created_at", "")})
        return items

    def delete_core(self, memory_id: str) -> str:
        if memory_id not in self._core_cache:
            return f"未找到记忆：{memory_id}"
        if self._storage is not None:
            self._storage.delete_memory(memory_id)
        self._core_cache.pop(memory_id, None)
        return f"已删除：{memory_id}"

    # ── Layer2: 情景记忆 ────────────────────────────────────

    def save_episodic(self, content: str) -> str:
        if not self.available:
            return "错误：情景记忆不可用（mem0 未初始化）"
        content = content.strip()
        if not content:
            return "错误：内容不能为空"
        memory_id = uuid.uuid4().hex[:8]
        try:
            self._mem0.add(
                [{"role": "assistant", "content": content}],
                agent_id=self._cfg.get("agent_id", "naga-agent"),
                metadata={"memory_id": memory_id},
            )
            return f"已保存，ID: {memory_id}"
        except Exception as e:
            return f"保存失败：{e}"

    def recall_episodic(self, query: str, top_k: int = 15) -> list[dict]:
        if not self.available or not query.strip():
            return []
        try:
            if self._activity is not None:
                self._activity.set("memory_search", query[:30])
            results = self._mem0.search(
                query, filters={"agent_id": self._cfg.get("agent_id", "naga-agent")}, top_k=top_k,
            )
            memories = results if isinstance(results, list) else results.get("results", [])
            out = []
            for m in memories:
                text = (m.get("memory") or "").strip()
                if not text:
                    continue
                meta = m.get("metadata") or {}
                out.append({
                    "id": meta.get("memory_id", ""),
                    "content": text,
                    "score": m.get("score", 0),
                })
            return self._rerank_episodic(query, out)
        except Exception as e:
            print(f"[MemoryManager] recall_episodic 失败：{e}")
            return []

    def _rerank_episodic(self, query: str, candidates: list[dict]) -> list[dict]:
        cfg = self._cfg.get("reranker", {})
        if len(candidates) < 2 or not cfg.get("model") or not cfg.get("base_url"):
            return candidates
        activity = getattr(self, "_activity", None)
        if activity is not None:
            activity.set("memory_rerank", f"{len(candidates)}条")
        try:
            response = requests.post(
                urljoin(cfg["base_url"].rstrip("/") + "/", "rerank"),
                headers={"Authorization": f"Bearer {os.environ.get('VECTOR_API_KEY') or cfg.get('api_key') or self._api_key}"},
                json={"model": cfg["model"], "query": query,
                      "documents": [item["content"] for item in candidates], "top_n": len(candidates)},
                timeout=cfg.get("timeout", 10),
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", []) if isinstance(payload, dict) else []
            if not isinstance(results, list) or not results:
                return candidates
            ranked = []
            seen = set()
            for result in sorted((item for item in results if isinstance(item, dict)),
                                 key=lambda item: item.get("relevance_score", 0) or 0, reverse=True):
                index = result.get("index")
                if not isinstance(index, int) or index < 0 or index >= len(candidates) or index in seen:
                    continue
                seen.add(index)
                score = result.get("relevance_score")
                ranked.append({**candidates[index], "score": score if isinstance(score, (int, float)) else candidates[index]["score"]})
            return ranked + [item for index, item in enumerate(candidates) if index not in seen]
        except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError) as exc:
            print(f"[MemoryManager] rerank 不可用，保留向量排序：{type(exc).__name__}")
            return candidates

    def delete_episodic(self, memory_id: str) -> str:
        if not self.available:
            return "错误：情景记忆不可用（mem0 未初始化）"
        if not memory_id.strip():
            return "错误：memory_id 不能为空"
        try:
            # mem0 搜索匹配的 memory_id
            scope = {"agent_id": self._cfg.get("agent_id", "naga-agent")}
            results = self._mem0.search(memory_id, filters={**scope, "memory_id": memory_id}, top_k=5)
            memories = results if isinstance(results, list) else results.get("results", [])
            deleted = 0
            for m in memories:
                mid = (m.get("metadata") or {}).get("memory_id", "")
                if mid == memory_id:
                    self._mem0.delete(m["id"])
                    deleted += 1
            if deleted > 0:
                return f"已删除：{memory_id}"
            # fallback: 搜索不到用 memory_id 作为查询词再试
            results2 = self._mem0.search(memory_id, filters=scope, top_k=10)
            memories2 = results2 if isinstance(results2, list) else results2.get("results", [])
            for m in memories2:
                mid = (m.get("metadata") or {}).get("memory_id", "")
                if mid == memory_id:
                    self._mem0.delete(m["id"])
                    deleted += 1
            return f"已删除：{memory_id}" if deleted > 0 else f"未找到记忆：{memory_id}"
        except Exception as e:
            return f"删除失败：{e}"

    def list_episodic(self, top_k: int = 20) -> list[dict]:
        if not self.available:
            return []
        try:
            results = self._mem0.get_all(filters={"agent_id": self._cfg.get("agent_id", "naga-agent")}, top_k=top_k)
            memories = results if isinstance(results, list) else results.get("results", [])
            out = []
            for m in memories:
                text = (m.get("memory") or "").strip()
                if not text:
                    continue
                meta = m.get("metadata") or {}
                out.append({
                    "id": meta.get("memory_id", ""),
                    "content": text[:200],
                    "score": m.get("score", 0),
                })
            return out
        except Exception as e:
            print(f"[MemoryManager] list_episodic 失败：{e}")
            return []

    # ── 自动提取 ─────────────────────────────────────────────

    def should_extract(self) -> bool:
        every_n = self._cfg.get("extract_every_n_turns", 10)
        if every_n <= 0:
            return False
        return self._turn_count > 0 and self._turn_count % every_n == 0

    def extract_async(self, messages: list):
        if not getattr(self, "available", False):
            return
        activity = getattr(self, "_activity", None)

        def _run(*args, **kwargs):
            if activity is not None:
                activity.set_background("extract")
            try:
                self._mem0.add(*args, **kwargs)
            except Exception as exc:
                print(f"[MemoryManager] 后台记忆提取失败：{exc}")
            finally:
                if activity is not None:
                    activity.clear_background()

        t = threading.Thread(
            target=_run,
            args=(messages,),
            kwargs={"agent_id": self._cfg.get("agent_id", "naga-agent"),
                    "metadata": {"layer": "episodic", "source": "auto_extract"}},
            daemon=True,
        )
        t.start()