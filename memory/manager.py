import os
import threading
from typing import Optional

from agent.skill_registry import _extract_match_terms, _normalize_skill_text

# 禁用 mem0 遥测（避免 atexit 时 posthog 关闭报错）
os.environ["MEM0_TELEMETRY"] = "False"

# 保留常量，仅用于向量库 metadata 查询时的前缀标识（不再用于 Redis）
_CORE_CACHE_PREFIX = "naga_agent:memory:core:"


def _build_mem0_config(memory_cfg: dict, api_key: str, base_url: Optional[str]):
    """根据 config.toml [memory] 段构建 mem0 MemoryConfig。

    统一使用 Qdrant server + 本地 bge-base-zh-v1.5 Daemon 嵌入（768 维）。
    mem0 集合名 "mem0"，手动记忆集合名 "memories"，两者共存于同一 Qdrant 实例。
    """
    from mem0.configs.base import MemoryConfig, VectorStoreConfig, EmbedderConfig, LlmConfig

    embedder_cfg = memory_cfg.get("embedder", {})
    embedder_base_url = embedder_cfg.get("base_url", "http://127.0.0.1:8000/v1")

    # 嵌入模型：统一走本地 Daemon 的 OpenAI 兼容接口（bge-base-zh-v1.5, 768 维）
    embedder = EmbedderConfig(
        provider="openai",
        config={
            "model": embedder_cfg.get("model", "bge-base-zh-v1.5"),
            "api_key": embedder_cfg.get("api_key", "daemon"),
            "openai_base_url": embedder_base_url,
        },
    )

    # LLM 配置（mem0 内部提取记忆时使用，复用主模型接入点）
    llm_model = memory_cfg.get("llm_model", "qwen3.6-flash")
    llm = LlmConfig(
        provider="openai",
        config={
            "model": llm_model,
            "api_key": api_key,
            **({"openai_base_url": base_url} if base_url else {}),
        },
    )

    # 向量库：统一 Qdrant server 模式
    cfg = memory_cfg.get("qdrant", {})
    qdrant_config = {
        "host": cfg.get("host", "localhost"),
        "port": cfg.get("port", 6333),
        "collection_name": "mem0",
        "embedding_model_dims": 768,  # bge-base-zh-v1.5 维度，默认 1536 会导致维度不匹配
    }
    vector_store = VectorStoreConfig(
        provider="qdrant",
        config=qdrant_config,
    )

    return MemoryConfig(
        vector_store=vector_store,
        embedder=embedder,
        llm=llm,
    )


class MemoryManager:
    """分层记忆管理器。

    Layer1（core）：用户核心偏好和事实，全量注入 system prompt。
                   持久存于 SQLite（精确查找，真相来源），进程内字典加速读取。
                   mem0 向量库异步同步（不影响主路径）。
    Layer2（episodic）：情节记忆，按需语义检索注入上下文。
                       持久存于向量库（metadata layer=episodic）。
    """

    def __init__(self, memory_cfg: dict, api_key: str = "", base_url: Optional[str] = None, storage=None):
        self._cfg = memory_cfg
        self._storage = storage  # SQLiteSessionManager 实例，Layer1 持久化真相来源
        # Layer1 核心记忆的进程内缓存 {key: value}
        self._core_cache: dict = {}
        self._mem0 = None
        self._lock = threading.Lock()

        # 初始化 mem0
        try:
            from mem0 import Memory
            config = _build_mem0_config(memory_cfg, api_key, base_url)
            self._mem0 = Memory(config=config)
        except Exception as e:
            print(f"[MemoryManager] 警告：mem0 初始化失败，向量语义检索不可用（精确查找仍正常）。错误：{e}")

        # 从 SQLite 预热 Layer1 缓存
        self._load_core_from_storage()

    @property
    def available(self) -> bool:
        return self._mem0 is not None

    # ── 内部初始化 ──────────────────────────────────────────────────

    def _load_core_from_storage(self):
        """从 SQLite 加载 Layer1 记忆预热进程内缓存。"""
        if self._storage is None:
            return
        try:
            memories = self._storage.list_memories()
            self._core_cache.update(memories)
        except Exception as e:
            print(f"[MemoryManager] 警告：预热 Layer1 缓存失败：{e}")

    # ── 写入 ────────────────────────────────────────────────────────────────

    def add(self, messages: list, user_id: str = "default", layer: str = "episodic"):
        """从消息列表中提取记忆，写入向量库。layer: core / episodic。"""
        if not self.available:
            return
        try:
            self._mem0.add(messages, user_id=user_id, metadata={"layer": layer})
        except Exception as e:
            print(f"[MemoryManager] add 失败：{e}")

    @staticmethod
    def _dedupe_text_items(items: list[str], limit: int = 12, max_chars: int = 240) -> list[str]:
        result = []
        seen = set()
        for item in items:
            text = " ".join((item or "").split()).strip()
            if not text:
                continue
            text = text[:max_chars]
            if text in seen:
                continue
            seen.add(text)
            result.append(text)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _score_episodic_candidate(query: str, text: str, source: str = "", base_score: float = 0.0) -> float:
        normalized_query = _normalize_skill_text(query)
        normalized_text = _normalize_skill_text(text)
        if not normalized_query or not normalized_text:
            return base_score

        score = float(base_score)
        query_terms = _extract_match_terms(normalized_query)[:18]
        priority_terms = []
        for raw_line in (query or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("当前步骤：", "最近工作结论：", "最近用户问题：")):
                _, _, value = line.partition("：")
                priority_terms.extend(_extract_match_terms(value))
        priority_terms = priority_terms[:12]

        if normalized_text in normalized_query or normalized_query in normalized_text:
            score += 3.5

        for term in query_terms:
            if term not in normalized_text:
                continue
            if len(term) >= 6:
                score += 2.2
            elif len(term) >= 4:
                score += 1.6
            else:
                score += 0.9

        for term in priority_terms:
            if term not in normalized_text:
                continue
            if len(term) >= 4:
                score += 2.6
            else:
                score += 1.5

        if text.startswith("步骤结论:"):
            score += 1.6
        elif text.startswith("工作结论:"):
            score += 2.1
        elif text.startswith("当前步骤:"):
            score += 1.1
        elif text.startswith("用户约束:"):
            score -= 0.4
        elif text.startswith("任务目标:"):
            score -= 1.2

        if source.startswith("step_summary"):
            score += 0.8
        elif source == "pre_compress":
            score += 0.4

        return score

    def _rerank_episodic_memories(self, query: str, memories: list, top_k: int) -> list[str]:
        scored = []
        for index, item in enumerate(memories):
            text = item.get("memory", "") or item.get("text", "")
            if not text:
                continue
            metadata = item.get("metadata") or {}
            raw_score = item.get("score")
            try:
                base_score = float(raw_score)
            except (TypeError, ValueError):
                base_score = 0.0
            score = self._score_episodic_candidate(query, text, metadata.get("source", ""), base_score)
            scored.append((score, index, text))

        scored.sort(key=lambda row: (-row[0], row[1]))
        pinned_prefixes = ("任务目标:", "用户约束:")
        execution_prefixes = ("步骤结论:", "工作结论:", "当前步骤:", "已完成步骤:")
        has_execution_items = any(text.startswith(execution_prefixes) for _, _, text in scored)
        primary = []
        fallback = []
        for _, _, text in scored:
            if has_execution_items and text.startswith(pinned_prefixes):
                fallback.append(text)
            else:
                primary.append(text)
        ranked_texts = primary if has_execution_items and len(primary) >= top_k else primary + fallback
        return self._dedupe_text_items(ranked_texts, limit=top_k, max_chars=320)

    def add_episodic_records(self, records: list[str], user_id: str = "default", source: str = "runtime"):
        """将宿主管理的结构化事实项写入 Layer2，避免把长原始消息整段写入向量记忆。"""
        if not self.available:
            return
        normalized = self._dedupe_text_items(records)
        if not normalized:
            return
        try:
            messages = [{"role": "assistant", "content": text} for text in normalized]
            self._mem0.add(messages, user_id=user_id, metadata={"layer": "episodic", "source": source[:40]})
        except Exception as e:
            print(f"[MemoryManager] add_episodic_records 失败：{e}")

    def save_core(self, key: str, value: str, user_id: str = "default"):
        """显式保存 Layer1 核心记忆。

        主路径：同步写 SQLite + 更新进程内缓存。
        mem0 向量库：后台线程异步写入（失败不影响主路径）。
        """
        # 主路径：SQLite 持久化
        if self._storage is not None:
            try:
                self._storage.set_memory(key, value)
            except Exception as e:
                print(f"[MemoryManager] save_core SQLite 写入失败：{e}")
        # 进程内缓存
        self._core_cache[key] = value
        # 异步写入 mem0 向量库（fire-and-forget）
        if self.available:
            def _async_write():
                try:
                    self._mem0.add(
                        [{"role": "user", "content": f"{key}: {value}"}],
                        user_id=user_id,
                        metadata={"layer": "core", "key": key},
                    )
                except Exception:
                    pass
            threading.Thread(target=_async_write, daemon=True).start()

    def delete(self, key: str, user_id: str = "default"):
        """删除 Layer1 核心记忆。

        主路径：同步删 SQLite + 清进程内缓存。
        mem0 向量库：后台线程异步清理（尽力而为）。
        """
        # 主路径：SQLite 删除
        if self._storage is not None:
            try:
                self._storage.delete_memory(key)
            except Exception as e:
                print(f"[MemoryManager] delete SQLite 删除失败：{e}")
        # 进程内缓存
        self._core_cache.pop(key, None)
        # 异步清理 mem0 向量库
        if self.available:
            def _async_delete():
                try:
                    results = self._mem0.search(key, filters={"user_id": user_id, "layer": "core"}, top_k=10)
                    memories = results if isinstance(results, list) else results.get("results", [])
                    for m in memories:
                        mem_key = (m.get("metadata") or {}).get("key", "")
                        if mem_key == key:
                            self._mem0.delete(m["id"])
                except Exception:
                    pass
            threading.Thread(target=_async_delete, daemon=True).start()

    # ── 检索 ────────────────────────────────────────────────────────────────

    def search_core(self, user_id: str = "default") -> list[str]:
        """获取所有 Layer1 核心记忆。直接读进程内缓存（已由 SQLite 预热）。"""
        max_items = self._cfg.get("core_max_items", 20)
        if self._core_cache:
            items = [f"{k}: {v}" for k, v in self._core_cache.items()]
            return items[:max_items]
        # 缓存为空（无任何已保存的记忆）—— 直接返回空列表
        return []

    def search_episodic(self, query: str, user_id: str = "default", top_k: int = 3) -> list[str]:
        """语义检索 Layer2 情节记忆，返回 top_k 条文本。"""
        if not self.available or not query.strip():
            return []
        try:
            fetch_k = max(top_k * 3, top_k + 3)
            results = self._mem0.search(query, filters={"user_id": user_id, "layer": "episodic"}, top_k=fetch_k)
            memories = results if isinstance(results, list) else results.get("results", [])
            return self._rerank_episodic_memories(query, memories, top_k)
        except Exception as e:
            print(f"[MemoryManager] search_episodic 失败：{e}")
            return []

    # ── 触发判断 ─────────────────────────────────────────────────────────────

    def should_extract(self, turn_count: int, last_content: str) -> bool:
        """判断是否应触发记忆提取。"""
        every_n = self._cfg.get("extract_every_n_turns", 3)
        min_len = self._cfg.get("min_content_length", 20)
        if len(last_content) < min_len:
            return False
        if every_n <= 0:
            return False
        return turn_count % every_n == 0

    def extract_async(self, messages: list, user_id: str = "default"):
        """在后台线程中异步提取情节记忆，不阻塞主对话流。"""
        if not self.available:
            return
        t = threading.Thread(
            target=self.add,
            args=(messages, user_id, "episodic"),
            daemon=True,
        )
        t.start()

    def extract_async_records(self, records: list[str], user_id: str = "default", source: str = "runtime"):
        """后台异步写入结构化情节记忆。"""
        if not self.available:
            return
        t = threading.Thread(
            target=self.add_episodic_records,
            args=(records, user_id, source),
            daemon=True,
        )
        t.start()
