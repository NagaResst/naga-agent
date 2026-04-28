import threading
from typing import Optional

# Redis Layer1 缓存 key 前缀
_CORE_CACHE_PREFIX = "naga_agent:memory:core:"


def _build_mem0_config(memory_cfg: dict, redis_cfg: dict, api_key: str, base_url: Optional[str]):
    """根据 config.toml [memory] 段构建 mem0 MemoryConfig。"""
    from mem0.configs.base import MemoryConfig, VectorStoreConfig, EmbedderConfig, LlmConfig

    backend = memory_cfg.get("backend", "chroma")
    embedder_cfg = memory_cfg.get("embedder", {})
    embedder_base_url = embedder_cfg.get("base_url", "") or base_url or None

    # 嵌入模型配置
    embedder_provider = embedder_cfg.get("provider", "openai")
    if embedder_provider == "ollama":
        embedder = EmbedderConfig(
            provider="ollama",
            config={
                "model": embedder_cfg.get("model", "nomic-embed-text"),
                "ollama_base_url": embedder_base_url or "http://localhost:11434",
            },
        )
    else:
        embedder = EmbedderConfig(
            provider="openai",
            config={
                "model": embedder_cfg.get("model", "text-embedding-3-small"),
                "api_key": api_key,
                **({"openai_base_url": embedder_base_url} if embedder_base_url else {}),
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

    # 向量库配置
    if backend == "qdrant":
        cfg = memory_cfg.get("qdrant", {})
        if cfg.get("mode", "local") == "server":
            vector_store = VectorStoreConfig(
                provider="qdrant",
                config={"host": cfg.get("host", "localhost"), "port": cfg.get("port", 6333)},
            )
        else:
            vector_store = VectorStoreConfig(
                provider="qdrant",
                config={"path": cfg.get("path", "./qdrant_db")},
            )
    elif backend == "redis":
        rv_cfg = memory_cfg.get("redis_vector", {})
        vector_store = VectorStoreConfig(
            provider="redis",
            config={
                "redis_url": (
                    f"redis://:{redis_cfg.get('password', '')}@"
                    f"{redis_cfg.get('host', 'localhost')}:{redis_cfg.get('port', 6379)}"
                    f"/{redis_cfg.get('db', 0)}"
                    if redis_cfg.get("password")
                    else f"redis://{redis_cfg.get('host', 'localhost')}:{redis_cfg.get('port', 6379)}/{redis_cfg.get('db', 0)}"
                ),
                "index_name": rv_cfg.get("index_name", "naga_memories"),
                "embedding_model_dims": rv_cfg.get("vector_dim", 1536),
            },
        )
    else:
        # 默认 chroma
        cfg = memory_cfg.get("chroma", {})
        if cfg.get("mode", "local") == "server":
            vector_store = VectorStoreConfig(
                provider="chroma",
                config={"host": cfg.get("host", "localhost"), "port": cfg.get("port", 8000)},
            )
        else:
            vector_store = VectorStoreConfig(
                provider="chroma",
                config={"path": cfg.get("path", "./chroma_db")},
            )

    return MemoryConfig(
        vector_store=vector_store,
        embedder=embedder,
        llm=llm,
    )


class MemoryManager:
    """分层记忆管理器。

    Layer1（core）：用户核心偏好和事实，全量注入 system prompt。
                   持久存于向量库（metadata layer=core），Redis 维护快速读取缓存。
    Layer2（episodic）：情节记忆，按需语义检索注入上下文。
                       持久存于向量库（metadata layer=episodic）。
    """

    def __init__(self, memory_cfg: dict, redis_cfg: dict, api_key: str, base_url: Optional[str]):
        self._cfg = memory_cfg
        self._redis = None
        self._mem0 = None
        self._lock = threading.Lock()

        # 初始化 mem0
        try:
            from mem0 import Memory
            config = _build_mem0_config(memory_cfg, redis_cfg, api_key, base_url)
            self._mem0 = Memory(config=config)
        except Exception as e:
            print(f"[MemoryManager] 警告：mem0 初始化失败，记忆功能降级为 Redis KV。错误：{e}")

        # 初始化 Redis 客户端（用于 Layer1 缓存）
        try:
            import redis as redis_lib
            self._redis = redis_lib.Redis(
                host=redis_cfg.get("host", "localhost"),
                port=redis_cfg.get("port", 6379),
                db=redis_cfg.get("db", 0),
                password=redis_cfg.get("password") or None,
                decode_responses=True,
                socket_connect_timeout=5,
            )
        except Exception as e:
            print(f"[MemoryManager] 警告：Redis 连接失败，Layer1 缓存不可用。错误：{e}")

    @property
    def available(self) -> bool:
        return self._mem0 is not None

    # ── 写入 ────────────────────────────────────────────────────────────────

    def add(self, messages: list, user_id: str = "default", layer: str = "episodic"):
        """从消息列表中提取记忆，写入向量库。layer: core / episodic。"""
        if not self.available:
            return
        try:
            self._mem0.add(messages, user_id=user_id, metadata={"layer": layer})
        except Exception as e:
            print(f"[MemoryManager] add 失败：{e}")

    def save_core(self, key: str, value: str, user_id: str = "default"):
        """显式保存 Layer1 核心记忆：写向量库 + 写 Redis 缓存。"""
        if self.available:
            try:
                # 用单条文本写入，附带 key 标记便于精确查找
                self._mem0.add(
                    [{"role": "user", "content": f"{key}: {value}"}],
                    user_id=user_id,
                    metadata={"layer": "core", "key": key},
                )
            except Exception as e:
                print(f"[MemoryManager] save_core mem0 写入失败：{e}")
        # 无论 mem0 是否可用，Redis 缓存都写入
        if self._redis:
            try:
                self._redis.set(f"{_CORE_CACHE_PREFIX}{key}", value)
            except Exception:
                pass

    def delete(self, key: str, user_id: str = "default"):
        """删除 Layer1 核心记忆（向量库 + Redis 缓存）。"""
        if self.available:
            try:
                results = self._mem0.search(key, filters={"user_id": user_id, "layer": "core"}, top_k=10)
                memories = results if isinstance(results, list) else results.get("results", [])
                for m in memories:
                    mem_key = (m.get("metadata") or {}).get("key", "")
                    if mem_key == key:
                        self._mem0.delete(m["id"])
            except Exception as e:
                print(f"[MemoryManager] delete 失败：{e}")
        if self._redis:
            try:
                self._redis.delete(f"{_CORE_CACHE_PREFIX}{key}")
            except Exception:
                pass

    # ── 检索 ────────────────────────────────────────────────────────────────

    def search_core(self, user_id: str = "default") -> list[str]:
        """获取所有 Layer1 核心记忆。优先读 Redis 缓存，miss 时从向量库回填。"""
        # 先尝试 Redis 缓存
        if self._redis:
            try:
                cursor = 0
                items = []
                while True:
                    cursor, keys = self._redis.scan(
                        cursor, match=f"{_CORE_CACHE_PREFIX}*", count=100
                    )
                    for k in keys:
                        short = k[len(_CORE_CACHE_PREFIX):]
                        val = self._redis.get(k) or ""
                        items.append(f"{short}: {val}")
                    if cursor == 0:
                        break
                if items:
                    max_items = self._cfg.get("core_max_items", 20)
                    return items[:max_items]
            except Exception:
                pass

        # Redis 缓存 miss → 从向量库查询并回填
        if not self.available:
            return []
        try:
            results = self._mem0.search("", filters={"user_id": user_id, "layer": "core"}, top_k=self._cfg.get("core_max_items", 20))
            memories = results if isinstance(results, list) else results.get("results", [])
            items = []
            for m in memories:
                text = m.get("memory", "") or m.get("text", "")
                meta = m.get("metadata") or {}
                key = meta.get("key", "")
                if self._redis and key:
                    # 回填缓存
                    try:
                        val = text.split(": ", 1)[-1] if ": " in text else text
                        self._redis.set(f"{_CORE_CACHE_PREFIX}{key}", val)
                    except Exception:
                        pass
                if text:
                    items.append(text)
            return items
        except Exception as e:
            print(f"[MemoryManager] search_core 失败：{e}")
            return []

    def search_episodic(self, query: str, user_id: str = "default", top_k: int = 3) -> list[str]:
        """语义检索 Layer2 情节记忆，返回 top_k 条文本。"""
        if not self.available or not query.strip():
            return []
        try:
            results = self._mem0.search(query, filters={"user_id": user_id, "layer": "episodic"}, top_k=top_k)
            memories = results if isinstance(results, list) else results.get("results", [])
            return [m.get("memory", "") or m.get("text", "") for m in memories if m.get("memory") or m.get("text")]
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
