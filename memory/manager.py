import threading
from typing import Optional

# 保留常量，仅用于向量库 metadata 查询时的前缀标识（不再用于 Redis）
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
        if not redis_cfg:
            raise ValueError(
                "[memory] backend = 'redis' 需要提供 Redis 连接配置（redis_cfg），"
                "当前项目已移除 Redis 依赖，请将 backend 改为 'chroma' 或 'qdrant'。"
            )
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
                   持久存于 SQLite（精确查找，真相来源），进程内字典加速读取。
                   mem0 向量库异步同步（不影响主路径）。
    Layer2（episodic）：情节记忆，按需语义检索注入上下文。
                       持久存于向量库（metadata layer=episodic）。
    """

    def __init__(self, memory_cfg: dict, redis_cfg: Optional[dict] = None, api_key: str = "", base_url: Optional[str] = None, storage=None):
        self._cfg = memory_cfg
        self._storage = storage  # SQLiteSessionManager 实例，Layer1 持久化真相来源
        # Layer1 核心记忆的进程内缓存 {key: value}
        self._core_cache: dict = {}
        self._mem0 = None
        self._lock = threading.Lock()

        # 初始化 mem0
        try:
            from mem0 import Memory
            config = _build_mem0_config(memory_cfg, redis_cfg or {}, api_key, base_url)
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
