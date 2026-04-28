import json
from datetime import datetime

import redis


class SessionManager:
    def __init__(self, redis_config: dict):
        try:
            self._client = redis.Redis(
                host=redis_config["host"],
                port=redis_config["port"],
                db=redis_config["db"],
                password=redis_config.get("password"),
                decode_responses=True,
                socket_connect_timeout=5,
            )
            self._client.ping()
        except (redis.ConnectionError, redis.TimeoutError) as e:
            raise ConnectionError(
                f"无法连接到 Redis ({redis_config['host']}:{redis_config['port']})：{e}\n"
                "请确认 Redis 服务已启动，或检查配置文件中的连接信息。"
            )

    def _meta_key(self, session_id: str) -> str:
        return f"naga_agent:session:{session_id}:meta"

    def _messages_key(self, session_id: str) -> str:
        return f"naga_agent:session:{session_id}:messages"

    def list_sessions(self) -> list:
        sessions = []
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor, match="naga_agent:session:*:meta", count=100)
            for key in keys:
                meta = self._client.hgetall(key)
                session_id = meta.get("id", "")
                msg_count = self._client.llen(self._messages_key(session_id))
                sessions.append({
                    "id": session_id,
                    "name": meta.get("name", ""),
                    "created_at": meta.get("created_at", ""),
                    "message_count": msg_count,
                })
            if cursor == 0:
                break
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions

    def create_session(self, name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = name.strip().replace(" ", "_") or "unnamed"
        session_id = f"{timestamp}_{safe_name}"
        self._client.hset(self._meta_key(session_id), mapping={
            "id": session_id,
            "name": safe_name,
            "created_at": datetime.now().isoformat(),
        })
        return session_id

    def get_history(self, session_id: str) -> list:
        raw_messages = self._client.lrange(self._messages_key(session_id), 0, -1)
        return [json.loads(m) for m in raw_messages]

    def append_message(self, session_id: str, role: str, content: str):
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }, ensure_ascii=False)
        self._client.rpush(self._messages_key(session_id), message)

    def rename_session(self, session_id: str, new_name: str):
        safe_name = new_name.strip().replace(" ", "_") or "unnamed"
        self._client.hset(self._meta_key(session_id), "name", safe_name)

    def delete_session(self, session_id: str):
        self._client.delete(self._meta_key(session_id))
        self._client.delete(self._messages_key(session_id))

    def clear_messages(self, session_id: str):
        self._client.delete(self._messages_key(session_id))

    # ── Token 用量 ──────────────────────────────────────────────────────────

    def _token_usage_key(self, session_id: str) -> str:
        return f"naga_agent:session:{session_id}:token_usage"

    def append_token_usage(self, session_id: str, record: dict):
        import json as _json
        self._client.rpush(self._token_usage_key(session_id), _json.dumps(record, ensure_ascii=False))

    def get_token_usage(self, session_id: str) -> list:
        import json as _json
        raw = self._client.lrange(self._token_usage_key(session_id), 0, -1)
        result = []
        for r in raw:
            try:
                result.append(_json.loads(r))
            except Exception:
                pass
        return result

    # ── 跨会话记忆 ──────────────────────────────────────────────────────────

    def _memory_key(self, key: str) -> str:
        return f"naga_agent:memory:{key}"

    def set_memory(self, key: str, value: str):
        self._client.set(self._memory_key(key), value)

    def get_memory(self, key: str) -> str | None:
        return self._client.get(self._memory_key(key))

    def list_memories(self) -> dict:
        result = {}
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor, match="naga_agent:memory:*", count=100)
            for k in keys:
                short_key = k[len("naga_agent:memory:"):]
                result[short_key] = self._client.get(k) or ""
            if cursor == 0:
                break
        return result

    def delete_memory(self, key: str):
        self._client.delete(self._memory_key(key))

    # ── Redis 客户端透传（供路由缓存等直接使用）──────────────────────────

    @property
    def redis_client(self):
        return self._client

    # ── 会话历史摘要（持久化压缩摘要）──────────────────────────────────

    def _summary_key(self, session_id: str) -> str:
        return f"naga_agent:session:{session_id}:summary"

    def get_summary(self, session_id: str) -> str:
        return self._client.get(self._summary_key(session_id)) or ""

    def set_summary(self, session_id: str, text: str):
        self._client.set(self._summary_key(session_id), text)
