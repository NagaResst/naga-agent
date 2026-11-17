import json
import sqlite3
import threading
from datetime import datetime


class SessionManager:
    """SQLite-backed session manager.

    Drop-in replacement for the Redis-based SessionManager.
    All public methods keep identical signatures.
    """

    def __init__(self, db_path: str = "naga.db"):
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    # ── 连接管理 ──────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """每个线程持有独立连接（thread-local）。"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                metadata   TEXT DEFAULT NULL,
                timestamp  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE TABLE IF NOT EXISTS token_usage (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                data       TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_token_usage_session ON token_usage(session_id);
            CREATE TABLE IF NOT EXISTS summaries (
                session_id TEXT PRIMARY KEY,
                text       TEXT NOT NULL DEFAULT '',
                last_message_id INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS memories (
                id         TEXT PRIMARY KEY,
                content    TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT ''
            );
        """)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
        if "metadata" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN metadata TEXT DEFAULT NULL")
        summary_columns = {row["name"] for row in conn.execute("PRAGMA table_info(summaries)")}
        if "last_message_id" not in summary_columns:
            conn.execute("ALTER TABLE summaries ADD COLUMN last_message_id INTEGER NOT NULL DEFAULT 0")
        conn.commit()

    # ── 会话管理 ──────────────────────────────────────────────────────────

    def list_sessions(self) -> list:
        conn = self._conn()
        rows = conn.execute(
            "SELECT s.id, s.name, s.created_at, COUNT(m.id) AS message_count "
            "FROM sessions s LEFT JOIN messages m ON m.session_id = s.id "
            "GROUP BY s.id ORDER BY s.created_at DESC"
        ).fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "message_count": row["message_count"],
            }
            for row in rows
        ]

    def create_session(self, name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = name.strip().replace(" ", "_") or "unnamed"
        session_id = f"{timestamp}_{safe_name}"
        conn = self._conn()
        conn.execute(
            "INSERT INTO sessions (id, name, created_at) VALUES (?, ?, ?)",
            (session_id, safe_name, datetime.now().isoformat()),
        )
        conn.commit()
        return session_id

    def rename_session(self, session_id: str, new_name: str):
        safe_name = new_name.strip().replace(" ", "_") or "unnamed"
        conn = self._conn()
        conn.execute("UPDATE sessions SET name = ? WHERE id = ?", (safe_name, session_id))
        conn.commit()

    def delete_session(self, session_id: str):
        conn = self._conn()
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM token_usage WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
        conn.commit()

    # ── 消息历史 ──────────────────────────────────────────────────────────

    def get_history(self, session_id: str) -> list:
        conn = self._conn()
        rows = conn.execute(
            "SELECT role, content, metadata, timestamp FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        result = []
        for r in rows:
            msg = {"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]}
            if r["metadata"]:
                try:
                    msg.update(json.loads(r["metadata"]))
                except Exception:
                    pass
            result.append(msg)
        return result

    def get_history_before(self, session_id: str, before_id: int | None = None, limit: int = 40) -> list[dict]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        query = (
            "SELECT id, role, content, metadata, timestamp FROM messages "
            "WHERE session_id = ?"
        )
        params: list[str | int] = [session_id]
        if before_id is not None:
            query += " AND id < ?"
            params.append(before_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn().execute(query, params).fetchall()
        result = []
        for row in reversed(rows):
            message = {"id": row["id"], "role": row["role"], "content": row["content"],
                       "timestamp": row["timestamp"]}
            if row["metadata"]:
                try:
                    metadata = json.loads(row["metadata"])
                    if isinstance(metadata, dict):
                        message.update(metadata)
                except (ValueError, TypeError):
                    pass
            message["id"] = row["id"]
            result.append(message)
        return result

    def get_history_since(self, session_id: str, after_id: int = 0) -> list[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT id, role, content, metadata FROM messages "
            "WHERE session_id = ? AND id > ? ORDER BY id", (session_id, after_id),
        ).fetchall()
        result = []
        for row in rows:
            message = {"id": row["id"], "role": row["role"], "content": row["content"]}
            if row["metadata"]:
                try:
                    metadata = json.loads(row["metadata"])
                    if isinstance(metadata, dict):
                        message.update({key: metadata[key] for key in ("tool_calls", "tool_call_id", "name") if key in metadata})
                except (ValueError, TypeError):
                    pass
            result.append(message)
        return result

    def append_message(self, session_id: str, role: str, content: str, metadata: dict = None):
        conn = self._conn()
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        conn.execute(
            "INSERT INTO messages (session_id, role, content, metadata, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, metadata_json, datetime.now().isoformat()),
        )
        conn.commit()

    def append_messages(self, session_id: str, messages: list[dict]):
        conn = self._conn()
        timestamp = datetime.now().isoformat()
        with conn:
            conn.executemany(
                "INSERT INTO messages (session_id, role, content, metadata, timestamp) VALUES (?, ?, ?, ?, ?)",
                [
                    (session_id, message["role"], message.get("content") or "",
                     json.dumps({key: message[key] for key in ("tool_calls", "tool_call_id", "name") if key in message},
                                ensure_ascii=False), timestamp)
                    for message in messages
                ],
            )

    def clear_messages(self, session_id: str):
        conn = self._conn()
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
        conn.commit()

    # ── Token 用量 ──────────────────────────────────────────────────────────

    def append_token_usage(self, session_id: str, record: dict):
        conn = self._conn()
        conn.execute(
            "INSERT INTO token_usage (session_id, data) VALUES (?, ?)",
            (session_id, json.dumps(record, ensure_ascii=False)),
        )
        conn.commit()

    def get_token_usage(self, session_id: str) -> list:
        conn = self._conn()
        rows = conn.execute(
            "SELECT data FROM token_usage WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        result = []
        for row in rows:
            try:
                result.append(json.loads(row["data"]))
            except Exception:
                pass
        return result

    # ── 会话历史摘要 ──────────────────────────────────────────────────────

    def get_summary(self, session_id: str) -> str:
        conn = self._conn()
        row = conn.execute(
            "SELECT text FROM summaries WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row["text"] if row else ""

    def set_summary(self, session_id: str, text: str):
        conn = self._conn()
        conn.execute(
            "INSERT INTO summaries (session_id, text) VALUES (?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET text = excluded.text",
            (session_id, text),
        )
        conn.commit()

    def get_summary_record(self, session_id: str) -> dict:
        row = self._conn().execute(
            "SELECT text, last_message_id FROM summaries WHERE session_id = ?", (session_id,),
        ).fetchone()
        return {"text": row["text"], "last_message_id": row["last_message_id"]} if row else {
            "text": "", "last_message_id": 0,
        }

    def set_summary_record(self, session_id: str, text: str, last_message_id: int):
        conn = self._conn()
        conn.execute(
            "INSERT INTO summaries (session_id, text, last_message_id) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET text = excluded.text, "
            "last_message_id = excluded.last_message_id "
            "WHERE excluded.last_message_id >= summaries.last_message_id",
            (session_id, text, last_message_id),
        )
        conn.commit()

    # ── 跨会话 KV 记忆（Layer1 持久化存储）──────────────────────────

    def save_memory(self, memory_id: str, content: str):
        conn = self._conn()
        from datetime import datetime
        conn.execute(
            "INSERT INTO memories (id, content, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET content = excluded.content",
            (memory_id, content, datetime.now().isoformat()),
        )
        conn.commit()

    def list_memories(self) -> list[dict]:
        conn = self._conn()
        rows = conn.execute("SELECT id, content, created_at FROM memories ORDER BY created_at").fetchall()
        return [{"id": row["id"], "content": row["content"], "created_at": row["created_at"]} for row in rows]

    def delete_memory(self, memory_id: str):
        conn = self._conn()
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()

    # ── 关闭连接 ──────────────────────────────────────────────────────

    def close(self):
        """关闭当前线程的数据库连接。"""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
