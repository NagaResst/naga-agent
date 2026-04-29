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
                text       TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS memories (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT ''
            );
        """)
        conn.commit()

    # ── 会话管理 ──────────────────────────────────────────────────────────

    def list_sessions(self) -> list:
        conn = self._conn()
        rows = conn.execute(
            "SELECT id, name, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        result = []
        for row in rows:
            msg_count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (row["id"],)
            ).fetchone()[0]
            result.append({
                "id": row["id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "message_count": msg_count,
            })
        return result

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
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in rows]

    def append_message(self, session_id: str, role: str, content: str):
        conn = self._conn()
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now().isoformat()),
        )
        conn.commit()

    def clear_messages(self, session_id: str):
        conn = self._conn()
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
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

    # ── 跨会话 KV 记忆（Layer1 持久化存储）──────────────────────────

    def set_memory(self, key: str, value: str):
        conn = self._conn()
        conn.execute(
            "INSERT INTO memories (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()

    def get_memory(self, key: str) -> str | None:
        conn = self._conn()
        row = conn.execute("SELECT value FROM memories WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def list_memories(self) -> dict:
        conn = self._conn()
        rows = conn.execute("SELECT key, value FROM memories").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def delete_memory(self, key: str):
        conn = self._conn()
        conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        conn.commit()

    # ── 关闭连接 ──────────────────────────────────────────────────────

    def close(self):
        """关闭当前线程的数据库连接。"""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
