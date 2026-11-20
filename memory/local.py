import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime


def _terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.casefold()):
        if "\u4e00" <= token[0] <= "\u9fff" and len(token) > 1:
            terms.update(token[i:i + 2] for i in range(len(token) - 1))
        else:
            terms.add(token)
    return terms


class LocalMemoryStore:
    def __init__(self, path: str):
        self.path = path
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS core_memories "
                "(id TEXT PRIMARY KEY, content TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS episodic_memories "
                "(id TEXT PRIMARY KEY, content TEXT NOT NULL, created_at TEXT NOT NULL)"
            )

    def list_memories(self) -> list[dict]:
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute(
                "SELECT id, content, created_at FROM core_memories ORDER BY rowid"
            ).fetchall()
        return [{"id": row[0], "content": row[1], "created_at": row[2]} for row in rows]

    def save_memory(self, memory_id: str, content: str):
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(
                "INSERT INTO core_memories VALUES (?, ?, ?)",
                (memory_id, content, datetime.now().isoformat()),
            )

    def delete_memory(self, memory_id: str):
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("DELETE FROM core_memories WHERE id = ?", (memory_id,))

    def save_episodic(self, memory_id: str, content: str):
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(
                "INSERT INTO episodic_memories VALUES (?, ?, ?)",
                (memory_id, content, datetime.now().isoformat()),
            )

    def delete_episodic(self, memory_id: str) -> bool:
        with closing(sqlite3.connect(self.path)) as conn, conn:
            cursor = conn.execute("DELETE FROM episodic_memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    def list_episodic(self, top_k: int) -> list[dict]:
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute(
                "SELECT id, content FROM episodic_memories ORDER BY rowid DESC LIMIT ?", (max(0, top_k),)
            ).fetchall()
        return [{"id": row[0], "content": row[1], "score": 0.0} for row in rows]

    def recall_episodic(self, query: str, top_k: int) -> list[dict]:
        query_terms = _terms(query)
        if not query_terms or top_k <= 0:
            return []
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute(
                "SELECT id, content FROM episodic_memories ORDER BY rowid DESC"
            ).fetchall()
        results = []
        for memory_id, content in rows:
            overlap = len(query_terms & _terms(content))
            if overlap:
                results.append({"id": memory_id, "content": content, "score": overlap / len(query_terms)})
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]
