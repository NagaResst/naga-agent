import sqlite3
import tempfile
import unittest
from pathlib import Path

from session.sqlite_session import SessionManager


class SessionSchemaTests(unittest.TestCase):
    def test_new_database_initializes_and_reopens(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "new.db")
            manager = SessionManager(db_path)
            columns = {row["name"] for row in manager._conn().execute("PRAGMA table_info(messages)")}
            self.assertIn("metadata", columns)
            session_id = manager.create_session("new")
            manager.append_message(session_id, "user", "hello")
            manager.close()

            reopened = SessionManager(db_path)
            self.assertEqual(reopened.get_history(session_id)[0]["content"], "hello")
            first = reopened.get_history_since(session_id)
            reopened.set_summary_record(session_id, "[source:1] hello", first[0]["id"])
            self.assertEqual(reopened.get_history_since(session_id, first[0]["id"]), [])
            self.assertEqual(reopened.get_summary_record(session_id)["last_message_id"], first[0]["id"])
            reopened.clear_messages(session_id)
            self.assertEqual(reopened.get_summary_record(session_id)["last_message_id"], 0)
            self.assertEqual(reopened.get_history_since(session_id), [])
            reopened.close()

    def test_old_messages_table_migrates_without_losing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "old.db")
            connection = sqlite3.connect(db_path)
            connection.execute(
                "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, role TEXT NOT NULL, "
                "content TEXT NOT NULL, timestamp TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                ("old", "user", "kept", "2026-01-01"),
            )
            connection.execute("CREATE TABLE summaries (session_id TEXT PRIMARY KEY, text TEXT NOT NULL DEFAULT '')")
            connection.execute("INSERT INTO summaries (session_id, text) VALUES ('old', 'legacy')")
            connection.commit()
            connection.close()

            manager = SessionManager(db_path)
            self.assertEqual(manager.get_history("old")[0]["content"], "kept")
            manager.append_message("old", "assistant", "reply", {"tool_calls": [{"id": "1"}]})
            self.assertEqual(manager.get_history("old")[1]["tool_calls"][0]["id"], "1")
            self.assertEqual(manager.get_summary_record("old"), {"text": "legacy", "last_message_id": 0})
            manager.set_summary_record("old", "new", manager.get_history_since("old")[0]["id"])
            manager.close()

            reopened = SessionManager(db_path)
            self.assertEqual(len(reopened.get_history("old")), 2)
            reopened.close()

    def test_tool_call_batch_roundtrips_in_order_after_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "tools.db")
            manager = SessionManager(path)
            session_id = manager.create_session("tools")
            calls = [{"id": "call1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
            manager.append_messages(session_id, [
                {"role": "assistant", "content": None, "tool_calls": calls},
                {"role": "tool", "tool_call_id": "call1", "name": "read_file", "content": "result"},
            ])
            manager.close()
            reopened = SessionManager(path)
            history = reopened.get_history_since(session_id)
            self.assertEqual([row["role"] for row in history], ["assistant", "tool"])
            self.assertEqual(history[0]["tool_calls"][0]["id"], history[1]["tool_call_id"])
            self.assertEqual(history[1]["content"], "result")
            reopened.close()


if __name__ == "__main__":
    unittest.main()