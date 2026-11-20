import os
import sqlite3
import tempfile
import tomllib
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from memory.manager import MemoryManager, _build_mem0_config
from agent.core import Agent
from session.sqlite_session import SessionManager
from tools import memory as memory_tool


class RemoteMemoryTests(unittest.TestCase):
    @staticmethod
    def _manager():
        manager = MemoryManager.__new__(MemoryManager)
        manager._activity = None
        manager._cfg = {"reranker": {
            "model": "Qwen3-Reranker-8B",
            "base_url": "https://example.test/v1",
            "timeout": 2,
        }}
        manager._api_key = "main-placeholder"
        manager._mem0 = Mock()
        manager._mem0.search.return_value = [
            {"memory": "first", "metadata": {"memory_id": "a"}, "score": 0.8},
            {"memory": "second", "metadata": {"memory_id": "b"}, "score": 0.5},
        ]
        return manager

    def test_config_uses_remote_embeddings_and_separate_collection(self):
        config = tomllib.loads((Path(__file__).parent.parent / "config.toml").read_text(encoding="utf-8"))
        with patch.dict(os.environ, {"VECTOR_API_KEY": "vector-placeholder"}):
            mem0_config = _build_mem0_config(config["memory"], "main-placeholder", config["model"]["base_url"])

        self.assertEqual(mem0_config.embedder.config["model"], "Qwen3-Embedding-8B")
        self.assertEqual(mem0_config.embedder.config["api_key"], "vector-placeholder")
        self.assertEqual(mem0_config.vector_store.config.embedding_model_dims, 4096)
        self.assertNotEqual(mem0_config.vector_store.config.collection_name, "mem0")
        self.assertEqual(config["embedding_daemon"]["script_path"], "")

    def test_recall_reranks_by_returned_document_index(self):
        manager = self._manager()
        reply = Mock()
        reply.json.return_value = {"results": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.1},
        ]}
        with patch.dict(os.environ, {"VECTOR_API_KEY": "vector-placeholder"}), \
             patch("memory.manager.requests.post", return_value=reply) as post:
            ranked = manager.recall_episodic("query", top_k=2)

        self.assertEqual([item["id"] for item in ranked], ["b", "a"])
        self.assertEqual([item["score"] for item in ranked], [0.95, 0.1])
        self.assertEqual(post.call_args.args[0], "https://example.test/v1/rerank")
        self.assertEqual(post.call_args.kwargs["json"]["documents"], ["first", "second"])
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer vector-placeholder")
        self.assertEqual(manager._mem0.search.call_args.kwargs["filters"], {"agent_id": "naga-agent"})

    def test_save_list_and_delete_share_scope(self):
        manager = self._manager()
        manager._cfg["agent_id"] = "my-agent"
        manager._mem0.get_all.return_value = {"results": manager._mem0.search.return_value}
        manager._mem0.search.return_value = [
            {"id": "stored-id", "memory": "first", "metadata": {"memory_id": "a"}},
        ]

        manager.save_episodic("fact")
        self.assertEqual(manager._mem0.add.call_args.kwargs["agent_id"], "my-agent")
        self.assertEqual([item["id"] for item in manager.list_episodic()], ["a", "b"])
        manager._mem0.get_all.assert_called_once_with(filters={"agent_id": "my-agent"}, top_k=20)
        self.assertEqual(manager.delete_episodic("a"), "已删除：a")
        self.assertEqual(manager._mem0.search.call_args.kwargs["filters"],
                         {"agent_id": "my-agent", "memory_id": "a"})
        manager._mem0.delete.assert_called_once_with("stored-id")

    def test_delete_fallback_keeps_scope(self):
        manager = self._manager()
        manager._mem0.search.side_effect = [
            {"results": []},
            {"results": [{"id": "stored-id", "metadata": {"memory_id": "a"}}]},
        ]

        self.assertEqual(manager.delete_episodic("a"), "已删除：a")
        self.assertEqual(manager._mem0.search.call_args.kwargs["filters"], {"agent_id": "naga-agent"})
        manager._mem0.delete.assert_called_once_with("stored-id")

    def test_async_extraction_uses_shared_scope(self):
        manager = self._manager()
        with patch("memory.manager.threading.Thread") as thread:
            manager.extract_async([{"role": "user", "content": "hello"}])

        self.assertEqual(thread.call_args.kwargs["kwargs"]["agent_id"], "naga-agent")
        thread.return_value.start.assert_called_once_with()

    def test_initialization_suppresses_only_qdrant_version_warning(self):
        def fake_memory(*, config):
            warnings.warn("Qdrant client version 1.19.0 is incompatible with server version 1.17.1.", UserWarning)
            warnings.warn("another warning", UserWarning)
            return Mock()

        with patch("mem0.Memory", side_effect=fake_memory), \
             patch("memory.manager._build_mem0_config", return_value=Mock()), \
             warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager = MemoryManager({}, storage=None)

        self.assertTrue(manager.available)
        self.assertEqual([str(item.message) for item in caught], ["another warning"])

    def test_rerank_failure_keeps_vector_order(self):
        manager = self._manager()
        with patch("memory.manager.requests.post", side_effect=requests.Timeout), \
             patch("builtins.print"):
            results = manager.recall_episodic("query")

        self.assertEqual([item["id"] for item in results], ["a", "b"])
        self.assertEqual([item["score"] for item in results], [0.8, 0.5])

    def test_unexpected_rerank_payload_keeps_vector_order(self):
        manager = self._manager()
        reply = Mock()
        reply.json.return_value = ["unexpected"]
        with patch("memory.manager.requests.post", return_value=reply):
            results = manager.recall_episodic("query")

        self.assertEqual([item["id"] for item in results], ["a", "b"])

    def test_single_candidate_does_not_call_reranker(self):
        manager = self._manager()
        manager._mem0.search.return_value = manager._mem0.search.return_value[:1]
        with patch("memory.manager.requests.post") as post:
            self.assertEqual(manager.recall_episodic("query")[0]["id"], "a")
        post.assert_not_called()


class LocalMemoryTests(unittest.TestCase):
    def test_local_memory_persists_and_searches_without_vector_services(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "memory.db")
            storage = Mock()
            config = {"backend": "local", "local_path": path, "extract_every_n_turns": 1}
            with patch("mem0.Memory") as mem0, patch("memory.manager.requests.post") as post:
                manager = MemoryManager(config, storage=storage)
                self.assertEqual(os.stat(path).st_mode & 0o077, 0)
                core_id = manager.save_core("喜欢中文").split(": ")[1]
                episodic_id = manager.save_episodic("PostgreSQL 数据库迁移方案").split(": ")[1]
                manager.save_episodic("部署 Redis 缓存")
                manager.increment_turn()
                self.assertFalse(manager.should_extract())
                reopened = MemoryManager(config, storage=storage)
                self.assertEqual(reopened.list_core()[0]["id"], core_id)
                self.assertEqual(reopened.recall_episodic("检查数据库 PostgreSQL")[0]["id"], episodic_id)
                self.assertEqual(reopened.recall_episodic("完全无关"), [])
                self.assertEqual(len(reopened.list_episodic(top_k=1)), 1)
                self.assertEqual(reopened.delete_episodic(episodic_id), f"已删除：{episodic_id}")
                self.assertEqual(reopened.delete_core(core_id), f"已删除：{core_id}")
                self.assertEqual(MemoryManager(config).list_core(), [])
                self.assertEqual(MemoryManager(config).recall_episodic("PostgreSQL"), [])
            mem0.assert_not_called()
            post.assert_not_called()
            storage.list_memories.assert_not_called()

    def test_local_storage_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.db"
            path.write_text("not a database", encoding="utf-8")
            with self.assertRaises(sqlite3.DatabaseError):
                MemoryManager({"backend": "local", "local_path": str(path)})


class MemoryStartupTests(unittest.TestCase):
    def test_disabled_and_local_skip_daemon_and_external_memory(self):
        self.addCleanup(memory_tool.set_memory_manager, None)
        with tempfile.TemporaryDirectory() as directory:
            storage = SessionManager(str(Path(directory) / "session.db"))
            self.addCleanup(storage.close)
            session_id = storage.create_session("test")
            storage.save_memory("old", "old core preference")
            storage.append_message(session_id, "user", "new request")
            config = {
                "api_key": "test", "model": {"default": "test"}, "delegation": {},
                "agent": {"auto_confirm": False, "max_history": 10, "context_token_limit": 4096},
                "embedding_daemon": {"script_path": __file__},
                "memory": {"enabled": False, "backend": "local", "local_path": str(Path(directory) / "memory.db")},
            }
            token_tracker = Mock()
            token_tracker.count_tokens.side_effect = len
            token_tracker.estimate.side_effect = lambda messages: sum(len(str(item)) for item in messages)
            token_tracker.estimate_tools.return_value = 0
            token_tracker.mode = "test"
            with patch("agent.core.OpenAI"), patch("agent.core.TokenTracker", return_value=token_tracker), \
                 patch("agent.core.discover_skills", return_value=[]), \
                 patch("agent.core.get_context_window", return_value=4096), \
                 patch("agent.core.subprocess.Popen") as daemon, patch("mem0.Memory") as mem0, \
                 patch("memory.manager.requests.post") as post:
                agent = Agent(config, storage, session_id, "test", Mock())
                self.assertIsNone(agent._memory_manager)
                self.assertNotIn("memory", [tool["function"]["name"] for tool in agent._select_tools("")])
                self.assertNotIn("memory", agent.tool_executors)
                self.assertNotIn("old core preference", str(agent._get_messages()))
                self.assertEqual(memory_tool.execute({}), "错误：记忆系统未初始化")
                self.assertFalse(Path(config["memory"]["local_path"]).exists())

                config["memory"]["enabled"] = True
                agent = Agent(config, storage, session_id, "test", Mock())
                self.assertTrue(agent._memory_manager.available)
                self.assertEqual(agent._memory_manager.list_core(), [])
                self.assertIn("memory", [tool["function"]["name"] for tool in agent._select_tools("")])
                agent._memory_manager.save_core("new local preference")
                agent._memory_manager.save_episodic("request guidance")
                messages = str(agent._get_messages())
                self.assertIn("new local preference", messages)
                self.assertIn("request guidance", messages)
                self.assertNotIn("old core preference", messages)
                daemon.assert_not_called()
                mem0.assert_not_called()
                post.assert_not_called()


if __name__ == "__main__":
    unittest.main()