import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rich.console import Console
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text

from agent.core import Agent
from agent.status import ActivityTracker
from agent.skill_registry import discover_skills
from agent.subagent import SubtaskRunner
from agent.token_tracker import TokenTracker
from main import OutputPane
from session.sqlite_session import SessionManager


def response(content=None, calls=None, usage=None):
    message = SimpleNamespace(content=content, tool_calls=calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def tool_call(name, arguments="{}"):
    return SimpleNamespace(id="child-call", function=SimpleNamespace(name=name, arguments=arguments))


class SubtaskRunnerTests(unittest.TestCase):
    def test_child_budget_truncates_tool_output_but_keeps_call_pair(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            response(calls=[tool_call("read_file")]), response("done"),
        ]
        original = "large output " * 500
        runner = SubtaskRunner(client, "child", [{"type": "function", "function": {"name": "read_file"}}],
                               {"read_file": Mock(return_value=original)}, Mock(), context_token_limit=1300)

        self.assertEqual(runner.execute({"tasks": [{"task": "read this"}]}).count('"status": "ok"'), 1)
        sent = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertEqual(sent[-2]["tool_calls"][0]["id"], sent[-1]["tool_call_id"])
        self.assertLess(len(sent[-1]["content"]), len(original))

    def test_isolated_tasks_use_child_model_and_all_registered_tools(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            response(calls=[tool_call("read_file")], usage={"prompt_tokens": 2, "completion_tokens": 1}),
            response("first result"),
            response("second result"),
        ]
        read_file = Mock(return_value="source text")
        execute_command = Mock()
        tool_dispatcher = Mock(return_value="source text")
        record_usage = Mock()
        definitions = [
            {"type": "function", "function": {"name": name}}
            for name in ("read_file", "execute_command", "memory", "skill_manager", "mcp", "delegate_tasks")
        ]
        runner = SubtaskRunner(
            client, "cheap-model", definitions,
            {"read_file": read_file, "execute_command": execute_command, "memory": Mock(),
             "skill_manager": Mock(), "mcp": Mock(), "delegate_tasks": Mock()},
            record_usage, tool_dispatcher=tool_dispatcher,
        )

        result = json.loads(runner.execute({"tasks": [{"task": "first"}, {"task": "second"}]}))

        self.assertEqual([item["result"] for item in result["results"]], ["first result", "second result"])
        self.assertEqual(client.chat.completions.create.call_count, 3)
        requests = client.chat.completions.create.call_args_list
        self.assertTrue(all(call.kwargs["model"] == "cheap-model" for call in requests))
        self.assertEqual(
            [item["function"]["name"] for item in requests[0].kwargs["tools"]],
            ["read_file", "execute_command", "memory", "skill_manager", "mcp"],
        )
        self.assertEqual(len(requests[2].kwargs["messages"]), 2)
        self.assertEqual(requests[1].kwargs["messages"][-1]["content"], "source text")
        tool_dispatcher.assert_called_once_with("read_file", {})
        read_file.assert_not_called()
        execute_command.assert_not_called()
        record_usage.assert_called_once_with("cheap-model", {"prompt_tokens": 2, "completion_tokens": 1})

    def test_rejects_recursive_calls_and_respects_limits(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            response(calls=[tool_call("delegate_tasks")]), response("cannot delegate recursively"),
        ]
        delegate_tasks = Mock()
        runner = SubtaskRunner(
            client, "cheap",
            [{"type": "function", "function": {"name": "delegate_tasks"}}],
            {"delegate_tasks": delegate_tasks}, Mock(), max_tasks=1,
        )

        self.assertTrue(runner.execute({"tasks": [{"task": "one"}, {"task": "two"}]}).startswith("错误："))
        result = json.loads(runner.execute({"tasks": [{"task": "one"}]}))
        self.assertEqual(result["results"][0]["status"], "ok")
        self.assertNotIn("tools", client.chat.completions.create.call_args_list[1].kwargs)
        self.assertEqual(
            client.chat.completions.create.call_args_list[1].kwargs["messages"][-1]["content"],
            "错误：子 Agent 不允许调用 delegate_tasks",
        )
        delegate_tasks.assert_not_called()

    def test_child_error_does_not_cancel_other_tasks(self):
        client = Mock()
        client.chat.completions.create.side_effect = [RuntimeError("child unavailable"), response("second ok")]
        runner = SubtaskRunner(client, "cheap", [], {}, Mock())
        result = json.loads(runner.execute({"tasks": [{"task": "first"}, {"task": "second"}]}))
        self.assertEqual([item["status"] for item in result["results"]], ["error", "ok"])

    def test_child_round_limit_rejects_unexpected_tool_call(self):
        client = Mock()
        client.chat.completions.create.return_value = response(calls=[tool_call("read_file")])
        read_file = Mock()
        runner = SubtaskRunner(client, "cheap", [], {"read_file": read_file}, Mock(), max_rounds=0)

        result = json.loads(runner.execute({"tasks": [{"task": "inspect"}]}))

        self.assertEqual(result["results"][0]["status"], "error")
        self.assertIn("上限", result["results"][0]["result"])
        read_file.assert_not_called()
        self.assertEqual(client.chat.completions.create.call_count, 1)


class MainAgentDelegationTests(unittest.TestCase):
    def test_expired_tool_results_do_not_trigger_compression(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SessionManager(str(Path(directory) / "short-tools.db"))
            session_id = manager.create_session("tools")
            for number in range(4):
                manager.append_message(session_id, "user", f"request {number}")
                manager.append_messages(session_id, [
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": f"call-{number}", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}},
                    ]},
                    {"role": "tool", "tool_call_id": f"call-{number}", "content": "result " * 100},
                ])
                manager.append_message(session_id, "assistant", f"answer {number}")
            manager.append_message(session_id, "user", "latest")
            agent = Agent.__new__(Agent)
            agent.activity = ActivityTracker()
            agent.config = {"agent": {"system_prompt": "rules", "context_token_limit": 3000,
                                      "tool_result_keep_rounds": 1}, "delegation": {}}
            agent.model = "main"
            agent.session_id = session_id
            agent.session_manager = manager
            agent._client = Mock()
            agent._token_tracker = Mock()
            agent._token_tracker.count_tokens.side_effect = len
            agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(item)) for item in messages)
            agent._token_tracker.estimate_tools.return_value = 0
            agent._memory_manager = None
            agent._subtask_runner = None
            agent._session_theme = ""
            agent._skills = []
            agent.max_history = 100
            agent._sync_active_skills = Mock()
            agent._select_tools = Mock(return_value=[])

            messages = agent._get_messages()

            self.assertEqual(manager.get_summary_record(session_id)["last_message_id"], 0)
            self.assertEqual([msg["tool_call_id"] for msg in messages if msg["role"] == "tool"], ["call-3"])
            self.assertEqual(messages[-1]["content"], "latest")
            self.assertEqual(len([msg for msg in manager.get_history_since(session_id) if msg["role"] == "tool"]), 4)
            manager.close()

    def test_old_tool_batches_are_pruned_without_orphaned_replies(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.config = {"agent": {"tool_result_keep_rounds": 2}}
        agent.max_history = 100
        agent._token_tracker = Mock()
        agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(item)) for item in messages)
        agent._token_tracker.estimate_tools.return_value = 0
        messages = [{"role": "system", "content": "rules"}]
        for number in range(4):
            messages.extend([
                {"role": "user", "content": f"request {number}"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"call-{number}", "function": {"name": "read_file"}},
                ]},
                {"role": "tool", "tool_call_id": f"call-{number}", "content": f"result {number}"},
                {"role": "assistant", "content": f"answer {number}"},
            ])

        fitted = agent._fit_messages(messages, 10000, None)

        self.assertEqual([message["tool_call_id"] for message in fitted if message["role"] == "tool"],
                         ["call-2", "call-3"])
        self.assertEqual([message["content"] for message in fitted if message["role"] == "user"],
                         [f"request {number}" for number in range(4)])
        self.assertEqual([message["content"] for message in fitted if message["role"] == "assistant"
                          and not message.get("tool_calls")], [f"answer {number}" for number in range(4)])
        self.assertEqual(len([message for message in messages if message["role"] == "tool"]), 4)

    def test_all_tool_results_from_current_turn_are_preserved(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.config = {"agent": {"tool_result_keep_rounds": 2}}
        agent.max_history = 100
        agent._token_tracker = Mock()
        agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(item)) for item in messages)
        agent._token_tracker.estimate_tools.return_value = 0
        messages = [{"role": "system", "content": "rules"}]
        for number in range(5):
            messages.extend([
                {"role": "user", "content": f"old request {number}"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"old-call-{number}", "function": {"name": "read_file"}},
                ]},
                {"role": "tool", "tool_call_id": f"old-call-{number}", "content": f"old result {number}"},
            ])
        messages.append({"role": "user", "content": "current request"})
        for number in range(5):
            messages.extend([
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"current-call-{number}", "function": {"name": "read_file"}},
                ]},
                {"role": "tool", "tool_call_id": f"current-call-{number}", "content": f"current result {number}"},
            ])

        fitted = agent._fit_messages(messages, 10000, None)

        results = [message["content"] for message in fitted if message["role"] == "tool"]
        self.assertEqual(results, [
            "current result 0", "current result 1", "current result 2", "current result 3", "current result 4",
        ])

    def test_assembled_context_restores_summary_after_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "context.db")
            manager = SessionManager(path)
            session_id = manager.create_session("context")
            manager.append_message(session_id, "user", "old " * 400)
            manager.append_message(session_id, "assistant", "old answer " * 80)
            manager.append_message(session_id, "user", "latest")

            def assemble(store):
                agent = Agent.__new__(Agent)
                agent.activity = ActivityTracker()
                agent.config = {"agent": {"system_prompt": "rules", "context_token_limit": 2000}, "delegation": {}}
                agent.model = "main"
                agent.session_id = session_id
                agent.session_manager = store
                agent._client = Mock()
                agent._token_tracker = Mock()
                agent._token_tracker.count_tokens.side_effect = len
                agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(item)) for item in messages)
                agent._token_tracker.estimate_tools.return_value = 0
                agent._memory_manager = None
                agent._subtask_runner = None
                agent._session_theme = ""
                agent._skills = []
                agent.max_history = 20
                agent._sync_active_skills = Mock()
                agent._select_tools = Mock(return_value=[])
                return agent._get_messages()

            first = assemble(manager)
            summary = manager.get_summary_record(session_id)
            self.assertGreater(summary["last_message_id"], 0)
            self.assertEqual(first[-1]["content"], "latest")
            manager.close()
            reopened = SessionManager(path)
            second = assemble(reopened)
            self.assertEqual(second, first)
            record = reopened.get_summary_record(session_id)
            reopened.set_summary_record(session_id, record["text"] + "\n[消息 999 / tool] SECRET",
                                        record["last_message_id"])
            smaller = assemble(reopened)
            self.assertNotIn("SECRET", str(smaller))
            reopened.close()

    def test_summary_survives_history_trimming(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent._token_tracker = Mock()
        agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(item)) for item in messages)
        agent._token_tracker.estimate_tools.return_value = 0
        agent.max_history = 2
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "【历史摘要：参考数据】source"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "latest"},
        ]

        fitted = agent._fit_messages(messages, 4096, None)

        self.assertIn(messages[1], fitted)
        self.assertEqual(fitted[-1]["content"], "latest")
        self.assertNotIn(messages[2], fitted)

    def test_compression_persists_complete_turns_with_source_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "history.db")
            manager = SessionManager(path)
            session_id = manager.create_session("long")
            manager.append_message(session_id, "user", "old request " * 20)
            manager.append_message(session_id, "assistant", "", {"tool_calls": [
                {"id": "call1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            ]})
            manager.append_message(session_id, "tool", "old tool result " * 20, {"tool_call_id": "call1"})
            manager.append_message(session_id, "assistant", "old reply " * 20)
            manager.append_message(session_id, "user", "current request")
            original = manager.get_history_since(session_id)
            agent = Agent.__new__(Agent)
            agent.activity = ActivityTracker()
            agent.config = {"agent": {"context_compact_ratio": 0.75}}
            agent.session_manager = manager
            agent.session_id = session_id
            agent._token_tracker = Mock()
            agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(item)) for item in messages)

            kept, summary = agent._compact_history(original, {"text": "", "last_message_id": 0}, [], 500)

            self.assertEqual(kept, original[-1:])
            self.assertEqual(summary["last_message_id"], original[3]["id"])
            self.assertIn("[消息", summary["text"])
            self.assertNotIn("old tool result", summary["text"])
            self.assertNotIn("read_file", summary["text"])
            manager.close()
            reopened = SessionManager(path)
            self.assertEqual(reopened.get_summary_record(session_id), summary)
            self.assertEqual(reopened.get_history_since(session_id, summary["last_message_id"]), kept)
            self.assertEqual(reopened.get_history_since(session_id), original)
            reopened.close()

    def test_compression_keeps_unfinished_tool_turn(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.config = {"agent": {"context_compact_ratio": 0.75}}
        agent.session_manager = Mock()
        agent.session_id = "test"
        agent._token_tracker = Mock()
        agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(item)) for item in messages)
        rows = [
            {"id": 1, "role": "user", "content": "old" * 500},
            {"id": 2, "role": "assistant", "content": "", "tool_calls": [
                {"id": "call1", "function": {"name": "read_file"}},
            ]},
            {"id": 3, "role": "tool", "tool_call_id": "call1", "content": "result"},
            {"id": 4, "role": "user", "content": "new"},
        ]

        kept, summary = agent._compact_history(rows, {"text": "", "last_message_id": 0}, [], 500)

        self.assertEqual(kept, rows)
        self.assertEqual(summary["last_message_id"], 0)
        agent.session_manager.set_summary_record.assert_not_called()

    def test_history_cache_only_reads_new_message_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SessionManager(str(Path(directory) / "cache.db"))
            session_id = manager.create_session("cache")
            manager.append_message(session_id, "user", "first")
            agent = Agent.__new__(Agent)
            agent.activity = ActivityTracker()
            agent.session_manager = manager
            agent.session_id = session_id
            manager.get_history_since = Mock(wraps=manager.get_history_since)

            first, summary = agent._load_session_history(manager.get_summary_record(session_id))
            manager.append_message(session_id, "assistant", "second")
            updated, _ = agent._load_session_history(summary)

            self.assertEqual(len(updated), 2)
            self.assertEqual(manager.get_history_since.call_args.args[1], first[-1]["id"])
            manager.close()

    def test_skill_match_uses_frontmatter_and_forced_selection(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.config = {"skills": {"min_match_score": 2.2, "match_top_k": 1}}
        agent._session_theme = ""
        agent._skills = [
            {"name": "coding", "enabled": True, "keywords": ["python"], "examples": []},
            {"name": "testing", "enabled": True, "keywords": ["python", "test"], "examples": []},
            {"name": "manual", "enabled": False, "keywords": [], "examples": []},
        ]

        self.assertEqual([skill["name"] for skill in agent._match_skills("python test")], ["testing"])
        self.assertEqual([skill["name"] for skill in agent._match_skills("/manual python test")], ["testing"])
        agent._skills[2]["enabled"] = True
        self.assertEqual([skill["name"] for skill in agent._match_skills("/manual python test")],
                         ["manual"])
        self.assertEqual([skill["name"] for skill in agent._match_skills("please /manual python test")],
                         ["testing"])

    def test_all_frontmatter_and_explicit_entry_file_are_injected_in_full(self):
        with tempfile.TemporaryDirectory() as directory:
            source = "---\nname: demo\nenabled: true\n---\n" + "full instruction\n" * 180
            disabled_frontmatter = "---\nname: disabled\ndescription: " + "x" * 1200 + "\nenabled: false\n---\n"
            (Path(directory) / "demo.md").write_text(source, encoding="utf-8")
            (Path(directory) / "disabled.md").write_text(disabled_frontmatter + "not active\n", encoding="utf-8")

            agent = Agent.__new__(Agent)
            agent.activity = ActivityTracker()
            agent.config = {"agent": {"system_prompt": "rules", "context_token_limit": 10000}, "delegation": {}}
            agent.model = "main"
            agent.session_id = "test"
            agent.max_history = 10
            agent.session_manager = Mock()
            agent.session_manager.get_summary_record.return_value = None
            agent.session_manager.get_history.return_value = [{"role": "user", "content": "/demo do task"}]
            agent._client = Mock()
            agent._token_tracker = Mock()
            agent._token_tracker.count_tokens.side_effect = len
            agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(message)) for message in messages)
            agent._token_tracker.estimate_tools.return_value = 0
            agent._memory_manager = None
            agent._subtask_runner = None
            agent._session_theme = ""
            agent._skills = discover_skills(directory, [])
            agent._sync_active_skills = Mock()
            agent._select_tools = Mock(return_value=[])

            messages = agent._get_messages()
            catalog = next(message["content"] for message in messages if message["content"].startswith("【Skill frontmatter 目录"))
            explicit = next(message["content"] for message in messages if message["content"].startswith("【用户指定 Skill："))
            self.assertIn(disabled_frontmatter.strip(), catalog)
            self.assertIn("---\nname: demo\nenabled: true\n---", catalog)
            self.assertTrue(explicit.endswith(source))
            self.assertGreater(len(source), 2048)

            agent.session_manager.get_history.return_value = [{"role": "user", "content": "demo task"}]
            automatic = agent._get_messages()
            self.assertFalse(any(message["content"].startswith("【用户指定 Skill：") for message in automatic))
            self.assertTrue(any("【激活 Skill：demo】" in message["content"] for message in automatic))

    def test_oversized_explicit_skill_fails_without_truncating(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.config = {"agent": {}}
        agent.max_history = 10
        agent._token_tracker = Mock()
        agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(message)) for message in messages)
        agent._prompt_budget = Mock(return_value=100)
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "【Skill frontmatter 目录：仅供参考】\n---\nname: demo\n---"},
            {"role": "user", "content": "【用户指定 Skill：demo】\n" + "content" * 100},
            {"role": "user", "content": "/demo do task"},
        ]
        with self.assertRaisesRegex(ValueError, "SKILL.md 全文超出上下文窗口"):
            agent._fit_messages(messages, 100, [])

    def test_runtime_state_refresh_preserves_stable_system_prefix(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent._token_tracker = Mock()
        agent._build_runtime_state_block = Mock(side_effect=[
            "【固定约束与运行状态】first", "【固定约束与运行状态】second",
        ])
        messages = [{"role": "system", "content": "rules\n\nSKILL_RULES"}]

        agent._upsert_runtime_state_message(messages)
        agent._upsert_runtime_state_message(messages)

        self.assertEqual(messages, [
            {"role": "system", "content": "rules\n\nSKILL_RULES"},
            {"role": "user", "content": "【固定约束与运行状态】second"},
        ])

    def test_budget_failure_after_tool_call_returns_text(self):
        agent = self._loop_agent()
        agent._fit_messages = Mock(side_effect=[agent._get_messages.return_value, ValueError("超出上下文窗口")])
        agent._subtask_runner.execute.return_value = '{"results":[{"status":"ok","result":"found"}]}'
        delegated = SimpleNamespace(index=0, id="main-call", function=SimpleNamespace(
            name="delegate_tasks", arguments='{"tasks":[{"task":"simple"}]}'
        ))
        agent._client.chat.completions.create.return_value = self._stream(calls=[delegated], finish="tool_calls")

        self.assertEqual(agent.chat("question"), "超出上下文窗口")
        self.assertEqual(agent.session_manager.append_message.call_args_list[-1].args[2], "超出上下文窗口")

    def test_memory_uses_remaining_budget_after_tools_and_skills(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.config = {"agent": {"system_prompt": "rules", "context_token_limit": 2048}, "delegation": {}}
        agent.model = "main"
        agent.session_id = "test"
        agent.max_history = 10
        agent.session_manager = Mock()
        agent.session_manager.get_history.return_value = [{"role": "user", "content": "question"}]
        agent._client = Mock()
        agent._token_tracker = Mock()
        agent._token_tracker.count_tokens.side_effect = len
        agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(message)) for message in messages)
        agent._token_tracker.estimate_tools.return_value = 30
        agent._memory_manager = Mock()
        agent._memory_manager.available = False
        agent._memory_manager.list_core.return_value = [
            {"id": "small", "content": "fact"}, {"id": "large", "content": "x" * 2000},
        ]
        agent._subtask_runner = None
        agent._session_theme = ""
        agent._skills = []
        agent._match_skills = Mock(return_value=[{"name": "coding", "description": "coding", "keywords": []}])
        agent._sync_active_skills = Mock()
        agent._build_skill_prompt_block = Mock(return_value="SKILL_RULES")
        agent._select_tools = Mock(return_value=[{"type": "function"}])

        messages = agent._get_messages()

        self.assertNotIn("SKILL_RULES", messages[0]["content"])
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("SKILL_RULES", "\n".join(message["content"] for message in messages[1:]))
        self.assertNotIn("[small]", messages[0]["content"])
        reference = next(message for message in messages if "【参考记忆" in message["content"])
        self.assertIn("[small] fact", reference["content"])
        self.assertNotIn("[large]", reference["content"])
        self.assertEqual(reference["role"], "user")
        self.assertEqual(messages[-1]["content"], "question")

    def test_long_skill_uses_space_before_memory(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.config = {"agent": {"system_prompt": "rules", "context_token_limit": 1100}, "delegation": {}}
        agent.model = "main"
        agent.session_id = "test"
        agent.max_history = 10
        agent.session_manager = Mock()
        agent.session_manager.get_history.return_value = [{"role": "user", "content": "question"}]
        agent._client = Mock()
        agent._token_tracker = Mock()
        agent._token_tracker.count_tokens.side_effect = len
        agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(message)) for message in messages)
        agent._token_tracker.estimate_tools.return_value = 0
        agent._memory_manager = Mock()
        agent._memory_manager.available = False
        agent._memory_manager.list_core.return_value = [{"id": "fact", "content": "value"}]
        agent._subtask_runner = None
        agent._session_theme = ""
        agent._skills = []
        agent._match_skills = Mock(return_value=[{"name": "coding", "description": "coding", "keywords": []}])
        agent._sync_active_skills = Mock()
        agent._build_skill_prompt_block = Mock(return_value="S" * 250)
        agent._select_tools = Mock(return_value=[])

        messages = agent._get_messages()

        self.assertNotIn("【参考记忆", "\n".join(message["content"] for message in messages))
        self.assertEqual(messages[-1]["content"], "question")

    def test_tool_schema_cache_invalidates_when_delegation_changes(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.config = {"delegation": {"enabled": False}}
        agent.tool_definitions = [{"type": "function", "function": {"name": "read_file"}, "tags": ["local"]}]
        agent._subtask_runner = Mock()
        agent._token_tracker = Mock()
        agent._token_tracker.estimate_multiplier = 1.0
        agent._token_tracker.estimate_tools.return_value = 40

        first = agent._select_tools("first")
        self.assertIs(first, agent._select_tools("second"))
        self.assertNotIn("tags", first[0])
        agent._prompt_budget(4096, first)
        agent._prompt_budget(4096, first)
        agent._token_tracker.estimate_tools.assert_called_once_with(first)

        agent.config["delegation"]["enabled"] = True
        delegated = agent._select_tools("third")
        self.assertIsNot(delegated, first)
        self.assertEqual(delegated[-1]["function"]["name"], "delegate_tasks")
        agent.tool_definitions = [{"type": "function", "function": {"name": "memory"}}]
        self.assertEqual(agent._select_tools("fourth")[0]["function"]["name"], "memory")
        agent._token_tracker.estimate_multiplier = 1.2
        agent._prompt_budget(4096, delegated)
        agent._prompt_budget(4096, delegated)
        self.assertEqual(agent._token_tracker.estimate_tools.call_count, 2)

    @staticmethod
    def _stream(content=None, calls=None, finish="stop"):
        return [SimpleNamespace(usage=None, choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content, reasoning_content=None, tool_calls=calls or []),
            finish_reason=finish,
        )])]

    def _loop_agent(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.config = {
            "agent": {"show_thinking": False, "context_token_limit": 4096}, "model": {"extra_params": {}},
            "delegation": {"enabled": True, "max_calls": 2}, "tools": {"tool_max_rounds": 2},
        }
        agent.model = "main-model"
        agent.session_id = "test-session"
        agent.session_manager = Mock()
        agent.console = Mock()
        agent._token_tracker = Mock()
        agent._token_tracker.estimate.return_value = 1
        agent._token_tracker.estimate_tools.return_value = 1
        agent._memory_manager = None
        agent.tool_definitions = []
        agent.tool_executors = {}
        agent._subtask_runner = Mock()
        agent._snapshot_runtime_state = Mock()
        agent._reset_runtime_state = Mock()
        agent._restore_followup_state = Mock()
        agent._get_messages = Mock(return_value=[{"role": "system", "content": "main"}, {"role": "user", "content": "question"}])
        agent._upsert_runtime_state_message = Mock()
        agent._record_tool_event = Mock()
        agent._summarize_reply_conclusion = Mock(return_value="")
        agent._client = Mock()
        return agent

    def test_limit_requests_tool_free_final_answer(self):
        agent = self._loop_agent()
        agent.config["tools"]["tool_max_rounds"] = 1
        agent._subtask_runner.execute.return_value = '{"results":[{"status":"ok","result":"found"}]}'
        delegated = SimpleNamespace(index=0, id="main-call", function=SimpleNamespace(name="delegate_tasks", arguments='{"tasks":[{"task":"simple"}]}'))
        agent._client.chat.completions.create.side_effect = [
            self._stream(calls=[delegated], finish="tool_calls"), self._stream("limited summary"),
        ]

        self.assertEqual(agent.chat("question"), "limited summary")
        self.assertIsNone(agent._client.chat.completions.create.call_args_list[1].kwargs["tools"])
        self.assertEqual(agent.session_manager.append_message.call_args_list[-1].args[2], "limited summary")

    def test_chat_preserves_markdown_in_session(self):
        agent = self._loop_agent()
        answer = "**Important**\n\n```python\nprint(1)\n```"
        agent._client.chat.completions.create.return_value = self._stream(answer)

        self.assertEqual(agent.chat("question"), answer)
        self.assertEqual(agent.session_manager.append_message.call_args_list[-1].args[2], answer)

    def test_chat_renders_streamed_reply_without_duplicate_or_losing_logs(self):
        pane = OutputPane(Mock())
        console = Console(file=pane, width=80, force_terminal=True, color_system="standard")
        agent = self._loop_agent()
        agent.console = console
        console.print("tool log")
        answer = "**Important**\n\n- first\n\n```python\nprint(1)\n```\n" + " filler" * 20
        answer += "\n\n## Phase 1\n\n| Name | Value |\n| --- | --- |\n| A | B |\n"

        def stream():
            yield self._stream(answer[:125])[0]
            self.assertIn("Important", pane._text)
            self.assertNotIn("**Important**", pane._text)
            queued_message = threading.Thread(target=lambda: console.print("queued user message"))
            queued_message.start()
            queued_message.join()
            yield self._stream(answer[125:])[0]
            self.assertNotIn("## Phase 1", pane._text)
            self.assertIn("queued user message", pane._text)

        agent._client.chat.completions.create.return_value = stream()
        # chat 流程含 activity.set 的额外 time.monotonic 调用（prepare/llm_wait/llm_stream/idle），
        # 用生成器按需供值：前半 1.0，后半 1.3，避免固定列表耗尽
        clock = iter([1.0, 1.0, 1.0, 1.0, 1.0, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3])
        with patch("agent.core.time.monotonic", side_effect=lambda: next(clock)):
            self.assertEqual(agent.chat("question"), answer)
        displayed = fragment_list_to_text(to_formatted_text(pane.formatted_text()))
        self.assertIn("tool log", displayed)
        self.assertIn("queued user message", displayed)
        self.assertEqual(displayed.count("Important"), 1)
        self.assertNotIn("**Important**", displayed)
        self.assertNotIn("```", displayed)
        self.assertNotIn("## Phase 1", displayed)
        self.assertNotIn("| --- |", displayed)
        self.assertIn("print(1)", displayed)

    def test_chat_renders_markdown_fence_with_surrounding_text(self):
        pane = OutputPane(Mock())
        agent = self._loop_agent()
        agent.console = Console(file=pane, width=80, force_terminal=True, color_system="standard")
        answer = "Here is the plan:\n\n```markdown\n## Phase 1\n\n**Done**\n\n| Name | Value |\n| --- | --- |\n| A | B |\n```\n\nNext step."

        def stream():
            yield self._stream(answer[:58])[0]
            self.assertNotIn("## Phase 1", pane._text)
            yield self._stream(answer[58:])[0]

        agent._client.chat.completions.create.return_value = stream()
        self.assertEqual(agent.chat("question"), answer)
        displayed = fragment_list_to_text(to_formatted_text(pane.formatted_text()))
        self.assertIn("Here is the plan:", displayed)
        self.assertIn("Next step.", displayed)
        self.assertIn("Phase 1", displayed)
        self.assertNotIn("## Phase 1", displayed)
        self.assertNotIn("**Done**", displayed)
        self.assertNotIn("| --- |", displayed)

    def test_limit_never_persists_empty_reply(self):
        agent = self._loop_agent()
        agent.config["tools"]["tool_max_rounds"] = 1
        agent._subtask_runner.execute.return_value = '{"results":[{"status":"ok","result":"found"}]}'
        delegated = SimpleNamespace(index=0, id="main-call", function=SimpleNamespace(name="delegate_tasks", arguments='{"tasks":[{"task":"simple"}]}'))
        agent._client.chat.completions.create.side_effect = [
            self._stream(calls=[delegated], finish="tool_calls"), self._stream(),
        ]

        reply = agent.chat("question")

        self.assertIn("工具调用已达限制", reply)
        self.assertEqual(agent.session_manager.append_message.call_args_list[-1].args[2], reply)
        self.assertIsNone(agent._client.chat.completions.create.call_args_list[1].kwargs["tools"])

    def test_failed_delegation_and_tool_errors_are_recognized(self):
        self.assertTrue(Agent._tool_result_failed("delegate_tasks", '{"results":[{"status":"error"}]}'))
        self.assertTrue(Agent._tool_result_failed("mcp", "错误：MCP server 请求失败"))
        self.assertTrue(Agent._tool_result_failed("memory", "[memory 参数错误] invalid"))
        self.assertTrue(Agent._tool_result_failed("execute_command", "[命令执行结果]\n状态：失败（返回码 1）"))
        self.assertTrue(Agent._tool_result_failed("execute_command", "[命令执行结果]\n状态：超时（>30 秒）已终止"))
        self.assertFalse(Agent._tool_result_failed("execute_command", "[命令执行结果]\n状态：成功"))
        self.assertFalse(Agent._tool_result_failed("delegate_tasks", '{"results":[{"status":"ok"}]}'))

    def test_error_budget_disables_tools_for_final_answer(self):
        agent = self._loop_agent()
        agent.config["tools"]["tool_max_errors"] = 1
        agent._subtask_runner.execute.return_value = '{"results":[{"status":"error","result":"failed"}]}'
        delegated = SimpleNamespace(index=0, id="main-call", function=SimpleNamespace(name="delegate_tasks", arguments='{"tasks":[{"task":"simple"}]}'))
        agent._client.chat.completions.create.side_effect = [
            self._stream(calls=[delegated], finish="tool_calls"), self._stream("task failed"),
        ]

        self.assertEqual(agent.chat("question"), "task failed")
        self.assertIsNone(agent._client.chat.completions.create.call_args_list[1].kwargs["tools"])

    def test_budget_rejects_oversized_latest_user(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.max_history = 10
        agent._token_tracker = Mock()
        agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(message)) for message in messages)
        agent._token_tracker.estimate_tools.return_value = 0
        with self.assertRaisesRegex(ValueError, "超出上下文窗口"):
            agent._fit_messages([{"role": "system", "content": "rules"}, {"role": "user", "content": "x" * 1000}], 600, [])

    def test_context_budget_keeps_complete_recent_turns(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.max_history = 10
        agent._token_tracker = Mock()
        agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(message)) for message in messages)
        agent._token_tracker.estimate_tools.return_value = 0
        system = {"role": "system", "content": "rules"}
        old = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"}]
        recent = [
            {"role": "user", "content": "new"},
            {"role": "assistant", "tool_calls": [{"id": "call", "function": {"name": "read_file"}}]},
            {"role": "tool", "tool_call_id": "call", "content": "data"},
            {"role": "assistant", "content": "answer"},
        ]

        result = agent._fit_messages([system] + old + recent, 512 + agent._token_tracker.estimate([system] + recent), [])

        self.assertEqual(result, [system] + recent)

    def test_context_budget_truncates_tool_output_without_mutating_original(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.max_history = 10
        agent._token_tracker = Mock()
        agent._token_tracker.estimate.side_effect = lambda messages: sum(len(str(message)) for message in messages)
        agent._token_tracker.estimate_tools.return_value = 0
        original = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "inspect"},
            {"role": "assistant", "tool_calls": [{"id": "call", "function": {"name": "read_file"}}]},
            {"role": "tool", "tool_call_id": "call", "content": "result" * 500},
        ]
        allowed = agent._token_tracker.estimate(original[:-1]) + 160

        result = agent._fit_messages(original, 512 + allowed, [])

        self.assertIn("已截断", result[-1]["content"])
        self.assertEqual(original[-1]["content"], "result" * 500)
        self.assertEqual(result[-2], original[-2])
        self.assertLessEqual(agent._token_tracker.estimate(result), allowed)

    def test_child_client_configuration_does_not_leak_main_api_key(self):
        config = {
            "api_key": "main-key", "base_url": "https://main.example/v1",
            "agent": {"auto_confirm": False, "max_history": 10},
            "model": {"default": "main"}, "memory": {},
            "delegation": {"enabled": True, "model": "cheap", "base_url": "https://child.example/v1"},
        }
        with patch("agent.core.OpenAI") as client_factory, patch("agent.core.MemoryManager") as memory, \
             patch("agent.core.discover_tools", return_value=([], {})), \
             patch("agent.core.TokenTracker"):
            client_factory.side_effect = [Mock(), Mock(), Mock()]
            memory.return_value.available = False
            agent = Agent(config, Mock(), "test", "main", Mock())
            self.assertIsNone(agent._subtask_runner)
            client_factory.assert_called_once_with(api_key="main-key", base_url="https://main.example/v1")

            config["delegation"]["api_key"] = "child-key"
            agent = Agent(config, Mock(), "test", "main", Mock())
            self.assertIsNotNone(agent._subtask_runner)
            client_factory.assert_called_with(api_key="child-key", base_url="https://child.example/v1")
            self.assertEqual(agent._subtask_runner.model, "cheap")

    def test_delegation_not_exposed_without_child_model(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.config = {"delegation": {"enabled": True}}
        agent.tool_definitions = []
        agent.tool_executors = {}
        agent._subtask_runner = None

        self.assertIsNone(agent._select_tools("question"))
        self.assertTrue(agent._call_tool("delegate_tasks", {}).startswith("错误："))

    def test_main_turn_stops_delegating_at_budget(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.config = {"delegation": {"enabled": True, "max_calls": 1}}
        agent._delegation_calls = 0
        agent._subtask_runner = Mock()
        agent._subtask_runner.execute.return_value = "done"

        self.assertEqual(agent._call_tool("delegate_tasks", {"tasks": []}), "done")
        self.assertTrue(agent._call_tool("delegate_tasks", {"tasks": []}).startswith("错误："))
        agent._subtask_runner.execute.assert_called_once()

    def test_main_prompt_explains_delegation_without_preclassification(self):
        agent = Agent.__new__(Agent)
        agent.activity = ActivityTracker()
        agent.config = {
            "agent": {"system_prompt": "main", "context_token_limit": 200000},
            "delegation": {"enabled": True},
        }
        agent.model = "main-model"
        agent.session_id = "test-session"
        agent.session_manager = Mock()
        agent.session_manager.get_history.return_value = [{"role": "user", "content": "question"}]
        agent._client = Mock()
        agent._memory_manager = None
        agent._subtask_runner = Mock()
        agent._token_tracker = Mock()
        agent._token_tracker.estimate.return_value = 1
        agent._token_tracker.estimate_tools.return_value = 0
        agent._skills = []
        agent._session_theme = ""
        agent._pinned_constraints = {}
        agent.tool_definitions = []

        messages = agent._get_messages()

        self.assertIn("【子任务分派】", messages[0]["content"])
        self.assertEqual(agent._pinned_constraints["active_skills"], [])
        self.assertEqual(messages[1]["content"], "question")

    def test_main_model_delegates_then_uses_child_result(self):
        agent = self._loop_agent()
        agent._context_metrics = {}
        agent._subtask_runner.execute.return_value = '{"results":[{"result":"child answer"}]}'
        delegated = SimpleNamespace(index=0, id="main-call", function=SimpleNamespace(
            name="delegate_tasks", arguments='{"tasks":[{"task":"simple"}]}'
        ))
        agent._client.chat.completions.create.side_effect = [
            self._stream(calls=[delegated], finish="tool_calls"), self._stream("final answer"),
        ]

        self.assertEqual(agent.chat("question"), "final answer")
        requests = agent._client.chat.completions.create.call_args_list
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(call.kwargs["model"] == "main-model" for call in requests))
        self.assertEqual(requests[0].kwargs["tools"][0]["function"]["name"], "delegate_tasks")
        self.assertIn("child answer", requests[1].kwargs["messages"][-1]["content"])
        self.assertIn("input_budget", agent._context_metrics)
        agent._subtask_runner.execute.assert_called_once()
        batch = agent.session_manager.append_messages.call_args.args[1]
        self.assertEqual(batch[0]["tool_calls"][0]["id"], batch[1]["tool_call_id"])
        self.assertIn("child answer", batch[1]["content"])
        self.assertEqual(agent.session_manager.append_message.call_args_list[-1].args[2], "final answer")

    def test_request_metrics_include_tool_schema_and_actual_prompt_usage(self):
        agent = self._loop_agent()
        agent._context_metrics = {"window": 4096}
        agent._token_tracker.estimate.return_value = 120
        agent._token_tracker.estimate_tools.return_value = 30
        agent._client.chat.completions.create.return_value = [
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=210, completion_tokens=4), choices=[]),
            *self._stream("answer"),
        ]

        self.assertEqual(agent.chat("question"), "answer")
        self.assertEqual(agent._context_metrics["message_tokens"], 120)
        self.assertEqual(agent._context_metrics["tool_tokens"], 30)
        self.assertEqual(agent._context_metrics["prompt_estimate"], 150)
        self.assertEqual(agent._context_metrics["actual_prompt_tokens"], 210)
        agent._token_tracker.calibrate_prompt_estimate.assert_called_once_with(150, 210)

    def test_calibration_converges_on_provider_usage(self):
        with patch("agent.token_tracker._load_tiktoken", return_value=None):
            tracker = TokenTracker()
        for _ in range(30):
            tracker.calibrate_prompt_estimate(round(100 * tracker.estimate_multiplier), 300)
        self.assertAlmostEqual(tracker.estimate_multiplier, 3, delta=0.1)

    def test_reasoning_is_neither_sent_back_nor_persisted(self):
        agent = self._loop_agent()
        agent._client.chat.completions.create.return_value = [SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(
                content="final answer", reasoning_content="private reasoning", tool_calls=[],
            ), finish_reason="stop")],
        )]

        self.assertEqual(agent.chat("question"), "final answer")
        self.assertNotIn("private reasoning", str(agent._client.chat.completions.create.call_args.kwargs["messages"]))
        stored = agent.session_manager.append_message.call_args_list
        self.assertEqual([entry.args[2] for entry in stored], ["question", "final answer"])


if __name__ == "__main__":
    unittest.main()