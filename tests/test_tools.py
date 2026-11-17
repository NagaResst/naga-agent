import asyncio
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from rich.console import Console
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.layout.controls import FormattedTextControl

from agent.core import Agent
from agent.subagent import SubtaskRunner
from main import ChatRunner, DialogueHistoryPager, OutputPane, _invoked_skill, _paint_status_line, _show_dialogue_history
from session.sqlite_session import SessionManager
from tools import mcp, read_file, skill_manager
from tools.registry import discover_tools


class ToolTests(unittest.TestCase):
    def tearDown(self):
        mcp.set_config({})
        skill_manager.set_agent(None)

    def test_registered_tools(self):
        definitions, executors = discover_tools()
        names = {item["function"]["name"] for item in definitions}
        self.assertEqual(names, {"execute_command", "read_file", "memory", "skill_manager", "mcp"})
        self.assertEqual(names, set(executors))

    def test_read_file_returns_more_than_3000_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.txt"
            expected = "line of file content\n" * 400
            path.write_text(expected, encoding="utf-8")

            result = read_file.execute({"path": str(path)})

        self.assertIn(expected, result)
        self.assertNotIn("已截断", result)

    def test_read_file_still_caps_very_large_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "huge.txt"
            path.write_text("x" * 20000, encoding="utf-8")

            result = read_file.execute({"path": str(path)})

        self.assertIn("已截断", result)
        self.assertIn("实际返回：1 行（已截断，末行不完整）", result)
        self.assertLess(len(result), 12500)

    def test_read_file_reports_requested_range_and_returned_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lines.txt"
            path.write_text("first\nsecond\nthird\nfourth\n", encoding="utf-8")
            args = {"path": str(path), "start_line": 2, "end_line": 3}
            result = read_file.execute(args)

        self.assertIn("请求行范围：2-3", Agent._read_file_range(args))
        self.assertEqual(Agent._read_file_result_summary(result),
                         "行范围：2-3 / 共 4 行；实际返回：2 行")
        self.assertEqual(Agent._read_file_result_summary("错误：文件不存在"),
                         "读取失败：错误：文件不存在")

    def test_text_tool_call_detection_ignores_fenced_examples(self):
        text = "<tool_call>execute_command<arg_key>command</arg_key><arg_value>echo test</arg_value></tool_call>"
        self.assertTrue(Agent._has_text_tool_call(text))
        self.assertFalse(Agent._has_text_tool_call(f"Example:\n```xml\n{text}\n```"))
        self.assertFalse(Agent._has_text_tool_call("工具调用已完成"))

    def test_structured_tool_call_requires_valid_identifiers_and_object_arguments(self):
        valid = {"id": "call", "function": {"name": "read_file", "arguments": '{"path":"a"}'}}
        self.assertTrue(Agent._valid_tool_calls([valid]))
        self.assertFalse(Agent._valid_tool_calls([valid, valid]))
        self.assertFalse(Agent._valid_tool_calls([
            {"id": "call", "function": {"name": "read_file", "arguments": "["}},
        ]))
        self.assertFalse(Agent._valid_tool_calls([
            {"id": "call", "function": {"name": "read_file", "arguments": "[]"}},
        ]))

    def test_text_tool_call_is_retried_but_never_executed_directly(self):
        agent = Agent.__new__(Agent)
        agent.activity = Mock()
        agent.config = {"agent": {"show_thinking": False}, "model": {"extra_params": {}},
                        "tools": {}, "delegation": {}}
        agent.model = "test"
        agent.session_id = "session"
        agent.session_manager = Mock()
        agent.console = Console(file=StringIO(), width=100, color_system=None)
        agent._token_tracker = Mock()
        agent._token_tracker.estimate.return_value = 1
        agent._token_tracker.estimate_tools.return_value = 1
        agent._memory_manager = None
        agent._snapshot_runtime_state = Mock()
        agent._reset_runtime_state = Mock()
        agent._restore_followup_state = Mock()
        agent._get_messages = Mock(return_value=[{"role": "user", "content": "question"}])
        agent._upsert_runtime_state_message = Mock()
        agent._select_tools = Mock(return_value=[{"type": "function", "function": {"name": "read_file"}}])
        agent._fit_messages = Mock(side_effect=lambda messages, *_: messages)
        agent._summarize_reply_conclusion = Mock(return_value="")
        agent._record_tool_event = Mock()
        agent._call_tool = Mock(return_value="file result")
        agent._client = Mock()
        agent._context_window = 4096

        def chunk(content=None, calls=None, finish="stop"):
            delta = SimpleNamespace(content=content, reasoning_content=None, tool_calls=calls or [])
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta, finish_reason=finish)])

        bad = "<tool_call>execute_command<arg_key>command</arg_key><arg_value>unsafe</arg_value></tool_call>"
        call = SimpleNamespace(index=0, id="call1", function=SimpleNamespace(
            name="read_file", arguments='{"path":"safe.txt"}',
        ))
        agent._client.chat.completions.create.side_effect = [
            [chunk(bad)], [chunk(calls=[call], finish="tool_calls")], [chunk("done")],
        ]

        self.assertEqual(agent.chat("question"), "done")
        self.assertEqual(agent._client.chat.completions.create.call_count, 3)
        agent._call_tool.assert_called_once_with("read_file", {"path": "safe.txt"})

    def test_skill_command_recognizes_only_exact_leading_name(self):
        skills = [{"name": "review", "enabled": True}]
        self.assertIs(_invoked_skill("/review check this", skills), skills[0])
        self.assertIsNone(_invoked_skill("/reviews check this", skills))
        self.assertIsNone(_invoked_skill("please /review check this", skills))

    def test_activity_icon_only_updates_bottom_toolbar_cell(self):
        output = Mock()
        output.get_size.return_value.rows = 18
        _paint_status_line(output, "◐", "待命")
        self.assertEqual(output.mock_calls, [
            unittest.mock.call.write_raw("\x1b7"),
            unittest.mock.call.get_size(),
            unittest.mock.call.cursor_goto(18, 2),
            unittest.mock.call.write_raw("\x1b[0;32;7m◐ 待命\x1b8"),
            unittest.mock.call.flush(),
        ])

    def test_output_pane_accumulates_rich_stream_without_printing_to_terminal(self):
        app = Mock()
        pane = OutputPane(app)
        self.assertEqual(pane.write("\x1b[32mfirst"), 10)
        pane.write("\x1b[0m\nsecond")
        self.assertEqual(fragment_list_to_text(to_formatted_text(pane.formatted_text())), "first\nsecond")
        self.assertEqual(pane.cursor_position(), Point(x=0, y=1))
        pane.scroll(1)
        self.assertEqual(fragment_list_to_text(to_formatted_text(pane.formatted_text())), "first\n")
        self.assertEqual(pane.cursor_position(), Point(x=0, y=1))
        pane.write("\nthird")
        self.assertNotIn("third", fragment_list_to_text(to_formatted_text(pane.formatted_text())))
        pane.scroll(-2)
        self.assertEqual(fragment_list_to_text(to_formatted_text(pane.formatted_text())), "first\nsecond\nthird")
        self.assertEqual(pane.cursor_position(), Point(x=0, y=2))
        self.assertEqual(app.invalidate.call_count, 5)

    def test_output_pane_cursor_uses_same_snapshot_as_rendered_text(self):
        pane = OutputPane(Mock())
        pane.write("first")

        def append_after_snapshot():
            snapshot = pane.formatted_text()
            pane.write("\nsecond")
            return snapshot

        control = FormattedTextControl(append_after_snapshot, get_cursor_position=pane.cursor_position)
        content = control.create_content(80, 10)
        self.assertEqual(content.line_count, 1)
        self.assertEqual(content.cursor_position, Point(x=0, y=0))

    def test_output_pane_renders_bounded_history_and_scrolls_to_old_lines(self):
        pane = OutputPane(Mock())
        pane.write("\x1b[31m" + "".join(f"line {n}\n" for n in range(500)) + "\x1b[0m")

        fragments = to_formatted_text(pane.formatted_text())
        displayed = fragment_list_to_text(fragments)
        self.assertNotIn("line 0\n", displayed)
        self.assertIn("line 499\n", displayed)
        self.assertTrue(any("ansired" in style for style, text in fragments if "l" in text))
        pane.scroll(499)
        self.assertIn("line 0\n", fragment_list_to_text(to_formatted_text(pane.formatted_text())))
        pane.scroll(-499)
        self.assertIn("line 499\n", fragment_list_to_text(to_formatted_text(pane.formatted_text())))

    def test_output_pane_keeps_scrollback_anchor_during_writes(self):
        pane = OutputPane(Mock())
        pane.write("".join(f"line {n}\n" for n in range(500)))
        pane.scroll(100)
        before = fragment_list_to_text(to_formatted_text(pane.formatted_text()))
        pane.write("new line\n")
        after = fragment_list_to_text(to_formatted_text(pane.formatted_text()))
        self.assertEqual(after, before)

    def test_output_pane_caches_snapshot_until_content_changes(self):
        pane = OutputPane(Mock())
        pane.write("first")
        snapshot = pane.formatted_text()
        self.assertIs(pane.formatted_text(), snapshot)
        pane.write(" second")
        self.assertIsNot(pane.formatted_text(), snapshot)
        self.assertEqual(fragment_list_to_text(to_formatted_text(pane.formatted_text())), "first second")

    def test_reasoning_fold_retains_all_chunks_and_interleaved_output(self):
        pane = OutputPane(Mock(), show_thinking=False)
        pane.start_reasoning()
        pane.write_reasoning("first ")
        pane.write("queued message\n")
        pane.write_reasoning("second")
        pane.end_reasoning()
        pane.start_response()
        pane.write("raw answer")
        pane.finish_response("final answer")

        collapsed = fragment_list_to_text(to_formatted_text(pane.formatted_text()))
        self.assertIn("思考过程已折叠", collapsed)
        self.assertNotIn("first ", collapsed)
        pane.set_thinking_expanded(True)
        expanded = fragment_list_to_text(to_formatted_text(pane.formatted_text()))
        self.assertIn("first queued message\nsecond", expanded)
        self.assertIn("final answer", expanded)
        pane.set_thinking_expanded(False)
        self.assertNotIn("first ", fragment_list_to_text(to_formatted_text(pane.formatted_text())))
        pane.set_thinking_expanded(True)
        self.assertEqual(fragment_list_to_text(to_formatted_text(pane.formatted_text())), expanded)

    def test_reasoning_fold_during_active_response_keeps_offsets(self):
        pane = OutputPane(Mock())
        pane.start_response()
        pane.write("raw ")
        pane.start_reasoning()
        pane.write_reasoning("reason")
        pane.end_reasoning()
        pane.write("answer")
        pane.set_thinking_expanded(False)
        pane.finish_response("final")
        pane.set_thinking_expanded(True)
        displayed = fragment_list_to_text(to_formatted_text(pane.formatted_text()))
        self.assertIn("reason", displayed)
        self.assertIn("final", displayed)
        self.assertNotIn("raw", displayed)

    def test_folded_reasoning_keeps_existing_scroll_anchor(self):
        pane = OutputPane(Mock(), show_thinking=False)
        pane.write("".join(f"line {number}\n" for number in range(50)))
        pane.scroll(10)
        before = fragment_list_to_text(to_formatted_text(pane.formatted_text()))
        pane.start_reasoning()
        pane.write_reasoning("hidden")
        self.assertEqual(fragment_list_to_text(to_formatted_text(pane.formatted_text())), before)

    def test_history_pager_loads_complete_turns_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SessionManager(str(Path(directory) / "history.db"))
            session_id = manager.create_session("pages")
            for number in range(5):
                manager.append_message(session_id, "user", f"question {number}")
                manager.append_message(session_id, "assistant", "", {"tool_calls": [{"id": str(number)}]})
                manager.append_message(session_id, "tool", f"internal {number}")
                manager.append_message(session_id, "assistant", f"answer {number}")
            pager = DialogueHistoryPager(manager, session_id, page_size=3)
            pages = []
            while not pager.exhausted:
                page = pager.next_page()
                visible = [
                    item for item in page
                    if item["role"] in ("user", "assistant") and not item.get("tool_calls")
                    and item["content"].strip()
                ]
                if visible:
                    self.assertEqual(visible[0]["role"], "user")
                    pages.append(visible)
            displayed = [item["content"] for page in reversed(pages) for item in page]
            self.assertEqual(
                displayed,
                [text for number in range(5) for text in (f"question {number}", f"answer {number}")],
            )
            manager.close()

    def test_prepend_keeps_scrollback_and_response_replacement(self):
        pane = OutputPane(Mock())
        pane.write("recent\n")
        pane.start_response()
        pane.write("stream")
        pane.scroll(1)
        pane.prepend("older\n")
        pane.finish_response("final")
        pane.scroll(-100)
        displayed = fragment_list_to_text(to_formatted_text(pane.formatted_text()))
        self.assertEqual(displayed, "older\nrecent\nfinal")

    def test_restored_session_shows_complete_dialogue_without_tool_output(self):
        output = StringIO()
        console = Console(file=output, width=100, color_system=None)
        history = [{"role": "user", "content": "old"}]
        history += [{"role": "tool", "content": "tool output"},
                    {"role": "assistant", "content": "tool call", "tool_calls": [{"id": "1"}]}]
        history += [
            {"role": role, "content": f"turn {number}"}
            for number in range(1, 7)
            for role in ("user", "assistant")
        ]
        history[-1]["content"] = "[literal] " + "x" * 700
        _show_dialogue_history(console, history)
        text = output.getvalue()
        self.assertIn("old", text)
        self.assertNotIn("tool output", text)
        self.assertNotIn("tool call", text)
        self.assertIn("turn 3", text)
        self.assertIn("turn 4", text)
        self.assertIn("[literal]", text)
        self.assertEqual(text.count("x"), 700)

    def test_restored_dialogue_shows_timestamp_and_reply_duration(self):
        output = StringIO()
        console = Console(file=output, width=100, color_system=None)
        _show_dialogue_history(console, [
            {"role": "user", "content": "question", "timestamp": "2026-10-30T09:00:00"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}], "timestamp": "2026-10-30T09:00:01"},
            {"role": "tool", "content": "internal", "timestamp": "2026-10-30T09:00:02"},
            {"role": "assistant", "content": "answer", "timestamp": "2026-10-30T09:00:03.250000"},
            {"role": "user", "content": "next", "timestamp": "invalid"},
            {"role": "assistant", "content": "later", "timestamp": "2026-10-30T09:00:10"},
            {"role": "user", "content": "last", "timestamp": "2026-10-30T09:00:11"},
            {"role": "assistant", "content": "undated", "timestamp": "invalid"},
            {"role": "assistant", "content": "unpaired", "timestamp": "2026-10-30T09:00:15"},
        ])
        text = output.getvalue()
        self.assertIn("2026-10-30 09:00:00", text)
        self.assertIn("用时 3.2s", text)
        self.assertEqual(text.count("用时"), 1)
        self.assertGreaterEqual(text.count("─"), 3)
        self.assertNotIn("internal", text)

    def test_restored_assistant_markdown_keeps_user_text_literal(self):
        output = StringIO()
        console = Console(file=output, width=80, color_system=None)
        _show_dialogue_history(console, [
            {"role": "user", "content": "**keep stars**"},
            {"role": "assistant", "content": "Here is the plan:\n\n```markdown\n## Phase 1\n\n**bold**\n\n| Name | Value |\n| --- | --- |\n| A | B |\n```\n\nNext step."},
        ])
        displayed = output.getvalue()
        self.assertIn("**keep stars**", displayed)
        self.assertNotIn("**bold**", displayed)
        self.assertNotIn("```", displayed)
        self.assertIn("Here is the plan:", displayed)
        self.assertIn("Next step.", displayed)
        self.assertIn("Phase 1", displayed)
        self.assertNotIn("## Phase 1", displayed)
        self.assertNotIn("| --- |", displayed)

    def test_new_session_clears_displayed_history(self):
        app = Mock()
        pane = OutputPane(app)
        pane.write("previous dialogue\n")
        pane.scroll(1)
        pane.clear()
        self.assertEqual(fragment_list_to_text(to_formatted_text(pane.formatted_text())), "")
        self.assertEqual(pane.cursor_position(), Point(x=0, y=0))

    def test_chat_runner_queues_messages_and_routes_confirmation(self):
        started = threading.Event()
        finished = threading.Event()
        calls = []
        console = Mock()
        console.input.return_value = "foreground"

        def chat(text):
            calls.append(text)
            if text == "first":
                started.set()
                calls.append(console.input("Confirm?"))
            else:
                finished.set()

        agent = Mock()
        agent.chat.side_effect = chat
        prompt_session = Mock()
        prompt_session.app.loop = None
        prompt_session.default_buffer = Buffer()
        draft = Document("draft message", cursor_position=5)
        prompt_session.default_buffer.set_document(draft)
        runner = ChatRunner(agent, console, prompt_session, Mock())
        try:
            runner.submit("first", "session")
            self.assertTrue(started.wait(2))
            runner.activate_confirmation()
            self.assertTrue(runner.needs_confirmation)
            self.assertEqual(prompt_session.default_buffer.text, "")
            runner.submit("second", "session")
            self.assertTrue(runner.busy)
            self.assertEqual(console.input("Other?"), "foreground")
            self.assertTrue(runner.answer_confirmation("y"))
            self.assertTrue(finished.wait(2))
            self.assertEqual(calls, ["first", "y", "second"])
            self.assertEqual(runner.take_draft(), draft)
            self.assertEqual(console.rule.call_count, 2)
            self.assertTrue(any("回复用时 " in call.args[0] for call in console.print.call_args_list))
        finally:
            if runner.needs_confirmation:
                runner.answer_confirmation("n")
            runner.close()
        self.assertIs(console.input, runner._original_input)

    def test_chat_runner_reports_markup_exception_and_continues(self):
        output = StringIO()
        console = Console(file=output, width=100, color_system=None)
        agent = Mock()
        finished = threading.Event()

        def chat(text):
            if text == "first":
                raise ValueError("closing tag '[/dim]' was unexpected")
            finished.set()

        agent.chat.side_effect = chat
        prompt_session = Mock()
        runner = ChatRunner(agent, console, prompt_session, Mock())
        try:
            runner.submit("first", "session")
            runner.submit("second", "session")
            self.assertTrue(finished.wait(2))
        finally:
            runner.close()
        self.assertIn("closing tag '[/dim]' was unexpected", output.getvalue())

    def test_cancel_current_does_not_cancel_queued_job(self):
        started = threading.Event()
        completed = threading.Event()
        agent = Mock()
        calls = []

        def chat(text):
            calls.append(text)
            if text == "first":
                started.set()
                self.assertTrue(agent._cancel_event.wait(2))
            else:
                self.assertFalse(agent._cancel_event.is_set())
                completed.set()

        agent.chat.side_effect = chat
        runner = ChatRunner(agent, Mock(), Mock(), Mock())
        try:
            runner.submit("first", "session")
            self.assertTrue(started.wait(2))
            runner.submit("second", "session")
            self.assertTrue(runner.cancel_current())
            self.assertTrue(completed.wait(2))
            self.assertEqual(calls, ["first", "second"])
        finally:
            runner.close()

    def test_cancel_stops_waiting_for_a_stalled_reasoning_stream(self):
        blocked = threading.Event()
        release = threading.Event()
        cancel = threading.Event()
        finished = threading.Event()
        first_received = threading.Event()
        received = []
        errors = []

        class Stream:
            def __iter__(self):
                yield "reasoning"
                blocked.set()
                release.wait()
                yield "late"

            def close(self):
                pass

        agent = Agent.__new__(Agent)
        agent._active_stream_lock = threading.Lock()
        agent._active_stream = None
        agent.console = Mock()

        def consume():
            try:
                for chunk in agent._iterate_stream(Stream(), cancel.is_set):
                    received.append(chunk)
                    first_received.set()
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        try:
            self.assertTrue(first_received.wait(2))
            self.assertTrue(blocked.wait(2))
            cancel.set()
            agent.cancel_active_stream()
            self.assertTrue(finished.wait(1))
            self.assertEqual(received, ["reasoning"])
            self.assertEqual(errors, [])
        finally:
            release.set()
            worker.join(2)

    def test_cancel_closes_stream_before_next_queued_job_starts(self):
        started = threading.Event()
        first_returned = threading.Event()
        close_started = threading.Event()
        release_close = threading.Event()
        second_started = threading.Event()
        agent = Mock()

        def chat(text):
            if text == "first":
                started.set()
                agent._cancel_event.wait(2)
                first_returned.set()
            else:
                second_started.set()

        def close_stream():
            close_started.set()
            release_close.wait()

        agent.chat.side_effect = chat
        agent.cancel_active_stream.side_effect = close_stream
        runner = ChatRunner(agent, Mock(), Mock(), Mock())
        cancel_thread = None
        try:
            runner.submit("first", "session")
            self.assertTrue(started.wait(2))
            runner.submit("second", "session")
            cancel_thread = threading.Thread(target=runner.cancel_current, daemon=True)
            cancel_thread.start()
            self.assertTrue(close_started.wait(2))
            self.assertTrue(first_returned.wait(2))
            self.assertFalse(second_started.wait(0.1))
            release_close.set()
            self.assertTrue(second_started.wait(2))
        finally:
            release_close.set()
            if cancel_thread is not None:
                cancel_thread.join(2)
            runner.close()

    def test_subagent_stops_before_following_tasks_after_cancellation(self):
        cancelled = threading.Event()
        client = Mock()

        def complete(**kwargs):
            cancelled.set()
            return SimpleNamespace(
                usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))]
            )

        client.chat.completions.create.side_effect = complete
        runner = SubtaskRunner(client, "child", [], {}, Mock(), cancelled=cancelled.is_set)
        result = runner.execute({"tasks": [{"task": "first"}, {"task": "second"}]})
        self.assertIn("已中断", result)
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_chat_renders_reasoning_as_literal_text(self):
        output = StringIO()
        agent = Agent.__new__(Agent)
        agent.activity = Mock()
        agent.config = {"agent": {"show_thinking": True, "context_token_limit": 4096},
                        "model": {"extra_params": {}}, "tools": {}, "delegation": {}}
        agent.model = "test"
        agent.session_id = "session"
        agent.session_manager = Mock()
        agent.console = Console(file=output, width=100, color_system=None)
        agent._token_tracker = Mock()
        agent._token_tracker.estimate.return_value = 1
        agent._token_tracker.estimate_tools.return_value = 1
        agent._memory_manager = None
        agent._snapshot_runtime_state = Mock()
        agent._reset_runtime_state = Mock()
        agent._restore_followup_state = Mock()
        agent._get_messages = Mock(return_value=[{"role": "user", "content": "question"}])
        agent._upsert_runtime_state_message = Mock()
        agent._select_tools = Mock(return_value=None)
        agent._fit_messages = Mock(side_effect=lambda messages, *_: messages)
        agent._summarize_reply_conclusion = Mock(return_value="")
        agent._client = Mock()
        agent._context_window = 4096
        delta = SimpleNamespace(content="answer", reasoning_content="literal [/dim]", tool_calls=[])
        agent._client.chat.completions.create.return_value = [
            SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta, finish_reason="stop")]),
        ]

        self.assertEqual(agent.chat("question"), "answer")
        self.assertIn("literal [/dim]", output.getvalue())

    def test_mcp_stdio_lists_and_calls_tools(self):
        server_code = (
            "from mcp.server.fastmcp import FastMCP\n"
            "server = FastMCP('test')\n"
            "@server.tool()\n"
            "def echo(message: str) -> str:\n"
            "    return message\n"
            "server.run(transport='stdio')\n"
        )
        mcp.set_config({"servers": {"local": {
            "command": sys.executable, "args": ["-c", server_code],
        }}})
        self.assertEqual(json.loads(mcp.execute({"action": "list"})), {"servers": ["local"]})
        available = json.loads(mcp.execute({"action": "list", "server": "local"}))
        self.assertEqual(available[0]["name"], "echo")
        detail = json.loads(mcp.execute({"action": "list", "server": "local", "tool": "echo"}))
        self.assertIn("message", detail["inputSchema"]["properties"])
        response = mcp.execute({
            "action": "call", "server": "local", "tool": "echo", "arguments": {"message": "hello"},
        })
        self.assertIn("hello", response)
        self.assertTrue(mcp.execute({
            "action": "call", "server": "local", "tool": "missing", "arguments": {},
        }).startswith("错误："))

    def test_mcp_http_expands_headers_and_connect_timeout(self):
        captured = {}

        @asynccontextmanager
        async def fake_http_client(url, *, http_client):
            captured["url"] = url
            captured["authorization"] = http_client.headers["Authorization"]
            captured["connect_timeout"] = http_client.timeout.connect
            yield None, None, None

        server = {
            "url": "https://api.anysearch.com/mcp", "connect_timeout": 60,
            "headers": {"Authorization": "Bearer ${MCP_TEST_KEY}"},
        }
        with patch.dict(os.environ, {"MCP_TEST_KEY": "dummy"}), \
             patch("tools.mcp.streamable_http_client", fake_http_client), \
             patch("tools.mcp._exchange", new_callable=AsyncMock, return_value="ok"):
            self.assertEqual(asyncio.run(mcp._request(server, "list", None, {})), "ok")
        self.assertEqual(captured, {
            "url": "https://api.anysearch.com/mcp", "authorization": "Bearer dummy",
            "connect_timeout": 60,
        })
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "MCP_TEST_KEY"):
                mcp._resolve_headers(server["headers"])

    def test_skill_manager_does_not_remove_files_and_preserves_toggle_on_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            skill_file = Path(directory) / "demo.md"
            skill_file.write_text("---\nname: demo\nenabled: true\n---\nInstructions\n", encoding="utf-8")
            agent = Agent.__new__(Agent)
            agent._skills_dir = directory
            agent._skills_enabled_names = []
            agent._skill_overrides = {}
            agent._last_skill_system_tokens = 1
            agent.reload_skills()
            skill_manager.set_agent(agent)
            self.assertIn("已停用", skill_manager.execute({"action": "disable", "name": "demo"}))
            self.assertIn("[停用] demo", skill_manager.execute({"action": "reload"}))
            self.assertTrue(skill_manager.execute({"action": "remove", "name": "demo"}).startswith("错误："))
            self.assertTrue(skill_file.exists())
            enabled = skill_manager.execute({"action": "enable", "name": "demo"})
            self.assertIn("已启用", enabled)
            self.assertIn(str(skill_file), enabled)
            self.assertIn("[激活] demo", skill_manager.execute({"action": "list"}))


if __name__ == "__main__":
    unittest.main()