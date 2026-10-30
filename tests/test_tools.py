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
from unittest.mock import AsyncMock, Mock, patch

from rich.console import Console
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.layout.controls import FormattedTextControl

from agent.core import Agent
from main import ChatRunner, OutputPane, _invoked_skill, _paint_activity_icon, _show_dialogue_history
from tools import mcp, skill_manager
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

    def test_skill_command_recognizes_only_exact_leading_name(self):
        skills = [{"name": "review", "enabled": True}]
        self.assertIs(_invoked_skill("/review check this", skills), skills[0])
        self.assertIsNone(_invoked_skill("/reviews check this", skills))
        self.assertIsNone(_invoked_skill("please /review check this", skills))

    def test_activity_icon_only_updates_bottom_toolbar_cell(self):
        output = Mock()
        output.get_size.return_value.rows = 18
        _paint_activity_icon(output, "◐")
        self.assertEqual(output.mock_calls, [
            unittest.mock.call.write_raw("\x1b7"),
            unittest.mock.call.get_size(),
            unittest.mock.call.cursor_goto(18, 2),
            unittest.mock.call.write_raw("\x1b[0;32;7m◐\x1b8"),
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