"""SlashCompleter 补全逻辑测试。"""

import unittest
from types import SimpleNamespace

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import set_app
from prompt_toolkit.buffer import CompletionState
from prompt_toolkit.data_structures import Size
from prompt_toolkit.document import Document
from prompt_toolkit.input import DummyInput
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.output import DummyOutput

from agent.slash_completer import SlashCompleter
from main import OutputPane, _configure_prompt_layout


def _agent():
    """构造带两个 skill 的假 agent（其中一个已停用）。"""
    return SimpleNamespace(_skills=[
        {"name": "codebase-investigation", "enabled": True, "description": "D" * 50},
        {"name": "roc-cli", "enabled": False, "description": "Rakuten CLI"},
    ])


_CONFIG = {"model": {"available": ["GLM-5.3"]}}


class SlashCompleterTests(unittest.TestCase):
    def _complete(self, text, max_items=5):
        """对给定输入文本执行补全，返回补全文本列表。"""
        completer = SlashCompleter(_agent(), _CONFIG, max_items=max_items)
        document = Document(text)
        return [c.text for c in completer.get_completions(document, None)]

    def test_slash_lists_five_items_starting_with_quit(self):
        """'/' 恰好 5 项且首项是 /quit。"""
        items = self._complete("/")
        self.assertEqual(len(items), 5)
        self.assertEqual(items[0], "/quit")

    def test_prefix_th_only_thinking(self):
        """'/th' 只出 /thinking。"""
        self.assertEqual(self._complete("/th"), ["/thinking"])

    def test_prefix_s_filters_session_and_skill(self):
        """'/s' 过滤出 /session /skill。"""
        self.assertEqual(self._complete("/s"), ["/session", "/skill"])

    def test_confirm_trailing_space_shows_on_off(self):
        """'/confirm '（尾空格）出 on off。"""
        self.assertEqual(self._complete("/confirm "), ["on", "off"])

    def test_thinking_trailing_space_shows_four_subcommands(self):
        """'/thinking ' 出 4 个子命令。"""
        self.assertEqual(self._complete("/thinking "), ["on", "off", "show", "hide"])

    def test_model_trailing_space_shows_auto_and_models(self):
        """'/model ' 出 auto 和 GLM-5.3。"""
        self.assertEqual(self._complete("/model "), ["auto", "GLM-5.3"])

    def test_skill_on_shows_all_skill_names(self):
        """'/skill on ' 出 skill 名（含停用的）。"""
        self.assertEqual(
            self._complete("/skill on "),
            ["codebase-investigation", "roc-cli"],
        )

    def test_skill_list_third_token_has_no_completion(self):
        """'/skill list ' 第 3 个 token 无补全。"""
        self.assertEqual(self._complete("/skill list "), [])

    def test_session_switch_has_no_completion(self):
        """'/session switch ' 无补全（自由文本）。"""
        self.assertEqual(self._complete("/session switch "), [])

    def test_prompt_set_has_no_completion(self):
        """'/prompt set ' 无补全。"""
        self.assertEqual(self._complete("/prompt set "), [])

    def test_plain_text_has_no_completion(self):
        """'hello' 无补全。"""
        self.assertEqual(self._complete("hello"), [])

    def test_empty_text_has_no_completion(self):
        """空串无补全。"""
        self.assertEqual(self._complete(""), [])

    def test_max_items_limits_results(self):
        """max_items=2 时只出 2 项。"""
        items = self._complete("/", max_items=2)
        self.assertEqual(len(items), 2)
        self.assertEqual(items, ["/quit", "/exit"])

    def test_skill_description_truncated_to_40_chars(self):
        """skill 描述截断到 40 字符。"""
        completer = SlashCompleter(_agent(), _CONFIG)
        document = Document("/codebase")
        completions = list(completer.get_completions(document, None))
        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0].text, "/codebase-investigation")
        self.assertEqual(completions[0].display_meta_text, "D" * 40)


class PromptLayoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_menu_and_status_fit_and_resize(self):
        class Output(DummyOutput):
            rows = 24

            def get_size(self):
                return Size(rows=self.rows, columns=100)

            def get_rows_below_cursor_position(self):
                return self.rows

        output = Output()
        session = PromptSession(
            message="---\ninput > ", completer=SlashCompleter(_agent(), _CONFIG),
            bottom_toolbar="STATUS", input=DummyInput(), output=output,
        )
        pane = OutputPane(session.app)
        pane.write("history\n" * 100)
        _configure_prompt_layout(session, pane)

        def render():
            screen = Screen()
            session.app.layout.container.write_to_screen(
                screen, MouseHandlers(), WritePosition(0, 0, 100, output.rows), "", False, None,
            )
            screen.draw_all_floats()
            return "\n".join(
                "".join(screen.data_buffer[y][x].char for x in range(100)).rstrip()
                for y in range(output.rows)
            )

        with set_app(session.app):
            idle = render()
            self.assertNotIn("Window too small", idle)
            self.assertIn("STATUS", idle)
            self.assertGreaterEqual(idle.count("history"), 20)
            session.default_buffer.document = Document("/", 1)
            completions = list(session.completer.get_completions(session.default_buffer.document, None))
            session.default_buffer.complete_state = CompletionState(
                session.default_buffer.document, completions=completions,
            )
            rendered = render()
            self.assertNotIn("Window too small", rendered)
            self.assertIn("STATUS", rendered)
            self.assertIn("input > /", rendered)
            for name in ("/quit", "/exit", "/help", "/confirm", "/thinking"):
                self.assertIn(name, rendered)

            output.rows = 5
            rendered = render()
            self.assertNotIn("Window too small", rendered)
            self.assertIn("STATUS", rendered)
            self.assertIn("input > /", rendered)

            output.rows = 24
            self.assertIn("/thinking", render())
            session.default_buffer.complete_state = None
            self.assertGreaterEqual(render().count("history"), 20)


if __name__ == "__main__":
    unittest.main()
