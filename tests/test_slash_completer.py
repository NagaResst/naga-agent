"""SlashCompleter 补全逻辑测试。"""

import unittest
from types import SimpleNamespace

from prompt_toolkit.document import Document

from agent.slash_completer import SlashCompleter


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


if __name__ == "__main__":
    unittest.main()
