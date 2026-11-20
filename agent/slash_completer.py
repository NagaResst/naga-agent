"""slash 命令输入补全器。

为终端交互中的 `/` 命令提供 prompt_toolkit 补全：
    - 第一层：命令名补全（内置命令 + 已发现的 skill 名）
    - 第二层：子命令 / 参数补全（如 `/confirm on`、`/model GLM-5.3`）

设计说明：
    - 纯逻辑类，不依赖 main.py，只读取 agent._skills 与 config['model']['available']。
    - 补全项数量受 max_items 限制，避免刷屏。
"""

from __future__ import annotations

from prompt_toolkit.completion import Completer, Completion


class SlashCompleter(Completer):
    """slash 命令补全器。"""

    # 子命令表：命令 -> (描述, 可选子命令列表)
    SUBCOMMANDS: dict[str, tuple[str, list[str]]] = {
        "/confirm": ("on|off 自动确认", ["on", "off"]),
        "/thinking": ("on|off|show|hide 思维链", ["on", "off", "show", "hide"]),
        "/delegate": ("on|off 子任务分派", ["on", "off"]),
        "/prompt": ("set <内容>|clear 系统提示词", ["set", "clear"]),
        "/session": ("new|list|switch <N> 会话管理", ["new", "list", "switch"]),
        "/skill": ("list|on|off <名称> skill 管理", ["list", "on", "off"]),
    }

    def __init__(self, agent, config, max_items: int = 5):
        self.agent = agent
        self.config = config
        self.max_items = max_items

    def _command_items(self) -> list[tuple[str, str]]:
        """返回第一层命令补全项（命令含斜杠, 描述）。"""
        items: list[tuple[str, str]] = [
            ("/quit", "退出程序"),
            ("/exit", "退出程序"),
            ("/help", "显示帮助"),
            ("/confirm", "on|off 自动确认"),
            ("/thinking", "on|off|show|hide 思维链"),
            ("/model", "<名称>|auto 切换主模型"),
            ("/delegate", "on|off 子任务分派"),
            ("/prompt", "set <内容>|clear 系统提示词"),
            ("/session", "new|list|switch <N> 会话管理"),
            ("/token", "本会话 token 明细"),
            ("/skill", "list|on|off <名称> skill 管理"),
            ("/history", "最近 5 条对话摘要"),
            ("/clear", "清除当前会话消息"),
        ]
        # 追加已发现的 skill，作为 /<skill 名> 快捷命令
        for skill in getattr(self.agent, "_skills", None) or []:
            items.append(("/" + skill["name"], (skill.get("description") or "")[:40]))
        return items

    def _arg_pool(self, cmd: str, tokens: list[str]) -> list[tuple[str, str]]:
        """返回第二层参数补全项（候选值, 描述）。"""
        if cmd == "/model":
            available = (self.config.get("model") or {}).get("available") or []
            return [("auto", "恢复默认模型")] + [(m, "") for m in available]

        # /skill on|off <名称>：列出全部 skill（含停用的）
        if cmd == "/skill" and len(tokens) in (3, 4) and tokens[1] in ("on", "off"):
            return [
                (skill["name"], (skill.get("description") or "")[:40])
                for skill in getattr(self.agent, "_skills", None) or []
            ]

        # 其余命令：仅补全第一层子命令
        if cmd in self.SUBCOMMANDS and len(tokens) == 2:
            _, words = self.SUBCOMMANDS[cmd]
            return [(w, self.SUBCOMMANDS[cmd][0]) for w in words]

        return []

    def get_completions(self, document, complete_event):
        """按当前输入产出补全项。"""
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        tokens = text.split(" ")
        current = tokens[-1]

        if len(tokens) == 1:
            pool = self._command_items()
        else:
            pool = self._arg_pool(tokens[0], tokens)

        count = 0
        for name, desc in pool:
            if not name.startswith(current):
                continue
            yield Completion(name, start_position=-len(current), display_meta=desc)
            count += 1
            if count >= self.max_items:
                return
