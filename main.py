import asyncio
import sys
import os
import queue
import threading
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.data_structures import Point
from prompt_toolkit.patch_stdout import patch_stdout

from agent.config import load_config
from agent.core import Agent, markdown_body
from agent.context_window import get_context_window
from session.sqlite_session import SessionManager


HELP_TEXT = """
可用内置命令：
  /quit, /exit        退出程序
  /confirm on         开启自动确认（命令无需手动确认）
  /confirm off        关闭自动确认（命令需手动确认，默认）
  /thinking on        开启思维链模式（需模型在 extra_params 中配置 enable_thinking）
  /thinking off       关闭思维链模式
  /thinking show      显示思考过程
  /thinking hide      隐藏思考过程
    /model <名称>       切换主模型，例如 /model qwen-plus
    /model auto         恢复配置中的默认主模型
    /delegate on/off    开关低成本子任务分派
  /prompt             查看当前系统提示词
  /prompt set <内容>  设置新的系统提示词
  /prompt clear       清除系统提示词
  /session            显示当前会话信息（含 token 消耗和费用）
  /session new        创建并切换到新会话
  /session list       列出最近 10 条历史会话
  /session switch <N> 切换到 list 中编号为 N 的会话
  /token              显示本会话详细 token 用量和费用明细
  /skill list         显示所有可用 skill 及激活状态
  /skill on <名称>    激活指定 skill
  /skill off <名称>   停用指定 skill
    /<Skill 名称> <任务> 显式使用已激活的 Skill
  /history            显示最近 5 条对话摘要
  /clear              清除当前会话的所有历史消息
  /help               显示此帮助信息
"""


def _invoked_skill(user_input: str, skills: list[dict]) -> dict | None:
    if not user_input.startswith("/"):
        return None
    name = user_input[1:].split(maxsplit=1)[0].casefold()
    return next((skill for skill in skills if skill["name"].casefold() == name), None)


def _show_dialogue_header(console: Console, role: str, timestamp=None, elapsed=None):
    label = "你" if role == "user" else "AI"
    when = f" · {timestamp.strftime('%Y-%m-%d %H:%M:%S')}" if timestamp else ""
    duration = f" · 用时 {elapsed:.1f}s" if elapsed is not None else ""
    console.rule(f"[bold cyan]{label}[/bold cyan][dim]{when}{duration}[/dim]", style="dim")


def _show_dialogue_message(console: Console, role: str, content: str, timestamp=None, elapsed=None):
    _show_dialogue_header(console, role, timestamp, elapsed)
    if role == "assistant":
        console.print(Markdown(markdown_body(content)))
    else:
        console.print(content, markup=False, highlight=False)


def _show_dialogue_history(console: Console, history: list[dict]):
    messages = [
        message for message in history
        if message["role"] in ("user", "assistant")
        and not message.get("tool_calls")
        and (message.get("content") or "").strip()
    ]
    if not messages:
        return
    last_user_time = None
    for message in messages:
        try:
            timestamp = datetime.fromisoformat(message["timestamp"])
        except (KeyError, TypeError, ValueError):
            timestamp = None
        elapsed = None
        if message["role"] == "user":
            last_user_time = timestamp
        else:
            if last_user_time and timestamp:
                try:
                    elapsed = max(0.0, (timestamp - last_user_time).total_seconds())
                except TypeError:
                    pass
            last_user_time = None
        _show_dialogue_message(console, message["role"], message["content"].strip(), timestamp, elapsed)
    console.print()


def select_session(console: Console, session_manager: SessionManager) -> str:
    sessions = session_manager.list_sessions()[:10]
    console.print("\n[bold]历史会话：[/bold]")
    for i, s in enumerate(sessions, 1):
        console.print(
            f"  {i}. [{s['created_at'][:19]}] {s['name']}  "
            f"[dim]({s['message_count']} 条消息)[/dim]"
        )
    console.print(f"  N. 新建会话")

    choice = Prompt.ask(
        "请选择会话编号，或输入 N 新建",
        default="N",
        console=console,
    ).strip().upper()

    if choice == "N" or not sessions:
        session_id = session_manager.create_session("new")
        console.print(f"[green]已创建新会话[/green]")
        return session_id

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(sessions):
            session_id = sessions[idx]["id"]
            console.print(f"[green]已加载会话：{session_id}[/green]")
            return session_id
    except ValueError:
        pass

    console.print("[yellow]输入无效，创建新会话。[/yellow]")
    session_id = session_manager.create_session("default")
    return session_id


def _auto_rename_session(session_manager, session_id: str, first_input: str, agent):
    try:
        summary_messages = [
            {"role": "user", "content": (
                f"请用不超过8个字总结以下问题，只输出总结文字，不要标点符号：\n{first_input}"
            )}
        ]
        resp = agent._client.chat.completions.create(
            model=agent.model,
            messages=summary_messages,
            temperature=0.3,
        )
        name = resp.choices[0].message.content.strip().replace(" ", "_")[:20] or "unnamed"
        session_manager.rename_session(session_id, name)
    except Exception:
        pass


class ChatRunner:
    def __init__(self, agent, console, prompt_session, session_manager):
        self._agent = agent
        self._console = console
        self._prompt_session = prompt_session
        self._session_manager = session_manager
        self._jobs = queue.Queue()
        self._confirmations = queue.Queue()
        self._confirmation_active = False
        self._draft = None
        self._original_input = console.input
        console.input = self._input
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def busy(self) -> bool:
        return self._jobs.unfinished_tasks > 0

    @property
    def needs_confirmation(self) -> bool:
        return self._confirmation_active

    def activate_confirmation(self):
        if self._confirmation_active or self._confirmations.empty():
            return
        self._draft = self._prompt_session.default_buffer.document
        self._prompt_session.default_buffer.reset()
        self._confirmation_active = True
        self._prompt_session.app.invalidate()

    def take_draft(self):
        draft = self._draft
        self._draft = None
        return draft if draft is not None else ""

    def submit(self, text: str, session_id: str, rename: bool = False):
        self._jobs.put((text, session_id, rename))
        self._prompt_session.app.invalidate()

    def close(self):
        self._jobs.put(None)
        self._worker.join()
        self._console.input = self._original_input

    def answer_confirmation(self, text: str) -> bool:
        if not self._confirmation_active:
            return False
        try:
            reply = self._confirmations.get_nowait()
        except queue.Empty:
            return False
        self._confirmation_active = False
        reply.put(text)
        self._prompt_session.app.invalidate()
        return True

    def _input(self, prompt="", *args, **kwargs):
        if threading.current_thread() is not self._worker:
            return self._original_input(prompt, *args, **kwargs)
        reply = queue.Queue(maxsize=1)
        self._confirmations.put(reply)
        loop = self._prompt_session.app.loop
        if loop is not None:
            loop.call_soon_threadsafe(self.activate_confirmation)
        self._prompt_session.app.invalidate()
        return reply.get()

    def _run(self):
        while True:
            job = self._jobs.get()
            if job is None:
                self._jobs.task_done()
                return
            text, session_id, rename = job
            try:
                started = time.monotonic()
                _show_dialogue_header(self._console, "assistant", datetime.now())
                self._agent.chat(text)
                self._console.print(f"[dim]回复用时 {time.monotonic() - started:.1f}s[/dim]")
                if rename:
                    threading.Thread(
                        target=_auto_rename_session,
                        args=(self._session_manager, session_id, text, self._agent),
                        daemon=True,
                    ).start()
            except Exception as exc:
                self._console.print(f"[bold red]错误：{exc}[/bold red]")
            finally:
                self._jobs.task_done()
                self._prompt_session.app.invalidate()


def _paint_activity_icon(output, frame):
    output.write_raw("\x1b7")
    output.cursor_goto(output.get_size().rows, 2)
    output.write_raw(f"\x1b[0;32;7m{frame}\x1b8")
    output.flush()


class OutputPane:
    def __init__(self, app):
        self._app = app
        self._text = ""
        self._lock = threading.Lock()
        self._scrollback = 0
        self._rendered_line = 0
        self._response_owner = None
        self._response_spans = []

    def write(self, text):
        with self._lock:
            if self._response_owner is threading.current_thread() and text:
                self._response_spans.append((len(self._text), len(self._text) + len(text)))
            self._text += text
            if self._scrollback:
                self._scrollback += text.count("\n")
        self._app.invalidate()
        return len(text)

    def flush(self):
        pass

    def start_response(self):
        with self._lock:
            self._response_owner = threading.current_thread()
            self._response_spans = []

    def update_response(self, rendered):
        self._replace_response(rendered, final=False)

    def finish_response(self, rendered):
        self._replace_response(rendered, final=True)

    def _replace_response(self, rendered, final):
        with self._lock:
            if not self._response_spans:
                if final:
                    self._response_owner = None
                return
            start = self._response_spans[0][0]
            previous = "".join(self._text[first:last] for first, last in self._response_spans)
            parts = [self._text[:start], rendered]
            end = self._response_spans[0][1]
            for first, last in self._response_spans[1:]:
                parts.append(self._text[end:first])
                end = last
            parts.append(self._text[end:])
            self._text = "".join(parts)
            if self._scrollback:
                self._scrollback = max(0, self._scrollback + rendered.count("\n") - previous.count("\n"))
            self._response_owner = None if final else threading.current_thread()
            self._response_spans = [] if final else [(start, start + len(rendered))]
        self._app.invalidate()

    def isatty(self):
        return True

    def formatted_text(self):
        with self._lock:
            text = self._text
            if self._scrollback:
                text = "".join(text.splitlines(keepends=True)[:-self._scrollback])
            self._rendered_line = text.count("\n")
        return ANSI(text)

    def cursor_position(self):
        with self._lock:
            return Point(x=0, y=self._rendered_line)

    def scroll(self, lines):
        with self._lock:
            max_scroll = max(0, len(self._text.splitlines(keepends=True)) - 1)
            self._scrollback = max(0, min(self._scrollback + lines, max_scroll))
        self._app.invalidate()

    def clear(self):
        with self._lock:
            self._text = ""
            self._scrollback = 0
            self._rendered_line = 0
            self._response_owner = None
            self._response_spans = []
        self._app.invalidate()


def main():
    console = Console()
    console.print(Panel("[bold cyan]Naga CLI Agent[/bold cyan]", subtitle="OpenAI 兴容模式对话助手"))

    config = load_config()

    try:
        session_manager = SessionManager(config["storage"]["db_path"])
    except Exception as e:
        console.print(f"[bold red]会话数据库初始化失败，无法启动：{e}[/bold red]")
        console.print(f"[dim]数据库路径：{config['storage']['db_path']}[/dim]")
        console.print("[dim]请确认该路径所在目录存在且有写入权限。[/dim]")
        sys.exit(1)

    model = config["model"]["default"]
    session_id = select_session(console, session_manager)

    try:
        agent = Agent(config, session_manager, session_id, model, console)
    except Exception as e:
        console.print(f"[bold red]记忆系统初始化失败，无法启动：{e}[/bold red]")
        console.print("[dim]请检查 config.toml 中 [memory] 配置，以及嵌入模型 API key 和 base_url 是否正确。[/dim]")
        sys.exit(1)
    prompt_prefix = config["agent"].get("prompt_prefix", "🤖")
    initial_history = session_manager.get_history(session_id)
    is_new_session = not initial_history

    console.print(f"\n[dim]会话：{session_id}  模型：{model}  输入 /help 查看命令[/dim]\n")

    def _format_token_limit(value: int) -> str:
        if value >= 1000:
            return f"{value // 1000}k"
        return str(value)

    frames = "◐◓◑◒"

    def _context_toolbar():
        """底部固定状态栏：显示工作状态和最近一次请求的输入用量。"""
        try:
            activity = (
                f"<ansigreen>{frames[int(time.monotonic() * 4) % len(frames)]}</ansigreen>"
                if chat_runner.busy else "<ansibrightblack>○</ansibrightblack>"
            )
            context = getattr(agent, "_context_metrics", {})
            if context.get("model") != agent.model or "prompt_estimate" not in context:
                return HTML(f" {activity}  <b>模型:</b> {agent.model}  <b>最近请求:</b> 尚未计算 ")
            used = context["prompt_estimate"]
            max_ctx = context["window"]
            pct = used / max_ctx * 100 if max_ctx > 0 else 0
            if pct < 60:
                color = "ansigreen"
            elif pct < 80:
                color = "ansiyellow"
            else:
                color = "ansired"
            max_k = _format_token_limit(max_ctx)
            return HTML(
                f" {activity}  <b>模型:</b> {agent.model}  "
                f"<b>最近请求:</b> <{color}>{used:,} / {max_k} ({pct:.1f}%)</{color}> "
            )
        except Exception:
            return ""

    prompt_session = PromptSession(erase_when_done=True)
    output_pane = OutputPane(prompt_session.app)
    layout = prompt_session.app.layout
    output_window = Window(
        FormattedTextControl(output_pane.formatted_text, get_cursor_position=output_pane.cursor_position),
        wrap_lines=True,
        height=Dimension(weight=1),
    )
    layout.container.height = Dimension(min=3, preferred=3, max=6)
    prompt_session.app.layout = Layout(HSplit([output_window, layout.container]), focused_element=layout.current_window)
    bindings = KeyBindings()

    def _scroll_output(lines):
        output_window.vertical_scroll = max(0, output_window.vertical_scroll - lines)
        output_pane.scroll(lines)

    @bindings.add("up")
    def _scroll_up(event):
        _scroll_output(1)

    @bindings.add("down")
    def _scroll_down(event):
        _scroll_output(-1)

    @bindings.add("pageup")
    def _page_up(event):
        page = output_window.render_info.window_height if output_window.render_info else 10
        _scroll_output(page)

    @bindings.add("pagedown")
    def _page_down(event):
        page = output_window.render_info.window_height if output_window.render_info else 10
        _scroll_output(-page)

    prompt_session.key_bindings = bindings
    original_console_file = console.file
    console.file = output_pane
    chat_runner = ChatRunner(agent, console, prompt_session, session_manager)
    _show_dialogue_history(console, initial_history)

    def _input_prompt():
        label = "确认命令 (y/N) > " if chat_runner.needs_confirmation else f"{prompt_prefix} > "
        width = max(0, prompt_session.app.output.get_size().columns - 2)
        return ANSI(f" \x1b[90m{'─' * width}\x1b[0m\n{label}")

    async def _animate_status():
        while True:
            await asyncio.sleep(0.25)
            if not chat_runner.busy or prompt_session.app._running_in_terminal:
                continue
            output = prompt_session.app.output
            _paint_activity_icon(output, frames[int(time.monotonic() * 4) % len(frames)])

    def _start_prompt():
        chat_runner.activate_confirmation()
        prompt_session.app.create_background_task(_animate_status())

    while True:
        try:
            with patch_stdout(raw=True):
                user_input = prompt_session.prompt(
                    _input_prompt,
                    bottom_toolbar=_context_toolbar,
                    default=chat_runner.take_draft(),
                    pre_run=_start_prompt,
                ).strip()
        except (KeyboardInterrupt, EOFError):
            console.file = original_console_file
            console.print("\n[dim]已退出。[/dim]")
            if not chat_runner.busy:
                chat_runner.close()
            break

        if chat_runner.answer_confirmation(user_input):
            continue

        if not user_input:
            continue

        invoked_skill = _invoked_skill(user_input, agent._skills)

        if chat_runner.busy:
            if user_input.startswith("/") and not (invoked_skill and invoked_skill["enabled"]):
                console.print("[yellow]正在回复，请等待完成后再执行命令。[/yellow]")
            else:
                _show_dialogue_message(console, "user", user_input, datetime.now())
                chat_runner.submit(user_input, agent.session_id)
                console.print("[dim]消息已排队，当前回复完成后发送。[/dim]")
            continue

        if user_input in ("/quit", "/exit"):
            chat_runner.close()
            console.file = original_console_file
            console.print("[dim]再见！[/dim]")
            break

        elif user_input == "/help":
            console.print(HELP_TEXT)

        elif user_input == "/confirm on":
            agent.auto_confirm = True
            console.print("[green]自动确认已开启[/green]")

        elif user_input == "/confirm off":
            agent.auto_confirm = False
            console.print("[green]自动确认已关闭（需手动确认）[/green]")

        elif user_input == "/thinking on":
            agent.config["model"].setdefault("extra_params_default", {})["enable_thinking"] = True
            for m in agent.config["model"].get("extra_params", {}).values():
                m["enable_thinking"] = True
            console.print("[green]思维链已开启[/green]")

        elif user_input == "/thinking off":
            agent.config["model"].get("extra_params_default", {}).pop("enable_thinking", None)
            agent.config["model"].get("extra_params_default", {}).pop("thinking_budget", None)
            for m in agent.config["model"].get("extra_params", {}).values():
                m.pop("enable_thinking", None)
                m.pop("thinking_budget", None)
            console.print("[green]思维链已关闭[/green]")

        elif user_input == "/thinking show":
            agent.config["agent"]["show_thinking"] = True
            console.print("[green]思考过程显示已开启[/green]")

        elif user_input == "/thinking hide":
            agent.config["agent"]["show_thinking"] = False
            console.print("[green]思考过程显示已关闭[/green]")

        elif user_input.startswith("/model "):
            new_model = user_input[7:].strip()
            if new_model == "auto":
                agent.model = config["model"]["default"]
                console.print(f"[green]已恢复默认主模型：{agent.model}[/green]")
            elif new_model in config["model"]["available"]:
                agent.model = new_model
                console.print(f"[green]主模型已切换为：{new_model}[/green]")
            else:
                console.print(f"[red]不支持的模型：{new_model}[/red]")
                console.print(f"可用模型：{', '.join(config['model']['available'])}")

        elif user_input == "/delegate on":
            if agent._subtask_runner is None:
                console.print("[yellow]低成本子模型未配置或无法连接，无法开启分派。[/yellow]")
                continue
            agent.config["delegation"]["enabled"] = True
            console.print("[green]子任务分派已开启[/green]")

        elif user_input == "/delegate off":
            agent.config["delegation"]["enabled"] = False
            console.print("[green]子任务分派已关闭[/green]")

        elif user_input == "/token":
            summary = agent.get_token_summary()
            if summary.get("turns", 0) == 0:
                console.print("[dim]本会话暂无 token 记录。[/dim]")
            else:
                console.print(f"\n[bold]Token 用量明细（API 返回值）[/bold]  (本地估算器: {summary['tokenizer_mode']})")
                for model_name, s in summary.get("per_model", {}).items():
                    console.print(
                        f"  {model_name}  {s['turns']} 轮  "
                        f"输入 {s['input']}  输出 {s['output']}"
                    )
                console.print(
                    f"  [bold]合计[/bold]  {summary['turns']} 轮  "
                    f"输入 {summary['total_input']}  输出 {summary['total_output']}  "
                    f"总计 {summary['total_tokens']} tokens"
                )
            context = summary.get("context", {})
            if context:
                console.print(
                    f"  最近请求  窗口 {context['window']}  消息预算 {context['input_budget']}  "
                    f"消息 {context.get('message_tokens', 0)}  工具 schema {context['tool_tokens']}  "
                    f"输入估算合计 {context['prompt_estimate']}  "
                    f"摘要截至消息 #{context['summary_through_id']}  估算器 {context['tokenizer_mode']}"
                )
                if context.get("actual_prompt_tokens") is not None:
                    console.print(f"  API 最近真实输入 {context['actual_prompt_tokens']} tokens")

        elif user_input == "/session new":
            new_id = session_manager.create_session("new")
            agent.switch_session(new_id)
            is_new_session = True
            output_pane.clear()
            console.print(f"[green]已创建并切换到新会话：{new_id}[/green]")

        elif user_input == "/session list":
            sessions = session_manager.list_sessions()[:10]
            if not sessions:
                console.print("[dim]暂无历史会话。[/dim]")
            else:
                console.print("\n[bold]历史会话（最近 10 条）：[/bold]")
                for i, s in enumerate(sessions, 1):
                    marker = " [bold cyan]← 当前[/bold cyan]" if s["id"] == agent.session_id else ""
                    console.print(
                        f"  {i}. [{s['created_at'][:19]}] {s['name']}"
                        f"  [dim]({s['message_count']} 条消息)[/dim]{marker}"
                    )

        elif user_input.startswith("/session switch "):
            arg = user_input[16:].strip()
            sessions = session_manager.list_sessions()[:10]
            try:
                idx = int(arg) - 1
                if 0 <= idx < len(sessions):
                    target = sessions[idx]
                    agent.switch_session(target["id"])
                    history = session_manager.get_history(target["id"])
                    is_new_session = not history
                    output_pane.clear()
                    console.print(f"[green]已切换到会话：{target['name']}（{target['id']}）[/green]")
                    _show_dialogue_history(console, history)
                else:
                    console.print(f"[red]编号超出范围，请用 /session list 查看有效编号。[/red]")
            except ValueError:
                console.print("[red]用法：/session switch <编号>，编号为整数。[/red]")

        elif user_input == "/session":
            history = session_manager.get_history(agent.session_id)
            current_prompt = agent.config["agent"].get("system_prompt", "")
            summary = agent.get_token_summary()
            console.print(f"会话 ID：{agent.session_id}")
            console.print(f"当前模型：{agent.model}")
            console.print(f"消息总数：{len(history)}")
            console.print(f"自动确认：{'是' if agent.auto_confirm else '否'}")
            console.print(f"子任务分派：{'开启' if agent.config.get('delegation', {}).get('enabled') and agent._subtask_runner else '关闭'}")
            console.print(f"系统提示词：{current_prompt[:80] or '[未设置]'}{'...' if len(current_prompt) > 80 else ''}")
            if summary.get("turns", 0) > 0:
                console.print(
                    f"Token 消耗：{summary['total_tokens']} tokens  "
                    f"(输入 {summary['total_input']} / 输出 {summary['total_output']})"
                )

        elif user_input.startswith("/prompt"):
            sub = user_input[7:].strip()
            if not sub:
                current = agent.config["agent"].get("system_prompt", "")
                console.print(f"当前系统提示词：{current or '[未设置]'}")
            elif sub.startswith("set "):
                new_prompt = sub[4:].strip()
                agent.config["agent"]["system_prompt"] = new_prompt
                console.print(f"[green]系统提示词已更新。[/green]")
            elif sub == "clear":
                agent.config["agent"]["system_prompt"] = ""
                console.print("[green]系统提示词已清除。[/green]")
            else:
                console.print("[yellow]用法：/prompt | /prompt set <内容> | /prompt clear[/yellow]")

        elif user_input == "/history":
            history = session_manager.get_history(session_id)
            recent = history[-5:] if len(history) >= 5 else history
            if not recent:
                console.print("[dim]暂无历史消息。[/dim]")
            else:
                for msg in recent:
                    role = msg["role"]
                    content = msg["content"][:80].replace("\n", " ")
                    ellipsis = "..." if len(msg["content"]) > 80 else ""
                    console.print(f"[dim][{role}][/dim] {content}{ellipsis}")

        elif user_input == "/clear":
            console.file = original_console_file
            try:
                confirm = Prompt.ask("确认清除当前会话所有消息？(y/N)", default="N", console=console).strip().lower()
            finally:
                console.file = output_pane
            if confirm == "y":
                session_manager.clear_messages(session_id)
                agent._history_cache = None
                output_pane.clear()
                console.print("[green]会话消息已清除。[/green]")
            else:
                console.print("[dim]已取消。[/dim]")

        elif user_input in ("/skill", "/skill list"):
            if not agent._skills:
                console.print("[dim]skills/ 目录为空，暂无可用 skill。[/dim]")
            else:
                console.print("\n[bold]Skill 列表[/bold]")
                for s in agent._skills:
                    status = "[green]• 已激活[/green]" if s["enabled"] else "[dim]◦ 未激活[/dim]"
                    desc = f"  {s['description']}" if s["description"] else ""
                    console.print(f"  {status}  [bold]{s['name']}[/bold]{desc}")

        elif user_input.startswith("/skill on "):
            name = user_input[10:].strip()
            if agent.toggle_skill(name, True):
                console.print(f"[green]Skill \'{name}\' 已激活。[/green]")
            else:
                console.print(f"[red]未找到 skill：{name}，请用 /skill list 查看可用列表。[/red]")

        elif user_input.startswith("/skill off "):
            name = user_input[11:].strip()
            if agent.toggle_skill(name, False):
                console.print(f"[dim]Skill \'{name}\' 已停用。[/dim]")
            else:
                console.print(f"[red]未找到 skill：{name}，请用 /skill list 查看可用列表。[/red]")

        elif user_input.startswith("/") and invoked_skill and not invoked_skill["enabled"]:
            console.print(f"[yellow]Skill '{invoked_skill['name']}' 已停用；请先用 /skill on {invoked_skill['name']} 启用。[/yellow]")

        elif user_input.startswith("/") and not invoked_skill:
            console.print("[yellow]未知命令，输入 /help 查看可用命令。[/yellow]")

        else:
            _show_dialogue_message(console, "user", user_input, datetime.now())
            chat_runner.submit(user_input, agent.session_id, rename=is_new_session)
            is_new_session = False


if __name__ == "__main__":
    main()
