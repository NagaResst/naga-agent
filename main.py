import asyncio
import bisect
import sys
import os
import queue
import re
import threading
import time
from datetime import datetime
from io import StringIO

sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI, HTML, to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_width
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.data_structures import Point
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.shortcuts.prompt import CompleteStyle
from prompt_toolkit.filters import has_completions

from agent.config import load_config
from agent.core import Agent, InstructionBuffer, markdown_body
from agent.status import ActivityTracker
from agent.slash_completer import SlashCompleter
from agent.context_window import get_context_window
from session.sqlite_session import SessionManager


HELP_TEXT = """
可用内置命令：
  F2                  中断当前任务（正在执行的同步调用返回后生效）
  F3                  折叠或展开思考过程
  F4                  切换排队 / 插入模式
  /quit, /exit        退出程序
  /confirm on         开启自动确认（命令无需手动确认）
  /confirm off        关闭自动确认（命令需手动确认，默认）
  /thinking on        开启思维链模式（需模型在 extra_params 中配置 enable_thinking）
  /thinking off       关闭思维链模式
  /thinking show      展开思考过程
  /thinking hide      折叠思考过程
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
    /<Skill 名称> <任务> 显式使用 Skill（停用的会自动激活）
  /history            显示最近 5 条对话摘要
  /clear              清除当前会话的所有历史消息
  /help               显示此帮助信息
"""


def _skill_help_lines(skills: list[dict]) -> list[str]:
    """生成 /help 中动态 skill 段：可 /name 调用的已激活 skill。"""
    enabled = [s for s in skills if s.get("enabled")]
    if not enabled:
        return []
    lines = ["", "已激活 Skill（可用 /<名称> <任务> 显式调用）："]
    for s in enabled:
        desc = (s.get("description") or "").strip()
        lines.append(f"  /{s['name']}" + (f"  — {desc[:60]}" if desc else ""))
    return lines


def _invoked_skill(user_input: str, agent) -> dict | None:
    """识别 /<skill 名> <任务> 形式的显式调用；未命中返回 None。"""
    if not user_input.startswith("/"):
        return None
    name = user_input[1:].split(maxsplit=1)[0]
    return agent.find_skill(name)


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


def _show_dialogue_history(console: Console, history: list[dict], trailing_blank=True):
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
    if trailing_blank:
        console.print()


class DialogueHistoryPager:
    def __init__(self, session_manager, session_id, page_size=40):
        self._session_manager = session_manager
        self._session_id = session_id
        self._page_size = page_size
        self._before_id = None
        self.exhausted = False

    def next_page(self):
        if self.exhausted:
            return []
        messages = []
        before_id = self._before_id
        while True:
            batch = self._session_manager.get_history_before(self._session_id, before_id, self._page_size)
            if not batch:
                self.exhausted = True
                self._before_id = None
                return messages
            messages = batch + messages
            visible_users = [
                i for i, message in enumerate(messages)
                if message["role"] == "user" and not message.get("tool_calls")
                and (message.get("content") or "").strip()
            ]
            if visible_users:
                first = visible_users[0]
                self._before_id = messages[first]["id"]
                return messages[first:]
            before_id = batch[0]["id"]


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
        self._job_lock = threading.Lock()
        self._current_cancel = None
        self._insertions = InstructionBuffer()
        self._agent._insertions = self._insertions
        self._original_input = console.input
        console.input = self._input
        self._closing = False
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
        self._agent.activity.set("confirm_wait")
        self._prompt_session.app.invalidate()

    def take_draft(self):
        draft = self._draft
        self._draft = None
        return draft if draft is not None else ""

    def submit(self, text: str, session_id: str, rename: bool = False):
        self._jobs.put((text, session_id, rename))
        self._prompt_session.app.invalidate()

    def insert(self, text: str, session_id: str) -> bool:
        with self._job_lock:
            if (self._current_cancel is not None and not self._current_cancel.is_set()
                    and self._insertions.insert(text)):
                return True
            self._jobs.put((text, session_id, False))
        self._prompt_session.app.invalidate()
        return False

    def cancel_current(self) -> bool:
        with self._job_lock:
            if self._current_cancel is None:
                return False
            self._current_cancel.set()
            try:
                reply = self._confirmations.get_nowait()
            except queue.Empty:
                reply = None
            self._confirmation_active = False
            if reply is not None:
                reply.put("n")
            self._agent.cancel_active_stream()
        self._prompt_session.app.invalidate()
        return True

    def close(self):
        with self._job_lock:
            self._closing = True
        if self.busy:
            # 中断当前任务：worker 会把已生成的部分回复落库后结束本轮
            self.cancel_current()
            # 丢弃排队中的任务（其文本尚未写入会话，退出时不再处理）
            while True:
                try:
                    self._jobs.get_nowait()
                except queue.Empty:
                    break
                self._jobs.task_done()
            # 等待 worker 完成中断与落库；若 worker 刚取走任务尚未注册取消事件则重试取消
            deadline = time.monotonic() + 10
            while self.busy and self._worker.is_alive() and time.monotonic() < deadline:
                self.cancel_current()
                self._worker.join(0.1)
            if self.busy:
                self._console.print(
                    "等待任务中断超时，未完成的回复可能未保存。",
                    style="yellow",
                )
                return
        self._jobs.put(None)
        self._worker.join(10)
        self._console.input = self._original_input

    def answer_confirmation(self, text: str) -> bool:
        if not self._confirmation_active:
            return False
        try:
            reply = self._confirmations.get_nowait()
        except queue.Empty:
            return False
        self._confirmation_active = False
        # 确认后命令继续执行，恢复为命令执行态
        self._agent.activity.set("cmd_exec", "确认后继续")
        reply.put(text)
        self._prompt_session.app.invalidate()
        return True

    def _input(self, prompt="", *args, **kwargs):
        if threading.current_thread() is not self._worker:
            return self._original_input(prompt, *args, **kwargs)
        reply = queue.Queue(maxsize=1)
        with self._job_lock:
            if self._current_cancel is not None and self._current_cancel.is_set():
                return "n"
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
            if self._closing:
                # 退出中：丢弃排队任务（其文本尚未写入会话）
                self._jobs.task_done()
                continue
            cancel_event = threading.Event()
            with self._job_lock:
                self._current_cancel = cancel_event
                self._agent._cancel_event = cancel_event
                self._insertions.open()
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
                self._console.print(f"错误：{exc}", style="bold red", markup=False, highlight=False)
            finally:
                with self._job_lock:
                    pending = self._insertions.close()
                    for instruction in pending:
                        if not self._closing:
                            self._jobs.put((instruction, session_id, False))
                    self._current_cancel = None
                if pending:
                    if self._closing:
                        self._console.print(
                            f"程序退出，{len(pending)} 条未发送的插入指令已丢弃。",
                            style="yellow",
                        )
                    else:
                        self._console.print(
                            f"当前任务已结束或中断，{len(pending)} 条未发送的插入指令已转入队列。",
                            style="yellow",
                        )
                self._agent.activity.set("idle")
                self._jobs.task_done()
                self._prompt_session.app.invalidate()


def _truncate_fragments(fragments, width):
    result = []
    remaining = max(0, width)
    for style, text, *extra in fragments:
        chars = []
        for char in text:
            char_width = get_cwidth(char)
            if char_width > remaining:
                return result + ([(style, "".join(chars), *extra)] if chars else [])
            chars.append(char)
            remaining -= char_width
        if chars:
            result.append((style, "".join(chars), *extra))
    return result


def _status_shortcuts(input_mode, thinking_expanded, width):
    mode = "插入" if input_mode == "insert" else "排队"
    thinking = "展开" if thinking_expanded else "折叠"
    full = f"F2 中断  F3 思考:{thinking}  F4 {mode} "
    compact = f"F2中断 F3{thinking} F4{mode}"
    if get_cwidth(full) <= width:
        return full
    if get_cwidth(compact) <= width:
        return compact
    return f"F2停 F3{thinking[0]} F4{mode[0]}"


def _format_status_toolbar(left, shortcuts, width):
    width = max(0, width)
    right = _truncate_fragments([("", shortcuts)], width)
    right_width = fragment_list_width(right)
    left = to_formatted_text(left)
    left_limit = max(0, width - right_width - 1)
    if fragment_list_width(left) > left_limit:
        left = _truncate_fragments(left, max(0, left_limit - 1))
        if left_limit:
            left.append(("", "…"))
    padding = max(0, width - fragment_list_width(left) - right_width)
    return left + [("", " " * padding)] + right


def _paint_status_line(output, frame, status_text, max_width=None):
    """流式输出期间直画状态行（保底路径，与 bottom_toolbar 内容一致）。"""
    fragments = [("", f"{frame} {status_text}")]
    if max_width is not None:
        fragments = _truncate_fragments(fragments, max_width)
    text = "".join(fragment[1] for fragment in fragments)
    if not text:
        return
    output.write_raw("\x1b7")
    output.cursor_goto(output.get_size().rows, 2)
    output.write_raw(f"\x1b[0;32;7m{text}\x1b8")
    output.flush()


class _OutputBuffer:
    def __init__(self):
        self.chunks = []
        self.ends = []

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def append(self, text):
        if text:
            self.chunks.append(text)
            self.ends.append(len(self) + len(text))

    def slice(self, start, end):
        if start >= end:
            return ""
        index = bisect.bisect_right(self.ends, start)
        position = self.ends[index - 1] if index else 0
        parts = []
        while position < end and index < len(self.chunks):
            chunk = self.chunks[index]
            parts.append(chunk[max(0, start - position):end - position])
            position = self.ends[index]
            index += 1
        return "".join(parts)

    def truncate(self, end):
        if end >= len(self):
            return
        index = bisect.bisect_right(self.ends, end)
        position = self.ends[index - 1] if index else 0
        remainder = self.chunks[index][:end - position]
        del self.chunks[index:]
        del self.ends[index:]
        self.append(remainder)

    def prepend(self, text):
        if text:
            self.chunks.insert(0, text)
            self.ends = [len(text)] + [end + len(text) for end in self.ends]


class _AnsiStyleState:
    def __init__(self):
        self.attributes = {}

    def update(self, sequence):
        parameters = sequence[2:-1] if sequence.startswith("\x1b[") else sequence[1:-1]
        codes = [min(int(part or 0), 9999) for part in parameters.split(";")]
        index = 0
        while index < len(codes):
            code = codes[index]
            index += 1
            if code == 0:
                self.attributes.clear()
            elif code in (38, 48) and index < len(codes):
                mode = codes[index]
                count = 2 if mode == 5 else 4 if mode == 2 else 1
                if mode in (2, 5) and len(codes) - index >= count:
                    self.attributes["fg" if code == 38 else "bg"] = [code] + codes[index:index + count]
                index += min(count, len(codes) - index)
            elif 30 <= code <= 37 or 90 <= code <= 97 or code == 39:
                self.attributes["fg"] = [code]
            elif 40 <= code <= 47 or 100 <= code <= 107 or code == 49:
                self.attributes["bg"] = [code]
            elif code in (1, 2, 3, 4, 5, 6, 7, 8, 9):
                key = 5 if code == 6 else code
                self.attributes[key] = [code]
            elif code == 22:
                self.attributes.pop(1, None)
                self.attributes.pop(2, None)
            elif code in (23, 24, 25, 27, 28, 29):
                self.attributes.pop(code - 20, None)

    def prefix(self):
        codes = [str(code) for value in self.attributes.values() for code in value]
        return "\x1b[" + ";".join(codes) + "m" if codes else ""


class OutputPane:
    _VISIBLE_LINES = 200
    _SGR_PATTERN = re.compile(r"(?:\x1b\[|\x9b)[0-9;]*m")
    _NEWLINE_PATTERN = re.compile("\n")
    _FOLDED_REASONING = "\x1b[2m[思考过程已折叠，F3 展开]\x1b[0m"

    def __init__(self, app, show_thinking=True):
        self._app = app
        self._buffer = _OutputBuffer()
        self._lock = threading.Lock()
        self._scrollback = 0
        self._rendered_line = 0
        self._response_owner = None
        self._response_spans = []
        self._reasoning_segments = []
        self._active_reasoning = None
        self._show_thinking = show_thinking
        self._formatted_cache = None
        self._line_starts = [0]
        self._style_offset = 0
        self._style_state = _AnsiStyleState()

    def _index_from(self, start):
        del self._line_starts[bisect.bisect_right(self._line_starts, start):]
        self._line_starts.extend(
            start + match.end() for match in self._NEWLINE_PATTERN.finditer(
                self._buffer.slice(start, len(self._buffer))
            )
        )
        if start < self._style_offset:
            self._style_offset = 0
            self._style_state = _AnsiStyleState()

    def _prefix_at(self, start):
        if start < self._style_offset:
            self._style_offset = 0
            self._style_state = _AnsiStyleState()
        for match in self._SGR_PATTERN.finditer(self._buffer.slice(self._style_offset, start)):
            self._style_state.update(match.group())
        self._style_offset = start
        return self._style_state.prefix()

    def write(self, text):
        with self._lock:
            start = len(self._buffer)
            old_lines = self._line_count()
            if self._response_owner is threading.current_thread() and text:
                self._response_spans.append((start, start + len(text)))
            self._buffer.append(text)
            self._index_from(start)
            self._formatted_cache = None
            if self._scrollback:
                self._scrollback += self._line_count() - old_lines
        self._app.invalidate()
        return len(text)

    def prepend(self, text):
        if not text:
            return
        with self._lock:
            self._buffer.prepend(text)
            self._response_spans = [(start + len(text), end + len(text)) for start, end in self._response_spans]
            for segment in self._reasoning_segments:
                segment["start"] += len(text)
                segment["end"] += len(text)
            self._line_starts = [0]
            self._style_offset = 0
            self._style_state = _AnsiStyleState()
            self._index_from(0)
            self._formatted_cache = None
        self._app.invalidate()

    def flush(self):
        pass

    def start_reasoning(self):
        with self._lock:
            self._active_reasoning = None

    def write_reasoning(self, text):
        if not text:
            return
        with self._lock:
            segment = self._active_reasoning
            start = len(self._buffer)
            if segment is None or segment["end"] != start:
                segment = {"start": start, "end": start, "parts": []}
                self._reasoning_segments.append(segment)
                self._active_reasoning = segment
                displayed = text if self._show_thinking else self._FOLDED_REASONING
            else:
                displayed = text if self._show_thinking else ""
            segment["parts"].append(text)
            if displayed:
                old_lines = self._line_count()
                self._buffer.append(displayed)
                segment["end"] += len(displayed)
                self._index_from(start)
                self._formatted_cache = None
                if self._scrollback:
                    self._scrollback += self._line_count() - old_lines
        if displayed:
            self._app.invalidate()

    def end_reasoning(self):
        with self._lock:
            self._active_reasoning = None

    def _splice(self, start, end, text, target=None):
        tail = self._buffer.slice(end, len(self._buffer))
        old = self._buffer.slice(start, end)
        self._buffer.truncate(start)
        self._buffer.append(text)
        self._buffer.append(tail)
        shift = len(text) - (end - start)
        self._response_spans = [
            (first + shift, last + shift) if first >= end else (first, last)
            for first, last in self._response_spans
        ]
        for segment in self._reasoning_segments:
            if segment is target:
                segment["end"] = start + len(text)
            elif segment["start"] >= end:
                segment["start"] += shift
                segment["end"] += shift
        if self._scrollback:
            self._scrollback = max(0, self._scrollback + text.count("\n") - old.count("\n"))

    def set_thinking_expanded(self, expanded):
        with self._lock:
            if expanded == self._show_thinking:
                return
            self._show_thinking = expanded
            for segment in reversed(self._reasoning_segments):
                text = "".join(segment["parts"]) if expanded else self._FOLDED_REASONING
                self._splice(segment["start"], segment["end"], text, target=segment)
            if self._reasoning_segments:
                self._index_from(self._reasoning_segments[0]["start"])
                self._formatted_cache = None
        self._app.invalidate()

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
            spans = self._response_spans[:]
            for first, last in reversed(spans[1:]):
                self._splice(first, last, "")
            self._splice(start, spans[0][1], rendered)
            self._index_from(start)
            self._formatted_cache = None
            self._response_owner = None if final else threading.current_thread()
            self._response_spans = [] if final else [(start, start + len(rendered))]
        self._app.invalidate()

    def isatty(self):
        return True

    def _line_count(self):
        end = len(self._buffer)
        return len(self._line_starts) - int(self._buffer.slice(end - 1, end) == "\n")

    @property
    def _text(self):
        return self._buffer.slice(0, len(self._buffer))

    def formatted_text(self):
        with self._lock:
            if self._formatted_cache is None:
                end = len(self._buffer)
                if self._scrollback:
                    line_count = self._line_count()
                    end = self._line_starts[line_count - self._scrollback]
                end_line = bisect.bisect_left(self._line_starts, end)
                start = self._line_starts[max(0, end_line - self._VISIBLE_LINES)]
                text = self._prefix_at(start) + self._buffer.slice(start, end)
                self._rendered_line = text.count("\n")
                self._formatted_cache = ANSI(text)
            return self._formatted_cache

    def cursor_position(self):
        with self._lock:
            return Point(x=0, y=self._rendered_line)

    def scroll(self, lines):
        with self._lock:
            line_count = self._line_count()
            max_scroll = max(0, line_count - 1)
            requested = self._scrollback + lines
            self._scrollback = max(0, min(requested, max_scroll))
            remaining = max(0, requested - max_scroll)
            self._formatted_cache = None
        self._app.invalidate()
        return remaining

    def clear(self):
        with self._lock:
            self._buffer = _OutputBuffer()
            self._scrollback = 0
            self._rendered_line = 0
            self._response_owner = None
            self._response_spans = []
            self._reasoning_segments = []
            self._active_reasoning = None
            self._formatted_cache = None
            self._line_starts = [0]
            self._style_offset = 0
            self._style_state = _AnsiStyleState()
        self._app.invalidate()


_BANNER_TITLE = "Naga CLI Agent"
_BANNER_DESCRIPTION = "终端 AI 助手 · 输入 /help 查看命令"


def _configure_prompt_layout(prompt_session, output_pane):
    # Keep the input shrinkable; reserve menu space in the outer layout instead.
    prompt_session.reserve_space_for_menu = 0
    layout = prompt_session.app.layout

    def banner_height():
        size = prompt_session.app.output.get_size()
        if size.rows < 5:
            return 0
        if size.rows < 12 or size.columns < get_cwidth(_BANNER_DESCRIPTION) + 4:
            return 1
        return 4

    def banner_text():
        width = prompt_session.app.output.get_size().columns
        if banner_height() < 4:
            return [("bold ansicyan", f" {_BANNER_TITLE}"), ("ansibrightblack", " · /help")]
        border = "─" * (width - 2)
        fragments = [("ansicyan", f"╭{border}╮\n")]
        for text, style in ((_BANNER_TITLE, "bold ansicyan"), (_BANNER_DESCRIPTION, "ansibrightblack")):
            fragments.extend([
                ("ansicyan", "│ "),
                (style, text),
                ("ansicyan", " " * (width - get_cwidth(text) - 3) + "│\n"),
            ])
        fragments.append(("ansicyan", f"╰{border}╯"))
        return fragments

    banner = Window(FormattedTextControl(banner_text), height=banner_height, wrap_lines=False)
    output_window = Window(
        FormattedTextControl(output_pane.formatted_text, get_cursor_position=output_pane.cursor_position),
        wrap_lines=True,
        height=lambda: Dimension(preferred=prompt_session.app.output.get_size().rows, weight=1),
    )
    layout.container.height = lambda: Dimension(
        min=3, preferred=8 if prompt_session.default_buffer.complete_state else 3, max=8,
    )
    prompt_session.app.layout = Layout(
        HSplit([banner, output_window, layout.container]), focused_element=layout.current_window,
    )
    return output_window


def main():
    console = Console()
    console.print(Panel(
        f"[bold cyan]{_BANNER_TITLE}[/bold cyan]\n[dim]{_BANNER_DESCRIPTION}[/dim]",
        border_style="cyan",
    ))

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
        console.print(f"[bold red]Agent 初始化失败，无法启动：{e}[/bold red]")
        console.print("[dim]请检查模型、子 Agent 和记忆配置，以及所需的环境变量。[/dim]")
        sys.exit(1)
    prompt_prefix = config["agent"].get("prompt_prefix", "🤖")
    history_pager = DialogueHistoryPager(session_manager, session_id)
    initial_history = history_pager.next_page()
    is_new_session = not initial_history

    console.print(f"\n[dim]会话：{session_id}  模型：{model}  输入 /help 查看命令[/dim]\n")

    def _format_token_limit(value: int) -> str:
        if value >= 1000:
            return f"{value // 1000}k"
        return str(value)

    frames = "◐◓◑◒"
    input_mode = "queue"

    def _context_toolbar():
        """底部固定状态栏：显示当前阶段 + 最近一次请求的输入用量。"""
        try:
            background = agent.activity.get_background()
            background_text = (
                f"  <ansicyan>⟳{ActivityTracker.LABELS.get(background, background)}</ansicyan>"
                if background else ""
            )
            if chat_runner.busy:
                phase, _, elapsed = agent.activity.get()
                label = ActivityTracker.LABELS.get(phase, phase)
                status = f"{label} {elapsed:.0f}s"
                activity = (
                    f"<ansigreen>{frames[int(time.monotonic() * 4) % len(frames)]}</ansigreen>"
                    f" <b>{status}</b>"
                )
            else:
                activity = "<ansibrightblack>○ 待命</ansibrightblack>"
            activity += background_text
            context = getattr(agent, "_context_metrics", {})
            if context.get("model") != agent.model or "prompt_estimate" not in context:
                left = HTML(f" {activity}  <b>模型:</b> {agent.model}  <b>最近请求:</b> 尚未计算 ")
            else:
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
                left = HTML(
                    f" {activity}  <b>模型:</b> {agent.model}  "
                    f"<b>最近请求:</b> <{color}>{used:,} / {max_k} ({pct:.1f}%)</{color}> "
                )
            width = prompt_session.app.output.get_size().columns
            shortcuts = _status_shortcuts(
                input_mode, agent.config["agent"].get("show_thinking", True), width
            )
            return _format_status_toolbar(left, shortcuts, width)
        except Exception:
            return ""

    prompt_session = PromptSession(erase_when_done=True, completer=SlashCompleter(agent, config), complete_style=CompleteStyle.COLUMN)
    output_pane = OutputPane(prompt_session.app, show_thinking=agent.config["agent"].get("show_thinking", True))
    output_window = _configure_prompt_layout(prompt_session, output_pane)
    bindings = KeyBindings()

    def _scroll_output(lines):
        output_window.vertical_scroll = max(0, output_window.vertical_scroll - lines)
        remaining = output_pane.scroll(lines)
        while remaining and not history_pager.exhausted:
            page = history_pager.next_page()
            if not page:
                break
            buffer = StringIO()
            history_console = Console(
                file=buffer, width=console.width, force_terminal=console.is_terminal,
                color_system=console.color_system, legacy_windows=console.legacy_windows,
            )
            _show_dialogue_history(history_console, page, trailing_blank=False)
            output_pane.prepend(buffer.getvalue())
            remaining = output_pane.scroll(remaining)

    @bindings.add("up", filter=~has_completions)
    def _scroll_up(event):
        _scroll_output(1)

    @bindings.add("down", filter=~has_completions)
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

    @bindings.add("f2")
    def _cancel_task(event):
        if chat_runner.cancel_current():
            draft = chat_runner.take_draft()
            if draft:
                event.app.current_buffer.set_document(draft)
            console.print(
                "已请求中断；模型流输出将停止，阻塞工具或远程操作可能仍在运行且可能产生费用。",
                style="yellow",
            )
        else:
            console.print("当前没有可中断的任务。", style="dim")

    @bindings.add("f3")
    def _toggle_thinking(event):
        enabled = not agent.config["agent"].get("show_thinking", True)
        agent.config["agent"]["show_thinking"] = enabled
        output_pane.set_thinking_expanded(enabled)
        event.app.invalidate()

    @bindings.add("f4")
    def _toggle_input_mode(event):
        nonlocal input_mode
        input_mode = "insert" if input_mode == "queue" else "queue"
        console.print(f"输入模式：{'插入' if input_mode == 'insert' else '排队'}。", style="green")
        event.app.invalidate()

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
            phase, _, elapsed = agent.activity.get()
            label = ActivityTracker.LABELS.get(phase, phase)
            status = f"{label} {elapsed:.0f}s"
            background = agent.activity.get_background()
            if background:
                status += f"  ⟳{ActivityTracker.LABELS.get(background, background)}"
            output = prompt_session.app.output
            width = output.get_size().columns
            shortcuts = _status_shortcuts(
                input_mode, agent.config["agent"].get("show_thinking", True), width
            )
            max_width = max(0, width - get_cwidth(shortcuts) - 2)
            _paint_status_line(output, frames[int(time.monotonic() * 4) % len(frames)], status, max_width)
            # busy 时阶段/秒数会变化，需周期性刷新工具栏；空闲时无需重绘
            prompt_session.app.invalidate()

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
            console.print("")
            if chat_runner.busy:
                console.print("[dim]正在中断当前任务并保存对话记录...[/dim]")
            chat_runner.close()
            console.print("[dim]已退出。[/dim]")
            break

        if chat_runner.answer_confirmation(user_input):
            continue

        if not user_input:
            continue

        invoked_skill = _invoked_skill(user_input, agent)

        if chat_runner.busy:
            if user_input.startswith("/") and not invoked_skill:
                console.print("[yellow]正在回复，请等待完成后再执行命令。[/yellow]")
            else:
                _show_dialogue_message(console, "user", user_input, datetime.now())
                if input_mode == "insert":
                    if chat_runner.insert(user_input, agent.session_id):
                        console.print("[dim]指令已插入，将在下一次模型请求时发送。[/dim]")
                    else:
                        console.print("[dim]当前没有可插入的任务，消息已转入队列。[/dim]")
                else:
                    chat_runner.submit(user_input, agent.session_id)
                    console.print("[dim]消息已排队，当前回复完成后发送。[/dim]")
            continue

        if user_input in ("/quit", "/exit"):
            chat_runner.close()
            console.file = original_console_file
            console.print("[dim]再见！[/dim]")
            break

        elif user_input == "/help":
            console.print(HELP_TEXT + "\n".join(_skill_help_lines(agent._skills)))

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
            output_pane.set_thinking_expanded(True)

        elif user_input == "/thinking hide":
            agent.config["agent"]["show_thinking"] = False
            output_pane.set_thinking_expanded(False)

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
            session_id = new_id
            history_pager = DialogueHistoryPager(session_manager, session_id)
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
                    session_id = target["id"]
                    history_pager = DialogueHistoryPager(session_manager, session_id)
                    history = history_pager.next_page()
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
                history_pager = DialogueHistoryPager(session_manager, session_id)
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
            skill = agent.toggle_skill(name, True)
            if skill:
                console.print(f"[green]Skill \'{skill['name']}\' 已激活。[/green]")
            else:
                console.print(f"[red]未找到 skill：{name}，请用 /skill list 查看可用列表。[/red]")

        elif user_input.startswith("/skill off "):
            name = user_input[11:].strip()
            skill = agent.toggle_skill(name, False)
            if skill:
                console.print(f"[dim]Skill \'{skill['name']}\' 已停用。[/dim]")
            else:
                console.print(f"[red]未找到 skill：{name}，请用 /skill list 查看可用列表。[/red]")

        elif user_input.startswith("/") and invoked_skill and not invoked_skill["enabled"]:
            # 显式 /<skill 名> 调用本身即启用意图：自动激活并放行本轮任务
            agent.toggle_skill(invoked_skill["name"], True)
            console.print(f"[green]Skill '{invoked_skill['name']}' 已自动激活（原为停用状态）。[/green]")
            _show_dialogue_message(console, "user", user_input, datetime.now())
            chat_runner.submit(user_input, agent.session_id, rename=is_new_session)
            is_new_session = False

        elif user_input.startswith("/") and not invoked_skill:
            console.print("[yellow]未知命令，输入 /help 查看可用命令。[/yellow]")

        else:
            _show_dialogue_message(console, "user", user_input, datetime.now())
            chat_runner.submit(user_input, agent.session_id, rename=is_new_session)
            is_new_session = False


if __name__ == "__main__":
    main()
