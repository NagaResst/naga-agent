import json
import atexit
import re
import subprocess
import sys
import threading
import time
from typing import Optional

import os
from openai import OpenAI

from tools.registry import discover_tools
from tools import execute_command as execute_command_module
import tools.memory as memory_module
import tools.web_search as web_search_module
from agent.token_tracker import TokenTracker
from agent.router import ModelRouter
from agent import summarizer
from agent.skill_registry import discover_skills, _extract_match_terms, _normalize_skill_text
from agent.context_window import get_context_window
from agent.context_window import (
    get_context_window,
)
from memory.manager import MemoryManager


class Agent:
    _RUNTIME_STATE_MARKER = "【固定约束与运行状态】"

    def __init__(self, config: dict, session_manager, session_id: str, model: str, console):
        self.config = config
        self.session_manager = session_manager
        self.session_id = session_id
        self.model = model          # 用户手动选定的基础模型（路由开启时可能被覆盖）
        self._manual_model = model  # 记录用户手动指定，/model 命令更新此值
        self._routing_locked = False  # True 时跳过自动路由，固定使用 _manual_model
        self.console = console
        self.auto_confirm = config["agent"]["auto_confirm"]
        self.max_history = config["agent"]["max_history"]

        self._client = OpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url") or None,
        )
        self.tool_definitions, self.tool_executors = discover_tools()
        # 常驻工具：无论上下文如何，每次都注入（不参与按需过滤）
        self._core_tools = {"web_search", "fetch_url", "execute_command", "memory"}

        # 启动语义引擎子进程（mem0 和手动记忆共用 bge-base-zh-v1.5）
        self._daemon_proc = None
        daemon_cfg = config.get("embedding_daemon", {})
        daemon_script = daemon_cfg.get("script_path", "")
        # 支持相对路径：相对于项目根目录解析
        if daemon_script and not os.path.isabs(daemon_script):
            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            daemon_script = os.path.join(_project_root, daemon_script)
        if daemon_script and os.path.isfile(daemon_script):
            daemon_host = daemon_cfg.get("host", "127.0.0.1")
            daemon_port = daemon_cfg.get("port", 8000)
            try:
                # PYTHONUNBUFFERED=1 确保 Python 子进程输出不缓冲，实时可见
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                self._daemon_proc = subprocess.Popen(
                    [sys.executable, daemon_script, "--host", daemon_host, "--port", str(daemon_port),
                     "--cpu-threads", str(daemon_cfg.get("cpu_threads", 0))],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=env,
                )
                # 后台线程：将语义引擎的 stdout 转发到控制台
                def _stream_daemon_output(proc):
                    try:
                        for line in iter(proc.stdout.readline, b""):
                            text = line.decode(errors="replace").rstrip()
                            if text:
                                console.print(f"[dim][语义引擎] {text}[/dim]")
                    except Exception:
                        pass

                threading.Thread(target=_stream_daemon_output, args=(self._daemon_proc,), daemon=True).start()

                # 等待语义引擎就绪（无硬性超时，首次运行下载模型可能需要数分钟）
                import requests as _req
                daemon_url = f"http://{daemon_host}:{daemon_port}"
                console.print("[dim]语义引擎正在初始化（首次运行可能需要下载模型，请耐心等待）…[/dim]")
                console.print("[dim]按 Ctrl+C 可跳过等待（语义引擎将在后台继续下载）[/dim]")
                wait_count = 0
                try:
                    while True:
                        # 检查进程是否已退出
                        if self._daemon_proc.poll() is not None:
                            console.print(f"[dim yellow]语义引擎进程异常退出（返回码 {self._daemon_proc.returncode}），向量功能不可用[/dim yellow]")
                            self._daemon_proc = None
                            break
                        try:
                            resp = _req.get(f"{daemon_url}/health", timeout=1)
                            # 验证：我们的进程仍在运行（防止命中旧实例）
                            if self._daemon_proc.poll() is not None:
                                console.print("[dim yellow]语义引擎进程已退出（health 命中的是旧实例），向量功能不可用[/dim yellow]")
                                self._daemon_proc = None
                                break
                            console.print(f"[dim]语义引擎已就绪 (PID {self._daemon_proc.pid}, {daemon_url})[/dim]")
                            # 主进程退出时自动关闭语义引擎子进程
                            atexit.register(self._cleanup_daemon)
                            break
                        except Exception:
                            time.sleep(2)
                            wait_count += 1
                            if wait_count % 5 == 0:
                                console.print(f"[dim]仍在等待语义引擎就绪…（已等待 {wait_count * 2}s）[/dim]")
                except KeyboardInterrupt:
                    console.print("\n[dim yellow]已跳过等待，语义引擎将在后台继续初始化。[/dim yellow]")
                    console.print(f"[dim yellow]模型下载完成后，下次启动将立即可用。[/dim yellow]")
                    atexit.register(self._cleanup_daemon)
            except Exception as e:
                console.print(f"[dim yellow]语义引擎启动失败：{e}，向量功能可能不可用[/dim yellow]")
        else:
            if daemon_script:
                console.print(f"[dim yellow]语义引擎脚本不存在：{daemon_script}[/dim yellow]")

        # 初始化分层记忆管理器（必要组件，失败时向上抛出）
        self._memory_manager = MemoryManager(
            config["memory"],
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            storage=session_manager,
        )
        memory_module.set_memory_manager(self._memory_manager)
        # 注入配置到手动记忆模块
        import tools.manual_memory as manual_memory_module
        daemon_cfg = config.get("embedding_daemon", {})
        qdrant_cfg = config.get("memory", {}).get("qdrant", {})
        manual_memory_module.configure(
            qdrant_host=qdrant_cfg.get("host", "localhost"),
            qdrant_port=qdrant_cfg.get("port", 6333),
            daemon_url=f"http://{daemon_cfg.get('host', '127.0.0.1')}:{daemon_cfg.get('port', 8000)}",
        )
        if self._memory_manager.available:
            console.print("[dim]分层记忆已启用（SQLite 精确查找 + mem0 向量语义检索）[/dim]")
        else:
            console.print("[dim]分层记忆已启用（SQLite 精确查找，向量语义检索不可用）[/dim]")

        # 注入 search 配置到 web_search 工具
        web_search_module.set_config(config.get("search", {}))

        # Token 追踪器
        self._token_tracker = TokenTracker()
        if self._token_tracker.mode == "unavailable":
            console.print(
                "[dim yellow]Token 估算模式：unavailable（字符粗估，误差较大）[/dim yellow]"
            )

        # 模型路由器（进程内缓存，无需外部依赖）
        self._router = ModelRouter(
            config.get("routing", {}),
            default_model=config["model"]["default"],
        )

        # 启动时扫描并加载 skills/ 目录下所有 skill
        _project_root = os.path.dirname(os.path.dirname(__file__))
        self._skills_dir = os.path.join(_project_root, "skills")
        self._skills_enabled_names = config.get("skills", {}).get("enabled", [])
        self._skills: list = discover_skills(self._skills_dir, self._skills_enabled_names)
        if self._skills:
            active = [s["name"] for s in self._skills if s["enabled"]]
            console.print(f"[dim]已加载 {len(self._skills)} 个 skill，激活：{active or '无'}[/dim]")

        # 注入 agent 引用到 skill_manager 工具
        import tools.skill_manager as skill_manager_module
        skill_manager_module.set_agent(self)

        # 会话主题追踪（内联标签方式，每轮从主模型回复中提取，零额外 API 调用）
        self._session_theme: str = ""

        # system prompt token 缓存（分两层独立维护，用于补偿历史 token 估算）
        # 主层：base_prompt + 行为规则 + 记忆块 + skill 摘要列表
        # skill 层：命中 skill 的全文（未命中时为 0）
        self._last_main_system_tokens: int = 0
        self._last_skill_system_tokens: int = 0
        self._last_runtime_state_tokens: int = 0
        # skill_layer_mode 降级标记：auto 模式下若 API 不支持多 system 消息则置 True
        self._skill_layer_merged: bool = False
        self._pinned_constraints: dict = {
            "user_goal": "",
            "user_constraints": [],
            "active_skills": [],
            "skill_constraints": [],
            "required_outputs": [],
        }
        self._runtime_task_state: dict = {
            "current_step": "",
            "pending_step": "",
            "plan_steps": [],
            "completed_steps": [],
            "read_skill_files": [],
            "working_set": [],
            "recent_tool_events": [],
            "recent_findings": [],
        }
        self._previous_turn_state: dict = {
            "pinned_constraints": {},
            "runtime_task_state": {},
        }

    def _cleanup_daemon(self):
        """主进程退出时终止语义引擎子进程。"""
        if self._daemon_proc and self._daemon_proc.poll() is None:
            try:
                self._daemon_proc.kill()  # SIGKILL 立即终止，不等待
            except Exception:
                pass

    def reload_skills(self):
        """重新扫描 skills/ 目录，热更新已加载的 skill 列表。"""
        self._skills = discover_skills(self._skills_dir, self._skills_enabled_names)

    @staticmethod
    def _dedupe_trimmed_lines(items: list[str], limit: int) -> list[str]:
        result = []
        seen = set()
        for item in items:
            text = (item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
            if len(result) >= limit:
                break
        return result

    def _extract_user_constraints(self, user_input: str) -> list[str]:
        clauses = []
        must_markers = (
            "必须", "不要", "不能", "禁止", "只", "仅", "务必", "需要", "按", "优先",
            "must", "must not", "do not", "don't", "only", "require",
        )
        normalized = (user_input or "").replace("\r", "\n")
        for segment in re.split(r'[\n。；;！？!?]+', normalized):
            text = segment.strip(" -\t")
            if len(text) < 2:
                continue
            lowered = text.lower()
            if any(marker in text or marker in lowered for marker in must_markers):
                clauses.append(text[:180])
        return self._dedupe_trimmed_lines(clauses, limit=8)

    def _extract_skill_constraints(self, prompt: str) -> list[str]:
        lines = []
        markers = (
            "必须", "禁止", "不要", "不能", "严禁", "务必", "强制", "缺一不可",
            "必须执行", "必须包含", "required", "must", "must not", "do not",
        )
        for raw_line in (prompt or "").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            text = stripped.lstrip("-*0123456789.[]✅⚠️❌")
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) < 2:
                continue
            lowered = text.lower()
            if any(marker in text or marker in lowered for marker in markers):
                lines.append(text[:180])
        return self._dedupe_trimmed_lines(lines, limit=10)

    def _extract_required_outputs(self, prompt: str) -> list[str]:
        outputs = []
        capture = False
        for raw_line in (prompt or "").splitlines():
            text = raw_line.strip()
            normalized = re.sub(r'\s+', ' ', text).strip()
            lowered = normalized.lower()
            if any(keyword in normalized for keyword in ("报告必须包含", "必须包含以下章节", "输出要求", "产出要求")):
                capture = True
                outputs.append(normalized[:180])
                continue
            if capture:
                if not normalized:
                    break
                if normalized.startswith("#"):
                    continue
                candidate = normalized.lstrip("-*0123456789.[]")
                candidate = candidate.strip()
                if len(candidate) >= 2:
                    outputs.append(candidate[:180])
                if len(outputs) >= 10:
                    break
        return self._dedupe_trimmed_lines(outputs, limit=8)

    def _track_skill_file_read(self, path: str):
        if not path:
            return
        abs_path = os.path.abspath(path)
        read_skill_files = self._runtime_task_state.get("read_skill_files", [])
        active_names = set(self._pinned_constraints.get("active_skills", []))
        for skill in self._skills:
            if skill.get("name") not in active_names:
                continue
            source_file = skill.get("source_file")
            skill_dir = skill.get("skill_dir")
            source_abs = os.path.abspath(source_file) if source_file else ""
            skill_dir_abs = os.path.abspath(skill_dir) if skill_dir else ""
            if source_abs and os.path.normpath(abs_path) == os.path.normpath(source_abs):
                label = f"{skill['name']}: SKILL.md"
            elif skill_dir_abs and os.path.normpath(abs_path).startswith(os.path.normpath(skill_dir_abs) + os.sep):
                rel_path = os.path.relpath(abs_path, skill_dir_abs)
                label = f"{skill['name']}: {rel_path}"
            else:
                continue
            read_skill_files.append(label)
            limit = int(self.config.get("agent", {}).get("runtime_read_skill_files_max_items", 8))
            self._runtime_task_state["read_skill_files"] = self._dedupe_trimmed_lines(read_skill_files[::-1], limit=limit)[::-1]
            return

    def _reset_runtime_state(self, user_input: str):
        self._pinned_constraints = {
            "user_goal": (user_input or "").strip()[:500],
            "user_constraints": self._extract_user_constraints(user_input),
            "active_skills": [],
            "skill_constraints": [],
            "required_outputs": [],
        }
        self._runtime_task_state = {
            "current_step": "",
            "pending_step": "",
            "plan_steps": [],
            "completed_steps": [],
            "read_skill_files": [],
            "working_set": [],
            "recent_tool_events": [],
            "recent_findings": [],
        }
        self._last_runtime_state_tokens = 0

    def _snapshot_runtime_state(self):
        self._previous_turn_state = {
            "pinned_constraints": {
                "user_goal": self._pinned_constraints.get("user_goal", ""),
                "user_constraints": list(self._pinned_constraints.get("user_constraints", [])),
                "active_skills": list(self._pinned_constraints.get("active_skills", [])),
                "skill_constraints": list(self._pinned_constraints.get("skill_constraints", [])),
                "required_outputs": list(self._pinned_constraints.get("required_outputs", [])),
            },
            "runtime_task_state": {
                "current_step": self._runtime_task_state.get("current_step", ""),
                "pending_step": self._runtime_task_state.get("pending_step", ""),
                "plan_steps": list(self._runtime_task_state.get("plan_steps", [])),
                "completed_steps": list(self._runtime_task_state.get("completed_steps", [])),
                "read_skill_files": list(self._runtime_task_state.get("read_skill_files", [])),
                "working_set": list(self._runtime_task_state.get("working_set", [])),
                "recent_tool_events": list(self._runtime_task_state.get("recent_tool_events", [])),
                "recent_findings": list(self._runtime_task_state.get("recent_findings", [])),
            },
        }

    def _restore_followup_state(self, user_input: str):
        if not self._is_followup_request(user_input):
            return

        previous_pinned = self._previous_turn_state.get("pinned_constraints", {})
        previous_runtime = self._previous_turn_state.get("runtime_task_state", {})
        if not previous_pinned and not previous_runtime:
            return

        if previous_pinned.get("user_goal"):
            self._pinned_constraints["user_goal"] = previous_pinned["user_goal"]

        merged_user_constraints = list(previous_pinned.get("user_constraints", [])) + list(self._pinned_constraints.get("user_constraints", []))
        self._pinned_constraints["user_constraints"] = self._dedupe_trimmed_lines(merged_user_constraints, limit=8)

        for key in ("active_skills", "skill_constraints", "required_outputs"):
            previous_value = previous_pinned.get(key, [])
            if previous_value:
                self._pinned_constraints[key] = list(previous_value)

        for key in ("current_step",):
            if previous_runtime.get(key):
                self._runtime_task_state[key] = previous_runtime[key]

        if previous_runtime.get("pending_step"):
            self._runtime_task_state["pending_step"] = previous_runtime["pending_step"]
            if not self._runtime_task_state.get("current_step"):
                self._runtime_task_state["current_step"] = previous_runtime["pending_step"]

        for key in ("plan_steps", "completed_steps", "read_skill_files", "working_set", "recent_tool_events", "recent_findings"):
            previous_value = previous_runtime.get(key, [])
            if previous_value:
                self._runtime_task_state[key] = list(previous_value)

    @staticmethod
    def _is_followup_request(text: str) -> bool:
        normalized = re.sub(r'\s+', '', (text or '').lower())
        if not normalized:
            return False
        followup_markers = (
            "继续", "接着", "然后", "往下", "下一步", "继续做", "继续执行", "继续完成",
            "goon", "continue", "next", "resume",
        )
        return len(normalized) <= 12 or any(marker in normalized for marker in followup_markers)

    @staticmethod
    def _select_recall_value(current_value, previous_value, prefer_previous: bool = False):
        if prefer_previous and previous_value:
            return previous_value
        if current_value:
            return current_value
        return previous_value

    def _extract_dialogue_records(self, messages: list, limit: int = 6) -> list[str]:
        records = []
        pending_user = ""
        for message in messages:
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            content = self._strip_markdown(message.get("content") or "")
            content = re.sub(r'<theme>.*?</theme>', '', content, flags=re.DOTALL).strip()
            content = re.sub(r'\s+', ' ', content).strip()
            if not content:
                continue
            if role == "user":
                pending_user = content[:220]
                records.append(f"对话问题: {pending_user}")
            else:
                records.append(f"对话结论: {content[:220]}")
                pending_user = ""
        return self._dedupe_trimmed_lines(records[::-1], limit=limit)[::-1]

    def _collect_pre_compress_records(self, messages: list, context_window: int, threshold: float, tool_max_chars: int, keep_hi: int) -> list[str]:
        turns, _ = summarizer._split_turns(messages)
        if not turns:
            return []
        preview = summarizer.compress_pipeline(
            messages,
            self._token_tracker,
            context_window,
            threshold,
            tool_max_chars,
            keep_hi=keep_hi,
        )
        preview_turns, _ = summarizer._split_turns(preview)
        dropped_turn_count = max(0, len(turns) - len(preview_turns))
        if dropped_turn_count <= 0:
            return []
        dropped_messages = []
        for turn in turns[:dropped_turn_count]:
            dropped_messages.extend(turn)
        return self._extract_dialogue_records(dropped_messages, limit=8)

    def _sync_active_skills(self, hit_skills: list):
        self._pinned_constraints["active_skills"] = [skill["name"] for skill in hit_skills]
        skill_constraints = []
        required_outputs = []
        for skill in hit_skills:
            skill_constraints.extend(self._extract_skill_constraints(skill.get("prompt", "")))
            required_outputs.extend(self._extract_required_outputs(skill.get("prompt", "")))
        constraint_limit = int(self.config.get("agent", {}).get("runtime_constraints_max_items", 10))
        output_limit = int(self.config.get("agent", {}).get("runtime_required_outputs_max_items", 8))
        self._pinned_constraints["skill_constraints"] = self._dedupe_trimmed_lines(skill_constraints, limit=constraint_limit)
        self._pinned_constraints["required_outputs"] = self._dedupe_trimmed_lines(required_outputs, limit=output_limit)

    def _set_plan_steps(self, steps: list[str]):
        self._runtime_task_state["plan_steps"] = [step.strip()[:160] for step in steps if step.strip()][:8]

    def _set_current_step(self, step_desc: str):
        self._runtime_task_state["current_step"] = (step_desc or "").strip()[:200]

    def _set_pending_step(self, step_desc: str):
        self._runtime_task_state["pending_step"] = (step_desc or "").strip()[:200]

    @staticmethod
    def _extract_pending_step(text: str) -> str:
        match = re.search(r'待执行步骤[:：]\s*(.+)', text or "")
        if match:
            return match.group(1).strip()[:200]
        return ""

    def _add_working_set_item(self, item: str):
        text = re.sub(r'\s+', ' ', (item or "")).strip()
        if len(text) < 2:
            return
        limit = int(self.config.get("agent", {}).get("runtime_working_set_max_items", 8))
        working_set = self._runtime_task_state.get("working_set", [])
        working_set.append(text[:220])
        self._runtime_task_state["working_set"] = self._dedupe_trimmed_lines(working_set[::-1], limit=limit)[::-1]

    def _summarize_step_result(self, step_desc: str, step_output: str) -> str:
        content = self._strip_markdown(step_output or "")
        content = re.sub(r'\[STEP_DONE\]', '', content)
        content = re.sub(r'\[STEP_CONTEXT\].*?\[/STEP_CONTEXT\]', '', content, flags=re.DOTALL)
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return f"步骤完成：{step_desc}"

        snippets = []
        for line in lines:
            if line.startswith("<theme>"):
                continue
            snippets.append(line)
            if len("；".join(snippets)) >= 120 or len(snippets) >= 2:
                break
        snippet = "；".join(snippets).strip("； ")[:180]
        return f"{step_desc}：{snippet}" if snippet else f"步骤完成：{step_desc}"

    def _record_step_completion(self, step_index: int, step_desc: str, step_output: str):
        summary = self._summarize_step_result(step_desc, step_output)
        completed = self._runtime_task_state.get("completed_steps", [])
        completed.append(f"Step {step_index}: {step_desc}"[:200])
        completed_limit = int(self.config.get("agent", {}).get("runtime_completed_steps_max_items", 8))
        self._runtime_task_state["completed_steps"] = self._dedupe_trimmed_lines(completed[::-1], limit=completed_limit)[::-1]
        self._add_working_set_item(summary)

    def _build_episodic_records(
        self,
        reason: str = "",
        recent_messages: Optional[list] = None,
        step_desc: str = "",
        step_output: str = "",
    ) -> list[str]:
        records = []
        current_step = self._runtime_task_state.get("current_step", "")
        if current_step:
            records.append(f"当前步骤: {current_step[:220]}")

        if step_desc and step_output:
            records.append(f"步骤结论: {self._summarize_step_result(step_desc, step_output)[:220]}")

        for item in self._runtime_task_state.get("working_set", [])[-4:]:
            records.append(f"工作结论: {item[:220]}")

        for item in self._runtime_task_state.get("completed_steps", [])[-3:]:
            records.append(f"已完成步骤: {item[:220]}")

        if recent_messages:
            records.extend(self._extract_dialogue_records(recent_messages, limit=6))

        return self._dedupe_trimmed_lines(records, limit=12)

    def _store_episodic_records(
        self,
        reason: str,
        recent_messages: Optional[list] = None,
        step_desc: str = "",
        step_output: str = "",
        async_write: bool = False,
    ):
        if not (self._memory_manager and self._memory_manager.available):
            return
        records = self._build_episodic_records(
            reason=reason,
            recent_messages=recent_messages,
            step_desc=step_desc,
            step_output=step_output,
        )
        if not records:
            return
        if async_write:
            self._memory_manager.extract_async_records(records, source=reason)
        else:
            self._memory_manager.add_episodic_records(records, source=reason)

    def _build_episodic_query(self, last_user_msg: str) -> str:
        parts = []
        previous_pinned = self._previous_turn_state.get("pinned_constraints", {})
        previous_runtime = self._previous_turn_state.get("runtime_task_state", {})
        prefer_previous = self._is_followup_request(last_user_msg)

        goal = self._select_recall_value(
            self._pinned_constraints.get("user_goal", ""),
            previous_pinned.get("user_goal", ""),
            prefer_previous=prefer_previous,
        )
        if goal:
            parts.append(f"任务目标：{goal[:160]}")

        user_constraints = self._select_recall_value(
            self._pinned_constraints.get("user_constraints", []),
            previous_pinned.get("user_constraints", []),
            prefer_previous=prefer_previous,
        )
        if user_constraints:
            parts.append("用户约束：" + "；".join(user_constraints[:3]))

        active_skills = self._select_recall_value(
            self._pinned_constraints.get("active_skills", []),
            previous_pinned.get("active_skills", []),
        )
        if active_skills:
            parts.append("激活技能：" + ", ".join(active_skills[:4]))

        current_step = self._select_recall_value(
            self._runtime_task_state.get("current_step", ""),
            previous_runtime.get("current_step", ""),
            prefer_previous=prefer_previous,
        )
        if not current_step:
            current_step = self._select_recall_value(
                self._runtime_task_state.get("pending_step", ""),
                previous_runtime.get("pending_step", ""),
                prefer_previous=prefer_previous,
            )
        if current_step:
            parts.append(f"当前步骤：{current_step[:160]}")

        working_set = self._select_recall_value(
            self._runtime_task_state.get("working_set", []),
            previous_runtime.get("working_set", []),
            prefer_previous=prefer_previous,
        )
        if working_set:
            parts.append("最近工作结论：" + "；".join(working_set[-3:]))

        if last_user_msg:
            parts.append(f"最近用户问题：{last_user_msg[:220]}")

        return "\n".join(part for part in parts if part).strip()[:1200]

    def _summarize_tool_call(self, tool_name: str, tool_args: dict) -> str:
        if tool_name == "read_file":
            path = tool_args.get("path") or tool_args.get("file_path") or ""
            return f"读取文件：{path}"[:180]
        if tool_name == "execute_command":
            desc = tool_args.get("description", "").strip()
            cmd = tool_args.get("command", "").strip()
            return (f"执行命令：{desc or cmd}"[:180]).strip()
        if tool_name == "fetch_url":
            return f"抓取网页：{tool_args.get('url', '')}"[:180]
        if tool_name == "web_search":
            return f"联网搜索：{tool_args.get('query', '')}"[:180]
        return f"调用工具：{tool_name}"[:180]

    def _summarize_tool_result(self, tool_name: str, tool_result: str) -> str:
        lines = [line.strip() for line in (tool_result or "").splitlines() if line.strip()]
        if not lines:
            return f"{tool_name}：返回空结果"

        if tool_name == "read_file":
            content_lines = []
            in_content = False
            in_frontmatter = False
            for line in lines:
                if not in_content:
                    if line == "---":
                        in_content = True
                    continue

                if line == "---":
                    in_frontmatter = not in_frontmatter
                    continue
                if in_frontmatter:
                    continue
                if line.startswith(("路径：", "类型：", "行范围：", "剩余未读取：")):
                    continue
                cleaned = re.sub(r'[`*_#>]+', '', line).strip()
                if not cleaned:
                    continue
                content_lines.append(cleaned)
                if len(content_lines) >= 2:
                    break

            if content_lines:
                return f"{tool_name}：{'；'.join(content_lines)[:180]}"

        first_line = re.sub(r'[`*_#>]+', '', lines[0]).strip()
        return f"{tool_name}：{first_line[:180]}"

    def _summarize_reply_conclusion(self, reply_text: str) -> str:
        content = self._strip_markdown(reply_text or "")
        content = re.sub(r'<theme>.*?</theme>', '', content, flags=re.DOTALL).strip()
        lines = [re.sub(r'\s+', ' ', line).strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return ""

        for line in lines:
            if "待执行步骤" in line or "下一步" in line:
                return line[:180]

        return "；".join(lines[:2])[:180]

    def _record_tool_event(self, tool_name: str, tool_args: dict, tool_result: str):
        if tool_name == "read_file":
            self._track_skill_file_read(tool_args.get("path") or tool_args.get("file_path") or "")

        event = self._summarize_tool_call(tool_name, tool_args)
        tool_limit = int(self.config.get("agent", {}).get("runtime_tool_events_max_items", 6))
        self._runtime_task_state["recent_tool_events"] = (
            self._runtime_task_state.get("recent_tool_events", []) + [event]
        )[-tool_limit:]

        if tool_result and not tool_result.startswith("错误："):
            findings = self._runtime_task_state.get("recent_findings", [])
            finding_summary = self._summarize_tool_result(tool_name, tool_result)
            findings.append(finding_summary)
            finding_limit = int(self.config.get("agent", {}).get("runtime_findings_max_items", 6))
            self._runtime_task_state["recent_findings"] = self._dedupe_trimmed_lines(findings[::-1], limit=finding_limit)[::-1]
            self._add_working_set_item(finding_summary)

    def _build_runtime_state_block(self) -> str:
        cfg = self.config.get("agent", {})
        goal = self._pinned_constraints.get("user_goal", "")
        user_constraints = self._pinned_constraints.get("user_constraints", [])
        active_skills = self._pinned_constraints.get("active_skills", [])
        skill_constraints = self._pinned_constraints.get("skill_constraints", [])
        required_outputs = self._pinned_constraints.get("required_outputs", [])
        current_step = self._runtime_task_state.get("current_step", "")
        pending_step = self._runtime_task_state.get("pending_step", "")
        plan_steps = self._runtime_task_state.get("plan_steps", [])
        completed_steps = self._runtime_task_state.get("completed_steps", [])
        read_skill_files = self._runtime_task_state.get("read_skill_files", [])
        working_set = self._runtime_task_state.get("working_set", [])
        recent_tool_events = self._runtime_task_state.get("recent_tool_events", [])
        recent_findings = self._runtime_task_state.get("recent_findings", [])

        sections = [self._RUNTIME_STATE_MARKER]
        if goal:
            sections.extend(["【当前目标】", goal])
        if user_constraints:
            sections.append("【用户明确约束】")
            sections.extend(f"- {item}" for item in user_constraints)
        if skill_constraints:
            sections.append("【Skill 关键约束】")
            sections.extend(f"- {item}" for item in skill_constraints)
        if active_skills:
            sections.append("【当前激活 Skill】")
            sections.extend(f"- {item}" for item in active_skills)
        if current_step:
            sections.extend(["【当前步骤】", current_step])
        if pending_step and pending_step != current_step:
            sections.extend(["【待执行步骤】", pending_step])
        if required_outputs:
            sections.append("【必须产出】")
            sections.extend(f"- {item}" for item in required_outputs)
        if working_set:
            sections.append("【当前工作集】")
            sections.extend(f"- {item}" for item in working_set)
        if completed_steps:
            sections.append("【已完成步骤】")
            sections.extend(f"- {item}" for item in completed_steps)
        if plan_steps:
            sections.append("【执行计划】")
            sections.extend(f"- {item}" for item in plan_steps)
        if read_skill_files:
            sections.append("【已读取 Skill 资源】")
            sections.extend(f"- {item}" for item in read_skill_files)
        if recent_tool_events:
            sections.append("【最近工具动作】")
            sections.extend(f"- {item}" for item in recent_tool_events)
        if recent_findings:
            sections.append("【最近已验证结果】")
            sections.extend(f"- {item}" for item in recent_findings)
        block = "\n".join(sections)
        max_chars = int(cfg.get("runtime_state_max_chars", 1800))
        if len(block) <= max_chars:
            return block

        # 过长时优先缩短低优先级部分，避免裁掉硬约束和必需产出。
        trimmed_sections = sections[:]
        for header in ("【最近已验证结果】", "【最近工具动作】", "【已读取 Skill 资源】", "【执行计划】", "【已完成步骤】"):
            while len("\n".join(trimmed_sections)) > max_chars and header in trimmed_sections:
                index = trimmed_sections.index(header)
                if index + 1 >= len(trimmed_sections) or trimmed_sections[index + 1].startswith("【"):
                    trimmed_sections.pop(index)
                    break
                trimmed_sections.pop(index + 1)
        block = "\n".join(trimmed_sections)
        if len(block) > max_chars:
            block = block[:max_chars].rstrip() + "\n[运行状态已截断]"
        return block

    def _upsert_runtime_state_message(self, messages: list):
        state_block = self._build_runtime_state_block()
        self._last_runtime_state_tokens = self._token_tracker.count_tokens(state_block) if hasattr(self, "_token_tracker") else 0
        for message in messages:
            if message.get("role") == "system" and self._RUNTIME_STATE_MARKER in (message.get("content") or ""):
                message["content"] = state_block
                return
        messages.insert(0, {"role": "system", "content": state_block})

    def _build_skill_prompt_block(self, skill: dict) -> str:
        directive = (
            "用户已显式指定该 Skill，必须优先遵守其主指令。"
            if skill.get("_forced")
            else "当前请求已命中该 Skill，执行时优先遵守其主指令。"
        )

        resource_files = skill.get("resource_files", [])
        resource_lines = []
        if resource_files:
            preview = resource_files[:8]
            resource_lines = ["【Skill 资源提示】"]
            resource_lines.extend(f"- {path}" for path in preview)
            if len(resource_files) > len(preview):
                resource_lines.append(f"- 其余 {len(resource_files) - len(preview)} 个文件未展开")

        protocol_lines = [
            "【Skill 使用协议】",
            "1. 将该 Skill 视为当前任务的执行合同，先识别其中的硬限制、禁止项、流程和输出要求，再决定下一步。",
            "2. 不要假设入口文件就是全部 Skill；如果给出了 Skill 包路径或资源提示，必要时继续读取同目录下相关 Markdown、说明文档和脚本。",
            "3. 优先使用 read_file 阅读 Skill 相关文件；只有在需要确认目录结构或定位文件时，才使用 execute_command 辅助查看。",
            "4. 长任务中以系统提供的固定约束与运行状态为主，不要自行改写、弱化或忽略这些约束。",
            "5. 当你读取新的 Skill 文件或获得关键工具结果时，应基于当前固定约束与运行状态继续推进，而不是反复依赖长历史消息。",
        ]

        block_parts = [
            f"【激活 Skill：{skill['name']}】",
            directive,
            "【Skill 入口文件】",
            skill.get("source_file", "未知"),
        ]

        if skill.get("skill_dir"):
            block_parts.extend(["【Skill 包路径】", skill["skill_dir"]])
        if resource_lines:
            block_parts.append("\n".join(resource_lines))

        block_parts.append("\n".join(protocol_lines))
        block_parts.extend([
            "【Skill 主指令】",
            skill["prompt"],
        ])
        return "\n".join(block_parts)

    def _extract_forced_skill_names(self, user_text: str) -> set[str]:
        import re

        forced_names = set()
        if not user_text:
            return forced_names

        raw_signal = user_text.lower()
        for match in re.findall(r'@skill\(([^)]+)\)', raw_signal):
            candidate = _normalize_skill_text(match)
            for skill in self._skills:
                if _normalize_skill_text(skill.get("name", "")) == candidate:
                    forced_names.add(skill["name"])
        return forced_names

    def _score_skill_match(self, skill: dict, signal: str, signal_terms: set[str]) -> tuple[float, list[str]]:
        score = 0.0
        reasons = []
        matched_phrases = set()

        skill_name = _normalize_skill_text(skill.get("name", ""))
        if len(skill_name) >= 2 and skill_name in signal:
            score += 8.0
            reasons.append(f"name:{skill.get('name', '')}")
            matched_phrases.add(skill_name)

        for keyword in skill.get("keywords", []):
            normalized = _normalize_skill_text(keyword)
            if len(normalized) < 2 or normalized in matched_phrases or normalized not in signal:
                continue
            score += 5.0 if len(normalized) >= 4 else 3.5
            matched_phrases.add(normalized)
            reasons.append(f"keyword:{keyword}")
            if len(reasons) >= 3:
                break

        for example in skill.get("examples", []):
            normalized = _normalize_skill_text(example)
            if len(normalized) < 2 or normalized in matched_phrases or normalized not in signal:
                continue
            score += 4.5
            matched_phrases.add(normalized)
            reasons.append(f"example:{example[:24]}")
            if len(reasons) >= 3:
                break

        phrase_hits = []
        for phrase in skill.get("match_phrases", []):
            if len(phrase) < 4 or phrase in matched_phrases or phrase not in signal:
                continue
            phrase_hits.append(phrase)
            matched_phrases.add(phrase)
            score += 2.5
            if len(phrase_hits) >= 2:
                break
        if phrase_hits and len(reasons) < 3:
            reasons.append("phrase:" + ",".join(phrase_hits[:2]))

        overlap = [term for term in skill.get("match_terms", []) if term in signal_terms]
        if overlap:
            ranked_terms = sorted(overlap, key=lambda term: (len(term), term), reverse=True)
            picked_terms = []
            overlap_score = 0.0
            for term in ranked_terms:
                picked_terms.append(term)
                overlap_score += 1.2 if len(term) >= 3 else 0.9
                if len(picked_terms) >= 4:
                    break
            score += min(overlap_score, 4.2)
            if len(reasons) < 3:
                reasons.append("term:" + ",".join(picked_terms[:3]))

        return score, reasons

    def _select_tools(self, last_user_msg: str) -> list:
        """按需工具注入：

        - 常驻工具 + 关键词命中的工具：发完整 schema
        - 其余工具：只发最小存根（name only），让模型知道工具存在但不占 token
        发给 API 的 tools 列表去掉自定义 tags 字段。
        """
        import re as _re
        signal = (self._session_theme + " " + last_user_msg[:200]).lower()

        full, stub_names = [], []
        for td in self.tool_definitions:
            name = td.get("function", {}).get("name", "")
            if name in self._core_tools or any(tag.lower() in signal for tag in td.get("tags", [])):
                # 完整 schema，去掉 tags
                full.append({k: v for k, v in td.items() if k != "tags"})
            else:
                stub_names.append(name)

        # 最小存根：只有 name，parameters 为空对象
        stubs = [
            {"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": {}}}}
            for n in stub_names
        ]
        return full + stubs or None

    def _match_skills(self, last_user_msg: str) -> tuple[list, list]:
        """根据会话主题 + 最后用户消息，将激活的 skill 分为命中（全文注入）和未命中（摘要）。

        匹配逻辑：综合 skill 名称、description、keywords、examples 做加权评分，
        按分数排序后仅注入 Top-K 命中 skill 的全文，其余只保留摘要。
        返回 (hit_skills, miss_skills)，均为 skill dict 列表。
        """
        signal = _normalize_skill_text(self._session_theme + " " + last_user_msg[:400])
        signal_terms = set(_extract_match_terms(signal))
        forced_names = self._extract_forced_skill_names(last_user_msg)
        ranked = []
        for skill in self._skills:
            if not skill["enabled"] and skill["name"] not in forced_names:
                continue
            score, reasons = self._score_skill_match(skill, signal, signal_terms)
            if skill["name"] in forced_names:
                score = max(score, 100.0)
                reasons = ["forced:@skill"] + reasons
            ranked.append({
                **skill,
                "_match_score": round(score, 2),
                "_match_reasons": reasons,
                "_forced": skill["name"] in forced_names,
            })

        ranked.sort(key=lambda item: (-item["_match_score"], item["name"]))

        skills_cfg = self.config.get("skills", {})
        min_match_score = float(skills_cfg.get("min_match_score", 2.2))
        match_top_k = max(1, int(skills_cfg.get("match_top_k", 2)))

        hit = [skill for skill in ranked if skill["_match_score"] >= min_match_score][:match_top_k]
        hit_names = {skill["name"] for skill in hit}
        miss = [skill for skill in ranked if skill["name"] not in hit_names]
        return hit, miss

    def _get_messages(self) -> list:
        import re as _re
        history = self.session_manager.get_history(self.session_id)

        # Bug1 fix：保留完整消息结构（含 tool_calls）用于正确 token 估算
        raw_messages = list(history) if history else []

        # 动态获取当前模型的上下文窗口大小
        context_window = get_context_window(
            self.model,
            self.config["agent"].get("context_token_limit", 0),
            client=self._client,
            base_url=self.config.get("base_url"),
        )
        threshold = self.config["agent"].get("compress_threshold", 0.95)
        tool_max_chars = self.config["agent"].get("tool_output_max_chars", 300)
        lo_threshold = self.config["agent"].get("history_lo_threshold", 0.30)
        keep_lo = self.config["agent"].get("history_keep_turns_lo", 10)
        keep_mid = self.config["agent"].get("history_keep_turns_mid", 6)
        keep_hi = self.config["agent"].get("history_keep_turns_hi", 3)
        last_user_msg = next((m.get("content", "") for m in reversed(raw_messages) if m.get("role") == "user"), "")

        active_tools = self._select_tools(last_user_msg) or []

        # Bug2 fix：估算时加上上一轮 system prompt 的缓存 token 数
        history_tokens = self._token_tracker.estimate(raw_messages)
        tool_schema_tokens = self._token_tracker.estimate_tools(active_tools)
        estimated_tokens = (
            history_tokens
            + tool_schema_tokens
            + self._last_main_system_tokens
            + self._last_skill_system_tokens
            + self._last_runtime_state_tokens
        )
        usage_ratio = estimated_tokens / context_window if context_window > 0 else 0.0

        # 三档滑动窗口（每次发消息前都执行，不落盘）
        keep_turns = summarizer.pick_keep_turns(
            usage_ratio,
            lo_threshold=lo_threshold,
            compress_threshold=threshold,
            keep_lo=keep_lo,
            keep_mid=keep_mid,
            keep_hi=keep_hi,
        )
        raw_messages = summarizer.sliding_window_trim(raw_messages, keep_turns=keep_turns)

        if summarizer.should_compress(estimated_tokens, context_window, threshold):
            # 触发前先同步 mem0 提取，确保即将被压缩的内容已落入记忆
            if self._memory_manager and self._memory_manager.available:
                dropped_records = self._collect_pre_compress_records(
                    raw_messages,
                    context_window,
                    threshold,
                    tool_max_chars,
                    keep_hi,
                )
                if dropped_records:
                    self._memory_manager.add_episodic_records(dropped_records, source="pre_compress")
            # 纯机械压缩，只在内存中生效，不落盘
            raw_messages = summarizer.compress_pipeline(
                raw_messages,
                self._token_tracker,
                context_window,
                threshold,
                tool_max_chars,
                keep_hi=keep_hi,
            )

        # ── 构建主层 system prompt ──────────────────────────────────────────
        base_prompt = self.config["agent"].get("system_prompt", "").strip()

        # 通用行为规则：信息不足时优先提问
        _ask_rule = (
            "\n\n【行为规则】\n"
            "当且仅当缺少只有用户才知的关键信息且无法合理推断时，才简短向用户提问。"
            "如果信息已足够执行至少一个有效动作，应先行动再根据结果调整，不要反复追问。"
            "可以通过搜索或文件读取获取的信息不算关键信息不足。"
        )
        base_prompt += _ask_rule

        # 会话主题（内联标签协议）
        _theme_instr = (
            "\n\n【主题标签协议】\n"
            "每次回复的**最开头**先输出 `<theme>主题关键词(10字以内)</theme>` 然后换行，"
            "再输出正文。不要解释标签。示例：`<theme>K8s集群运维</theme>`"
        )
        base_prompt += _theme_instr
        if self._session_theme:
            base_prompt += f"\n\n【当前会话主题】\n{self._session_theme}"

        # Layer1 核心记忆
        if self._memory_manager and self._memory_manager.available:
            core_items = self._memory_manager.search_core()
            memory_block = ("\n\n【用户记忆】\n" + "\n".join(f"  {m}" for m in core_items)) if core_items else ""
        else:
            memories = self.session_manager.list_memories()
            memory_block = ("\n\n【用户记忆】\n" + "\n".join(f"  {k}: {v}" for k, v in memories.items())) if memories else ""

        # ── Skill 处理：命中判断 + 摘要常驻 ──────────────────────────────────
        hit_skills, miss_skills = self._match_skills(last_user_msg)
        self._sync_active_skills(hit_skills)

        # Layer2 情节记忆（语义检索，组合任务目标、步骤态和最后一条用户消息）
        episodic_block = ""
        if self._memory_manager and self._memory_manager.available and last_user_msg:
            top_k = self.config.get("memory", {}).get("episodic_top_k", 3)
            episodic_query = self._build_episodic_query(last_user_msg)
            episodic_items = self._memory_manager.search_episodic(episodic_query, top_k=top_k)
            if episodic_items:
                episodic_block = "\n\n【相关记忆】\n" + "\n".join(f"  {m}" for m in episodic_items)

        # 双记忆源描述
        dual_memory_desc = (
            "\n\n【记忆系统说明】\n"
            "你有两种记忆来源：\n"
            "1. 对话记忆（自动）：从对话中自动提取的用户偏好和情节记忆。\n"
            "2. 手动知识库（手动）：用户主动导入的文档知识，存储在 memories 集合中，支持按标题或来源删除。\n"
            "使用 memory 工具的 search 操作可同时检索两个来源；add_document 用于导入文档；forget 用于删除记忆。"
        )

        skills_cfg = self.config.get("skills", {})
        summary_top_k = max(1, int(skills_cfg.get("summary_top_k", 6)))
        all_active = (hit_skills + miss_skills)[:summary_top_k]

        skill_summary_block = ""
        if all_active:
            lines = [
                "以下仅列出当前最相关的 Skill；标记为“当前相关”的规则优先，其余仅作为可选能力参考。"
            ]
            for s in all_active:
                desc = s.get("description", "") or s["name"]
                if s.get("_forced"):
                    status = "用户指定"
                else:
                    status = "当前相关" if s in hit_skills else "可选"
                lines.append(f"- {s['name']}（{status}）：{desc}")
            skill_summary_block = "\n\n【可用 Skill】\n" + "\n".join(lines)

        # 主层 system prompt（不含命中 skill 全文）
        main_system = base_prompt + memory_block + episodic_block + dual_memory_desc + skill_summary_block

        # 缓存主层 token 数（供下一轮估算补偿）
        self._last_main_system_tokens = self._token_tracker.count_tokens(main_system)

        # ── Skill 层：命中 skill 全文 ─────────────────────────────────────────
        skill_full_content = ""
        if hit_skills:
            parts = []
            for s in hit_skills:
                parts.append(self._build_skill_prompt_block(s))
            skill_full_content = "\n\n".join(parts)

        # 缓存 skill 层 token 数
        self._last_skill_system_tokens = self._token_tracker.count_tokens(skill_full_content) if skill_full_content else 0

        # ── 拼装最终消息列表 ────────────────────────────────────────────────
        skill_layer_mode = self.config["agent"].get("skill_layer_mode", "auto")
        use_multi_system = (
            skill_full_content
            and not self._skill_layer_merged
            and skill_layer_mode in ("auto", "multi_system")
        ) or (
            skill_full_content
            and skill_layer_mode == "multi_system"
        )
        use_merge = skill_layer_mode == "merge" or (skill_full_content and self._skill_layer_merged)

        if main_system.strip():
            if use_merge and skill_full_content:
                # 降级模式：拼接到主层末尾
                merged = main_system + "\n\n" + skill_full_content
                raw_messages.insert(0, {"role": "system", "content": merged.strip()})
            elif use_multi_system:
                # 双 system 消息模式
                raw_messages.insert(0, {"role": "system", "content": skill_full_content.strip()})
                raw_messages.insert(0, {"role": "system", "content": main_system.strip()})
            else:
                # 无命中 skill / merge 已禁用
                raw_messages.insert(0, {"role": "system", "content": main_system.strip()})

        return raw_messages

    def _call_tool(self, name: str, args: dict) -> str:
        executor = self.tool_executors.get(name)
        if executor is None:
            return f"错误：未找到工具 '{name}'"
        tools_cfg = self.config.get("tools", {})
        if name == "execute_command":
            return execute_command_module.execute(
                args,
                auto_confirm=self.auto_confirm,
                console=self.console,
                timeout=tools_cfg.get("command_timeout", 30),
                output_max_chars=tools_cfg.get("output_max_chars", 3000),
            )
        return executor(args)

    def _humanize_tool_hint(self, tool_name: str, tool_args: dict) -> str:
        if tool_name == "execute_command":
            cmd = tool_args.get("command", "")
            desc = tool_args.get("description", "")
            if desc:
                return f"⏳ {desc}…"
            if cmd.startswith("curl") or "http" in cmd:
                return "⏳ 正在联网获取信息…"
            if cmd.startswith("ls") or cmd.startswith("find") or cmd.startswith("cat"):
                return "⏳ 正在查阅本地文件…"
            if cmd.startswith("pip") or cmd.startswith("brew") or cmd.startswith("apt"):
                return "⏳ 正在安装依赖…"
            return "⏳ 正在处理，请稍候…"
        if tool_name == "generate_script":
            filename = tool_args.get("filename", "文件")
            return f"⏳ 正在生成 {filename}…"
        if tool_name == "read_file":
            return f"⏳ 正在读取文件…"
        if tool_name == "edit_file":
            return f"⏳ 正在编辑文件…"
        if tool_name == "memory":
            return f"⏳ 正在操作记忆…"
        return "⏳ 正在处理，请稍候…"

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """脱除常见 Markdown 语法，返回纯文本。"""
        import re
        # 代码块（```...``` 或 ~~~...~~~）替换为其内容
        text = re.sub(r'```[\w]*\n?', '', text)
        text = re.sub(r'~~~[\w]*\n?', '', text)
        # 粗体 / 斜体
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'__(.+?)__', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'\*(.+?)\*', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'_(.+?)_', r'\1', text, flags=re.DOTALL)
        # 标题（行首 # 开头）
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # 表格分隔行（|---|---| 类）
        text = re.sub(r'^\|[-|: ]+\|\s*$', '', text, flags=re.MULTILINE)
        # 表格行：去掉首尾 |，单元格用空格连接
        text = re.sub(r'^\|(.+)\|\s*$',
                      lambda m: '  '.join(c.strip() for c in m.group(1).split('|')),
                      text, flags=re.MULTILINE)
        # 引用块
        text = re.sub(r'^>+\s?', '', text, flags=re.MULTILINE)
        # 行内代码
        text = re.sub(r'`(.+?)`', r'\1', text)
        # 多余空行收拢
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    # OpenAI 顶层参数（不能放进 extra_body，需直接注入 kwargs）
    _OPENAI_TOP_LEVEL_PARAMS = frozenset({"reasoning_effort", "max_completion_tokens"})

    def _build_extra_body(self, model: str):
        """返回 (extra_body, top_level_params) 两个 dict（均可能为 None）。

        - extra_body: Qwen enable_thinking 等供应商扩展参数
        - top_level_params: OpenAI reasoning_effort 等顶层参数
        """
        default = dict(self.config["model"].get("extra_params_default", {}))
        model_specific = dict(self.config["model"].get("extra_params", {}).get(model, {}))
        merged = {**default, **model_specific}
        top_level = {k: v for k, v in merged.items() if k in self._OPENAI_TOP_LEVEL_PARAMS}
        extra_body = {k: v for k, v in merged.items() if k not in self._OPENAI_TOP_LEVEL_PARAMS}
        return (extra_body or None), (top_level or None)

    _PLAN_SYSTEM = (
        "你是一个任务规划器。根据用户的任务，输出一个严格按照以下格式的执行计划，"
        "不要有任何额外解释、不要执行任何操作、不要输出代码：\n\n"
        "[ ] Step 1: <具体步骤描述>\n"
        "[ ] Step 2: <具体步骤描述>\n"
        "...\n\n"
        "要求：\n"
        "- 每个步骤必须是可独立执行的最小工作单元\n"
        "- 步骤数量 3~8 个，不要过于拆碎也不要合并复杂操作\n"
        "- 只输出步骤列表，不要有前言和总结"
    )

    def _plan_node(self, user_input: str, messages: list, model: str):
        """规划节点：生成 checkbox 步骤列表并展示，返回 (steps, plan_text)。"""
        import re as _re
        from rich.panel import Panel as RichPanel

        plan_messages = [
            {"role": "system", "content": self._PLAN_SYSTEM},
            {"role": "user", "content": user_input},
        ]
        show_thinking = self.config["agent"].get("show_thinking", True)
        plan_text = ""
        try:
            extra_body, top_level_params = self._build_extra_body(model)
            kwargs = dict(
                model=model,
                messages=plan_messages,
                temperature=0.3,
                stream=True,
            )
            if extra_body:
                kwargs["extra_body"] = extra_body
            if top_level_params:
                kwargs.update(top_level_params)

            stream = self._client.chat.completions.create(**kwargs)
            in_reasoning = False
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    if not in_reasoning:
                        if show_thinking:
                            self.console.print("\n[dim italic]规划思考中...[/dim italic]")
                        in_reasoning = True
                    if show_thinking:
                        self.console.print(f"[dim]{rc}[/dim]", end="")

                if delta.content:
                    if in_reasoning:
                        if show_thinking:
                            self.console.print()
                        in_reasoning = False
                    plan_text += delta.content

            plan_text = plan_text.strip()
        except Exception as e:
            plan_text = f"[ ] Step 1: 完成用户任务：{user_input}"
            self.console.print(f"[dim yellow]规划阶段异常（{e}），使用默认单步计划[/dim yellow]")

        self.console.print(RichPanel(
            plan_text,
            title="[bold cyan]执行计划[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        ))

        # 解析步骤列表：匹配 "[ ] Step N: ..." 格式
        steps = _re.findall(r'\[ \]\s*Step\s*\d+:\s*(.+)', plan_text)
        if not steps:
            steps = [line.strip() for line in plan_text.splitlines() if line.strip()]
        return steps, plan_text

    _SUMMARIZE_SYSTEM = (
        "你是一个结果整理助手。以下是按步骤执行任务时每一步的输出，"
        "请将所有步骤的结果整合为一份连贯、完整、清晰的最终回答。\n"
        "要求：\n"
        "- 去除重复信息，合并相关内容\n"
        "- 保留所有关键事实和数据\n"
        "- 以用户易读的方式组织，不要保留步骤编号\n"
        "- 如果某步执行失败，如实说明但不重复错误细节\n"
        "- 直接输出整合结果，不要有前言"
    )

    _CHUNK_SUMMARIZE_SYSTEM = (
        _SUMMARIZE_SYSTEM + "\n"
        "- 你当前处理的是完整任务的一部分结果，必须尽量保留其中的事实、数据、限制和结论，"
        "为后续总整合服务，不要写无关铺垫。"
    )

    def _estimate_text_tokens(self, text: str) -> int:
        return self._token_tracker.estimate([{"role": "user", "content": text or ""}])

    def _split_step_block(self, header: str, content: str, token_budget: int) -> list[str]:
        block = f"{header}\n{content}".strip()
        if self._estimate_text_tokens(block) <= token_budget:
            return [block]

        lines = [line for line in (content or "").splitlines() if line.strip()]
        if not lines:
            max_chars = max(400, token_budget * 4)
            trimmed = (content or "")[:max_chars]
            return [f"{header}\n{trimmed}\n[单步结果已截断]".strip()]

        parts = []
        current_lines = []
        current_part = 1
        max_chars = max(300, token_budget * 4)

        def _flush():
            nonlocal current_lines, current_part
            if not current_lines:
                return
            part_header = f"{header} (part {current_part})"
            parts.append(f"{part_header}\n" + "\n".join(current_lines))
            current_part += 1
            current_lines = []

        for raw_line in lines:
            line = raw_line.strip()
            while line:
                candidate_lines = current_lines + [line]
                candidate_text = f"{header} (part {current_part})\n" + "\n".join(candidate_lines)
                if current_lines and self._estimate_text_tokens(candidate_text) > token_budget:
                    _flush()
                    continue
                if self._estimate_text_tokens(candidate_text) <= token_budget:
                    current_lines.append(line)
                    break

                chunk = line[:max_chars]
                candidate_chunk = f"{header} (part {current_part})\n" + "\n".join(current_lines + [chunk])
                if current_lines and self._estimate_text_tokens(candidate_chunk) > token_budget:
                    _flush()
                    continue
                current_lines.append(chunk)
                line = line[max_chars:]
                if line:
                    _flush()
        _flush()
        return parts or [block]

    def _build_summary_chunks(self, steps: list, step_contents: list, token_budget: int) -> list[str]:
        step_blocks = []
        for i, (step, content) in enumerate(zip(steps, step_contents), 1):
            header = f"### Step {i}: {step}"
            step_blocks.extend(self._split_step_block(header, content, token_budget))

        chunks = []
        current_blocks = []
        for block in step_blocks:
            candidate = "\n\n".join(current_blocks + [block])
            if current_blocks and self._estimate_text_tokens(candidate) > token_budget:
                chunks.append("\n\n".join(current_blocks))
                current_blocks = [block]
            else:
                current_blocks.append(block)
        if current_blocks:
            chunks.append("\n\n".join(current_blocks))
        return chunks or [""]

    def _run_summary_pass(self, model: str, system_prompt: str, user_prompt: str, stream_output: bool) -> str:
        extra_body, top_level_params = self._build_extra_body(model)
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            stream=stream_output,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        if top_level_params:
            kwargs.update(top_level_params)

        if stream_output:
            resp = self._client.chat.completions.create(**kwargs)
            summary = ""
            for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    summary += content
                    self.console.print(content, end="", markup=False, highlight=False)
            self.console.print()
            return summary

        kwargs["stream"] = False
        resp = self._client.chat.completions.create(**kwargs)
        return ((resp.choices[0].message.content or "") if resp.choices else "").strip()

    def _summarize_steps(self, user_input: str, steps: list, step_contents: list, model: str) -> str:
        """汇总多步骤执行结果为一份连贯的最终回答。"""
        # 1. 保存步骤内容到记忆，确保压缩后信息不丢失
        if self._memory_manager and self._memory_manager.available:
            for i, (step, content) in enumerate(zip(steps, step_contents), 1):
                self._store_episodic_records(f"step_summary_{i}", step_desc=step, step_output=content)

        # 2. 根据上下文窗口构造按步骤分块的汇总输入
        context_window = get_context_window(
            model,
            self.config["agent"].get("context_token_limit", 0),
            client=self._client,
            base_url=self.config.get("base_url"),
        )
        reserved_tokens = 2000
        token_budget = max(1200, context_window - reserved_tokens)
        chunks = self._build_summary_chunks(steps, step_contents, token_budget)

        # 3. 若超长则先分块汇总，再做最终整合
        try:
            if len(chunks) == 1:
                return self._run_summary_pass(
                    model,
                    self._SUMMARIZE_SYSTEM,
                    f"用户原始问题：{user_input}\n\n各步骤执行结果：\n\n{chunks[0]}",
                    stream_output=True,
                )

            partials = []
            for index, chunk_text in enumerate(chunks, 1):
                partial = self._run_summary_pass(
                    model,
                    self._CHUNK_SUMMARIZE_SYSTEM,
                    (
                        f"用户原始问题：{user_input}\n\n"
                        f"这是分块汇总的第 {index}/{len(chunks)} 块步骤结果。"
                        "请保留事实、数据、限制、结论和未完成项，供后续最终整合：\n\n"
                        f"{chunk_text}"
                    ),
                    stream_output=False,
                )
                partials.append(f"### Chunk {index}\n{partial}")

            combined = "\n\n".join(partials)
            return self._run_summary_pass(
                model,
                self._SUMMARIZE_SYSTEM,
                f"用户原始问题：{user_input}\n\n以下是各步骤结果的分块汇总，请整合为最终回答：\n\n{combined}",
                stream_output=True,
            )
        except Exception as e:
            self.console.print(f"[dim yellow]汇总失败（{e}），返回原始结果[/dim yellow]")
            return "\n\n".join(step_contents)

    _REPLAN_SYSTEM = (
        "你是一个任务监控器。根据当前步骤描述和最近工具调用情况，"
        "判断执行状态并只输出以下三个词之一，不要有任何其他内容：\n"
        "continue\n"
        "skip\n"
        "abort\n\n"
        "判断标准：\n"
        "- continue：遇到短暂障碍但方向正确，重置计数器继续当前步骤\n"
        "- skip：当前步骤无法完成，跳过进入下一步骤\n"
        "- abort：整个任务无法继续执行"
    )

    def _replan_node(self, step_desc: str, recent_tool_summary: str, low_model: str) -> str:
        """Re-planner：走低价模型，返回 continue / skip / abort。"""
        show_thinking = self.config["agent"].get("show_thinking", True)
        decision = ""
        try:
            extra_body, top_level_params = self._build_extra_body(low_model)
            kwargs = dict(
                model=low_model,
                messages=[
                    {"role": "system", "content": self._REPLAN_SYSTEM},
                    {"role": "user", "content": f"当前步骤：{step_desc}\n\n最近工具调用摘要：\n{recent_tool_summary}"},
                ],
                temperature=0.0,
                max_completion_tokens=10,
                stream=True,
            )
            if extra_body:
                kwargs["extra_body"] = extra_body
            if top_level_params:
                kwargs.update(top_level_params)

            stream = self._client.chat.completions.create(**kwargs)
            in_reasoning = False
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    if not in_reasoning:
                        if show_thinking:
                            self.console.print("\n[dim italic]重规划思考中...[/dim italic]")
                        in_reasoning = True
                    if show_thinking:
                        self.console.print(f"[dim]{rc}[/dim]", end="")

                if delta.content:
                    if in_reasoning:
                        if show_thinking:
                            self.console.print()
                        in_reasoning = False
                    decision += delta.content

            decision = decision.strip().lower()
        except Exception:
            decision = "continue"

        if decision not in ("continue", "skip", "abort"):
            decision = "continue"
        return decision

    @staticmethod
    def _should_replan(
        step_round: int,
        tool_max_rounds: int,
        tool_error_count: int,
        tool_max_errors: int,
        recent_hashes: list,
        repeat_window: int,
        replan_threshold: float,
    ):
        """算法四：三信号 OR 触发，返回 (bool, reason)。"""
        # 信号A：连续错误达上限
        if tool_error_count >= tool_max_errors:
            return True, f"连续错误达上限（{tool_max_errors}）"
        # 信号B：轮次消耗比超阈值
        if tool_max_rounds > 0 and step_round / tool_max_rounds > replan_threshold:
            return True, f"轮次消耗比 {step_round}/{tool_max_rounds} 超过 {replan_threshold}"
        # 信号C：工具输出重复（滑动窗口内出现相同哈希）
        if len(recent_hashes) >= repeat_window and len(set(recent_hashes[-repeat_window:])) == 1:
            return True, "工具输出重复检测"
        return False, ""

    def chat(self, user_input: str) -> str:
        self._snapshot_runtime_state()
        self._reset_runtime_state(user_input)
        self._restore_followup_state(user_input)
        self.session_manager.append_message(self.session_id, "user", user_input)
        messages = self._get_messages()
        self._upsert_runtime_state_message(messages)
        agent_cfg = self.config["agent"]
        tools_cfg = self.config.get("tools", {})
        tool_max_rounds = tools_cfg.get("tool_max_rounds", 10)
        tool_max_errors = tools_cfg.get("tool_max_errors", 3)

        # 模型路由
        history = self.session_manager.get_history(self.session_id)
        history_len = len(history)
        # 取最近 4 条消息作为路由上下文（不含刚写入的 user 消息）
        recent_ctx = [{"role": m["role"], "content": (m["content"] or "")[:300]}
                      for m in history[-5:-1]]
        routed_model, route_reason, complexity = self._router.route(
            user_input, history_len, agent_cfg, self._client,
            manual_model=self._manual_model if self._routing_locked else None,
            context_messages=recent_ctx,
            session_tag=self._session_theme,
        )
        active_model = routed_model

        # Token 估算（在 API 调用前本地估算 input token 数）
        estimated_input = self._token_tracker.estimate(messages)

        if self.config.get("routing", {}).get("show_routing_decision", True) and route_reason != "manual":
            self.console.print(f"[dim]路由决策：{active_model}  ({route_reason})  预估输入 ~{estimated_input} tokens[/dim]")

        # 规划节点：仅 plan 任务触发分步执行计划（由 AI 分类器显式判断）
        if complexity == "plan":
            steps, plan_text = self._plan_node(user_input, messages, active_model)
            self._set_plan_steps(steps)
            self._upsert_runtime_state_message(messages)
            # 注入全局计划供模型参考，明确禁止提前执行未到步骤
            messages.append({
                "role": "system",
                "content": (
                    f"【完整执行计划（仅供参考）】\n{plan_text}\n\n"
                    "重要：系统会逐步调用每个步骤，每次只执行当前被指定的步骤，不要提前执行其他步骤。"
                ),
            })
        else:
            steps = []

        # re-planner 所需配置
        replan_threshold = tools_cfg.get("replan_threshold", 0.6)
        replan_repeat_window = tools_cfg.get("replan_repeat_window", 3)
        low_model = self.config["model"].get("tier_to_model", {}).get("low") or active_model

        # 累加所有轮次工具调用的 token，避免中间轮消耗丢失
        _acc_input = 0
        _acc_output = 0
        _task_aborted = False

        def _run_tool_loop(step_desc: str = "", silent: bool = False):
            """单个 step（或无 plan 模式）的工具调用循环。
            返回 (full_content, loop_break_reason)，loop_break_reason: 'done'/'abort'
            silent=True 时不输出中间过程到终端，但 full_content 正常累加。
            """
            nonlocal _acc_input, _acc_output, _task_aborted

            step_round_count = 0
            step_error_count = 0
            recent_hashes: list = []
            from collections import deque as _deque
            recent_tool_lines: _deque = _deque(maxlen=replan_repeat_window)

            while True:
                _last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
                _active_tools = self._select_tools(_last_user) or None
                kwargs = dict(
                    model=active_model,
                    messages=messages,
                    tools=_active_tools,
                    stream=True,
                    temperature=agent_cfg.get("temperature", 0.7),
                    top_p=agent_cfg.get("top_p", 0.8),
                    stream_options={"include_usage": True},
                )
                extra_body, top_level_params = self._build_extra_body(active_model)
                if extra_body:
                    kwargs["extra_body"] = extra_body
                if top_level_params:
                    kwargs.update(top_level_params)

                try:
                    stream = self._client.chat.completions.create(**kwargs)
                except Exception as _api_err:
                    # auto 模式：若多条 system 消息导致 4xx 错误，自动降级并重试一次
                    _err_str = str(_api_err).lower()
                    _cur_msgs = kwargs["messages"]
                    _has_multi_sys = sum(1 for m in _cur_msgs if m.get("role") == "system") > 1
                    _skill_mode = self.config["agent"].get("skill_layer_mode", "auto")
                    if (
                        _has_multi_sys
                        and _skill_mode == "auto"
                        and not self._skill_layer_merged
                        and ("400" in _err_str or "bad request" in _err_str or "invalid" in _err_str)
                    ):
                        self.console.print("[dim yellow]多 system 消息不受支持，自动降级为 merge 模式并重试…[/dim yellow]")
                        self._skill_layer_merged = True
                        _sys_msgs = [m for m in _cur_msgs if m.get("role") == "system"]
                        _non_sys = [m for m in _cur_msgs if m.get("role") != "system"]
                        _merged = "\n\n".join(m["content"] for m in _sys_msgs if m.get("content"))
                        kwargs["messages"] = [{"role": "system", "content": _merged}] + _non_sys
                        stream = self._client.chat.completions.create(**kwargs)
                    else:
                        raise

                request_prompt_estimate = (
                    self._token_tracker.estimate(kwargs["messages"])
                    + self._token_tracker.estimate_tools(kwargs.get("tools"))
                )

                full_content = ""
                # <theme> 标签内联提取状态（每轮重置）
                _tbuf = ""          # 流开头的静默缓冲区
                _theme_done = False  # True = 标签阶段结束，进入正常输出
                _TOPEN = "<theme>"
                _TCLOSE = "</theme>"
                tool_calls_acc = []
                finish_reason = None
                in_reasoning = False
                call_input_tokens = 0

                for chunk in stream:
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                        completion_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
                        _acc_input += prompt_tokens
                        _acc_output += completion_tokens
                        call_input_tokens += prompt_tokens

                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason or finish_reason

                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        if not in_reasoning:
                            if not silent and self.config["agent"].get("show_thinking", True):
                                self.console.print("\n[dim italic]思考中...[/dim italic]")
                            in_reasoning = True
                        if not silent and self.config["agent"].get("show_thinking", True):
                            self.console.print(f"[dim]{rc}[/dim]", end="")

                    if delta.content:
                        if in_reasoning:
                            if not silent:
                                self.console.print()
                            in_reasoning = False
                        if not _theme_done:
                            _tbuf += delta.content
                            close_idx = _tbuf.find(_TCLOSE)
                            if close_idx != -1:
                                # 找到闭合标签
                                open_idx = _tbuf.find(_TOPEN)
                                if open_idx == 0:
                                    # 标签在最开头，提取主题并丢弃标签
                                    theme_val = _tbuf[len(_TOPEN):close_idx].strip()[:30]
                                    if theme_val:
                                        self._session_theme = theme_val
                                    remainder = _tbuf[close_idx + len(_TCLOSE):].lstrip("\n")
                                else:
                                    # 标签不在开头，整段当正文
                                    remainder = _tbuf
                                _theme_done = True
                                if remainder:
                                    if not silent:
                                        self.console.print(remainder, end="", markup=False, highlight=False)
                                    full_content += remainder
                            elif len(_tbuf) > 120:
                                # 超过 120 字符仍无标签，当正文处理
                                _theme_done = True
                                if not silent:
                                    self.console.print(_tbuf, end="", markup=False, highlight=False)
                                full_content += _tbuf
                        else:
                            if not silent:
                                self.console.print(delta.content, end="", markup=False, highlight=False)
                            full_content += delta.content

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            while len(tool_calls_acc) <= idx:
                                tool_calls_acc.append({
                                    "id": "", "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                })
                            if tc_delta.id:
                                tool_calls_acc[idx]["id"] = tc_delta.id
                            if tc_delta.function and tc_delta.function.name:
                                tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
                            if tc_delta.function and tc_delta.function.arguments:
                                tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

                # 流结束时若缓冲区未处理（极短回复无法闭合标签），作正文输出
                if not _theme_done and _tbuf:
                    if not silent:
                        self.console.print(_tbuf, end="", markup=False, highlight=False)
                    full_content += _tbuf

                if call_input_tokens and request_prompt_estimate:
                    self._token_tracker.calibrate_prompt_estimate(
                        request_prompt_estimate,
                        call_input_tokens,
                    )

                if full_content or in_reasoning:
                    if not silent:
                        self.console.print()

                # 模型主动停止 or [STEP_DONE] 信号 → 当前 step 完成
                if finish_reason != "tool_calls" or "[STEP_DONE]" in full_content:
                    return full_content, "done"

                # 轮次上限硬停（兜底，正常应由 re-planner 提前处理）
                if step_round_count >= tool_max_rounds:
                    self.console.print(f"[bold red]工具调用已达最大轮次（{tool_max_rounds}），强制停止。[/bold red]")
                    return full_content, "done"

                step_round_count += 1

                messages.append({
                    "role": "assistant",
                    "content": full_content or None,
                    "tool_calls": [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                        for tc in tool_calls_acc
                    ],
                })

                for tc in tool_calls_acc:
                    tool_name = tc["function"]["name"]
                    try:
                        tool_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}

                    if not silent:
                        show_thinking = self.config["agent"].get("show_thinking", True)
                        if show_thinking:
                            self.console.print(f"\n[bold cyan]调用工具：[/bold cyan]{tool_name}")
                            if tool_name == "execute_command":
                                cmd = tool_args.get("command", "")
                                desc = tool_args.get("description", "")
                                if cmd:
                                    self.console.print(f"[dim]$ {cmd}[/dim]")
                                if desc:
                                    self.console.print(f"[dim]意图：{desc}[/dim]")
                            elif tool_name == "generate_script":
                                filename = tool_args.get("filename", "")
                                language = tool_args.get("language", "")
                                if filename:
                                    self.console.print(f"[dim]生成文件：{filename}  语言：{language}[/dim]")
                            elif tool_name == "read_file":
                                self.console.print(f"[dim]读取：{tool_args.get('path', '')}[/dim]")
                            elif tool_name == "edit_file":
                                self.console.print(f"[dim]编辑：{tool_args.get('path', '')}[/dim]")
                        else:
                            hint = self._humanize_tool_hint(tool_name, tool_args)
                            self.console.print(f"[dim italic]{hint}[/dim italic]")

                    tool_result = self._call_tool(tool_name, tool_args)
                    self._record_tool_event(tool_name, tool_args, tool_result)
                    self._upsert_runtime_state_message(messages)

                    if tool_result.startswith("错误："):
                        step_error_count += 1
                    else:
                        step_error_count = 0

                    # 信号C：记录工具调用哈希
                    import hashlib as _hl
                    call_hash = _hl.md5(f"{tool_name}:{tool_result[:200]}".encode()).hexdigest()
                    recent_hashes.append(call_hash)
                    recent_tool_lines.append(f"  {tool_name}: {tool_result[:100]}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tool_name,
                        "content": tool_result,
                    })

                # 算法四：三信号检测
                triggered, reason = self._should_replan(
                    step_round_count, tool_max_rounds,
                    step_error_count, tool_max_errors,
                    recent_hashes, replan_repeat_window, replan_threshold,
                )
                if triggered and step_desc:
                    self.console.print(f"[dim yellow]Re-planner 触发（{reason}）[/dim yellow]")
                    recent_summary = "\n".join(recent_tool_lines)
                    decision = self._replan_node(step_desc, recent_summary, low_model)
                    self.console.print(f"[dim]Re-planner 决策：{decision}[/dim]")
                    if decision == "skip":
                        return full_content, "done"
                    elif decision == "abort":
                        _task_aborted = True
                        return full_content, "abort"
                    else:  # continue：重置计数器
                        step_round_count = 0
                        step_error_count = 0
                        recent_hashes.clear()

            # unreachable
            return "", "done"

        # ── 主执行逻辑 ──────────────────────────────────────────────
        last_content = ""
        step_contents = []  # 收集每步的输出
        if steps:
            # plan 模式：逐 step 执行
            silent_steps = not self.config["agent"].get("show_thinking", True)
            step_context = ""        # 注入到下一步 system message 的提示文本
            step_context_lines: list = []  # 用于构建 step_context 的原始条目
            for i, step_desc in enumerate(steps, 1):
                self.console.print(f"\n[bold cyan]► Step {i}/{len(steps)}：[/bold cyan]{step_desc}")
                self._set_current_step(step_desc)
                self._upsert_runtime_state_message(messages)
                # C+D：记录插入位置，注入含 [STEP_DONE] 协议的单步约束消息
                step_msg_idx = len(messages)
                step_context_hint = f"\n\n【前序步骤关键结论】\n{step_context}" if step_context else ""
                messages.append({
                    "role": "system",
                    "content": (
                        f"【执行第 {i} 步，共 {len(steps)} 步】{step_desc}\n"
                        f"你当前只负责执行第 {i} 步。其余步骤会由系统在后续轮次单独调用，不要跨步骤执行。"
                        "完成本步骤后立即在回复末尾输出 [STEP_DONE] 停止。"
                        "如果本步骤产生了对后续步骤有影响的重要信息（消歧义结论、关键发现、决策、确认的参数等），"
                        "请在 [STEP_DONE] 之前额外输出 [STEP_CONTEXT]一句话总结[/STEP_CONTEXT]。"
                        + step_context_hint
                    ),
                })
                last_content, reason = _run_tool_loop(step_desc, silent=silent_steps)
                # C：步骤结束后移除约束消息，工具调用链保留
                messages.pop(step_msg_idx)
                step_contents.append(last_content)
                # 解析 [STEP_CONTEXT] 标签，提取 AI 主动总结的关键信息
                import re as _re2
                ctx_match = _re2.search(r'\[STEP_CONTEXT\](.*?)\[/STEP_CONTEXT\]', last_content, _re2.DOTALL)
                if ctx_match:
                    entry = ctx_match.group(1).strip()[:100]  # AI 主动标注，限 100 字
                    step_context_lines.append(f"Step {i}（{step_desc[:20]}）：{entry}")
                elif last_content.strip():
                    snippet = last_content.strip()[:80].replace("[STEP_DONE]", "").strip()
                    step_context_lines.append(f"Step {i}（{step_desc[:20]}）：{snippet}")
                # 更新 step_context 供下一步注入（最多保留最近 3 条）
                step_context = "\n".join(step_context_lines[-3:])
                self._record_step_completion(i, step_desc, last_content)
                self._upsert_runtime_state_message(messages)
                self.console.print(f"[dim green]✓ Step {i} 完成[/dim green]")
                if _task_aborted:
                    self.console.print("[bold red]任务已中止。[/bold red]")
                    break
        else:
            # simple 模式：单层循环（无 step 分割，step_desc 为空，re-planner 不介入）
            last_content, _ = _run_tool_loop("")
            reply_summary = self._summarize_reply_conclusion(last_content)
            if reply_summary:
                self._add_working_set_item(reply_summary)
            pending_step = self._extract_pending_step(last_content)
            if pending_step:
                self._set_pending_step(pending_step)
                if not self._runtime_task_state.get("current_step"):
                    self._set_current_step(pending_step)

        # 多步骤汇总
        if steps and step_contents and not _task_aborted:
            self.console.print("\n[bold green]所有步骤执行完毕，正在整理结果…[/bold green]")
            last_content = self._summarize_steps(user_input, steps, step_contents, low_model)

        # 记录 session + token + 记忆
        self.session_manager.append_message(self.session_id, "assistant", self._strip_markdown(last_content))
        if _acc_input or _acc_output:
            self._token_tracker.record_usage(
                self.session_manager, self.session_id, active_model,
                {"prompt_tokens": _acc_input, "completion_tokens": _acc_output},
            )
        if self._memory_manager and self._memory_manager.available:
            history_for_mem = self.session_manager.get_history(self.session_id)
            turn_count = len(history_for_mem)
            if self._memory_manager.should_extract(turn_count, last_content):
                recent = [{"role": m["role"], "content": m["content"]} for m in history_for_mem[-6:]]
                self._store_episodic_records("periodic_extract", recent_messages=recent, async_write=True)
        return last_content

    def get_token_summary(self) -> dict:
        return self._token_tracker.get_session_summary(
            self.session_manager, self.session_id
        )

    def toggle_skill(self, name: str, enabled: bool) -> bool:
        """激活或停用指定 skill，返回是否找到该 skill。"""
        for skill in self._skills:
            if skill["name"] == name:
                skill["enabled"] = enabled
                # 同步 _skills_enabled_names，确保 reload_skills() 后状态不丢失
                if enabled:
                    if name not in self._skills_enabled_names:
                        self._skills_enabled_names.append(name)
                else:
                    if name in self._skills_enabled_names:
                        self._skills_enabled_names.remove(name)
                # skill 激活状态变化，清零 skill token 缓存，下一轮重新计算
                self._last_skill_system_tokens = 0
                return True
        return False

    def switch_session(self, new_session_id: str):
        """切换到指定会话：更新 session_id、重置 token 计数和会话主题。"""
        self.session_id = new_session_id
        self._token_tracker = TokenTracker()
        self._session_theme = ""  # 新会话重置主题，等待重新提取
        self._last_main_system_tokens = 0
        self._last_skill_system_tokens = 0
        self._skill_layer_merged = False  # 新会话重新尝试双 system 模式

    def toggle_auto_confirm(self):
        self.auto_confirm = not self.auto_confirm
        state = "开启（自动确认）" if self.auto_confirm else "关闭（需手动确认）"
        self.console.print(f"[bold green]命令自动确认已{state}[/bold green]")
