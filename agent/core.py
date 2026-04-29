import json

from openai import OpenAI

import os

from tools.registry import discover_tools
from tools import execute_command as execute_command_module
import tools.memory as memory_module
import tools.web_search as web_search_module
from agent.token_tracker import TokenTracker
from agent.router import ModelRouter
from agent import summarizer
from agent.skill_registry import discover_skills
from agent.context_window import get_context_window
from memory.manager import MemoryManager


class Agent:
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
        # 历史摘要：优先从持久化存储恢复，确保重启后上下文不丢失
        self._history_summary = session_manager.get_summary(session_id)

        self._client = OpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url") or None,
        )
        self.tool_definitions, self.tool_executors = discover_tools()

        # 初始化分层记忆管理器（必要组件，失败时向上抛出）
        self._memory_manager = MemoryManager(
            config["memory"],
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            storage=session_manager,
        )
        memory_module.set_memory_manager(self._memory_manager)
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

    def reload_skills(self):
        """重新扫描 skills/ 目录，热更新已加载的 skill 列表。"""
        self._skills = discover_skills(self._skills_dir, self._skills_enabled_names)

    def _get_messages(self) -> list:
        history = self.session_manager.get_history(self.session_id)

        # 构建不含 system prompt 的原始消息列表，用于 token 估算
        raw_messages = [{"role": m["role"], "content": m["content"]} for m in history]

        # 动态获取当前模型的上下文窗口大小
        context_window = get_context_window(
            self.model,
            self.config["agent"].get("context_token_limit", 0),
        )
        threshold = self.config["agent"].get("compress_threshold", 0.60)
        tool_max_chars = self.config["agent"].get("tool_output_max_chars", 300)

        estimated_tokens = self._token_tracker.estimate(raw_messages)
        if summarizer.should_compress(estimated_tokens, context_window, threshold):
            # 触发前先同步 mem0 提取，确保即将被压缩的内容已落入记忆
            if self._memory_manager and self._memory_manager.available:
                self._memory_manager.add(raw_messages, layer="episodic")
            # 纯机械压缩，只在内存中生效，不落盘
            raw_messages = summarizer.compress_pipeline(
                raw_messages,
                self._token_tracker,
                context_window,
                threshold,
                tool_max_chars,
            )

        # 每轮动态构建 system prompt：base + 记忆层 + 历史摘要（不修改 config，避免累积）
        base_prompt = self.config["agent"].get("system_prompt", "").strip()

        # 通用行为规则：信息不足时优先提问
        _ask_rule = (
            "\n\n【行为规则】\n"
            "当且仅当缺少只有用户才知的关键信息且无法合理推断时，才简短向用户提问。"
            "如果信息已足够执行至少一个有效动作，应先行动再根据结果调整，不要反复追问。"
            "可以通过搜索或文件读取获取的信息不算关键信息不足。"
        )
        base_prompt += _ask_rule

        # 会话意图锚：取第一条 user 消息作为锁定开始点
        first_user = next(
            (m["content"] for m in (self.session_manager.get_history(self.session_id) or [])
             if m.get("role") == "user"), ""
        )
        if first_user:
            anchor = first_user[:100] + ("..." if len(first_user) > 100 else "")
            base_prompt += f"\n\n【本次会话起点】\n{anchor}"

        # Layer1 核心记忆
        if self._memory_manager and self._memory_manager.available:
            core_items = self._memory_manager.search_core()
            memory_block = ("\n\n【用户记忆】\n" + "\n".join(f"  {m}" for m in core_items)) if core_items else ""
        else:
            memories = self.session_manager.list_memories()
            memory_block = ("\n\n【用户记忆】\n" + "\n".join(f"  {k}: {v}" for k, v in memories.items())) if memories else ""

        # Layer2 情节记忆（语义检索，取最后一条用户消息作为 query）
        episodic_block = ""
        if self._memory_manager and self._memory_manager.available:
            last_user = next((m["content"] for m in reversed(raw_messages) if m.get("role") == "user"), "")
            if last_user:
                top_k = self.config.get("memory", {}).get("episodic_top_k", 3)
                episodic_items = self._memory_manager.search_episodic(last_user, top_k=top_k)
                if episodic_items:
                    episodic_block = "\n\n【相关记忆】\n" + "\n".join(f"  {m}" for m in episodic_items)

        # Layer3：只读旧摘要，不再写入
        summary_block = f"\n\n【早期对话摘要】\n{self._history_summary}" if self._history_summary else ""

        # 激活的 skill prompt 追加在最末尾
        skill_block = "\n\n".join(s["prompt"] for s in self._skills if s["enabled"])
        system_prompt = base_prompt + memory_block + episodic_block + summary_block
        if skill_block:
            system_prompt += "\n\n" + skill_block
        if system_prompt.strip():
            raw_messages.insert(0, {"role": "system", "content": system_prompt.strip()})
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

    def _build_extra_body(self, model: str):
        extra = dict(self.config["model"].get("extra_params", {}).get(model, {}))
        return extra if extra else None

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
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=plan_messages,
                temperature=0.3,
                stream=False,
            )
            plan_text = resp.choices[0].message.content.strip()
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

    def _summarize_steps(self, user_input: str, steps: list, step_contents: list, model: str) -> str:
        """汇总多步骤执行结果为一份连贯的最终回答。"""
        # 1. 保存步骤内容到记忆，确保压缩后信息不丢失
        if self._memory_manager and self._memory_manager.available:
            for i, (step, content) in enumerate(zip(steps, step_contents), 1):
                self._memory_manager.add(
                    [{"role": "assistant", "content": f"[Step {i}: {step}]\n{content}"}],
                    layer="episodic",
                )

        # 2. 构造完整的步骤文本（不截断）
        steps_text = ""
        for i, (step, content) in enumerate(zip(steps, step_contents), 1):
            steps_text += f"### Step {i}: {step}\n{content}\n\n"

        # 3. 估算 token，若超上下文窗口则压缩步骤内容
        context_window = get_context_window(model, self.config["agent"].get("context_token_limit", 0))
        reserved_tokens = 2000
        steps_messages = [{"role": "user", "content": steps_text}]
        estimated_tokens = self._token_tracker.estimate(steps_messages)
        if estimated_tokens + reserved_tokens > context_window:
            threshold = 0.5
            steps_messages = summarizer.compress_pipeline(
                steps_messages,
                self._token_tracker,
                context_window - reserved_tokens,
                threshold,
                self.config["agent"].get("tool_output_max_chars", 300),
            )
            steps_text = steps_messages[0]["content"] if steps_messages else steps_text

        # 4. 调用模型汇总
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self._SUMMARIZE_SYSTEM},
                    {"role": "user", "content": f"用户原始问题：{user_input}\n\n各步骤执行结果：\n\n{steps_text}"},
                ],
                temperature=0.3,
                stream=True,
            )
            summary = ""
            for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    summary += content
                    self.console.print(content, end="", markup=False, highlight=False)
            self.console.print()
            return summary
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
        try:
            resp = self._client.chat.completions.create(
                model=low_model,
                messages=[
                    {"role": "system", "content": self._REPLAN_SYSTEM},
                    {"role": "user", "content": f"当前步骤：{step_desc}\n\n最近工具调用摘要：\n{recent_tool_summary}"},
                ],
                temperature=0.0,
                max_tokens=10,
                stream=False,
            )
            decision = resp.choices[0].message.content.strip().lower()
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
        self.session_manager.append_message(self.session_id, "user", user_input)
        messages = self._get_messages()
        agent_cfg = self.config["agent"]
        tools_cfg = self.config.get("tools", {})
        tool_max_rounds = tools_cfg.get("tool_max_rounds", 10)
        tool_max_errors = tools_cfg.get("tool_max_errors", 3)

        # 模型路由
        history_len = len(self.session_manager.get_history(self.session_id))
        routed_model, route_reason, complexity = self._router.route(
            user_input, history_len, agent_cfg, self._client,
            manual_model=self._manual_model if self._routing_locked else None,
        )
        active_model = routed_model

        # Token 估算（在 API 调用前本地估算 input token 数）
        estimated_input = self._token_tracker.estimate(messages)

        if self.config.get("routing", {}).get("show_routing_decision", True) and route_reason != "manual":
            self.console.print(f"[dim]路由决策：{active_model}  ({route_reason})  预估输入 ~{estimated_input} tokens[/dim]")

        # 规划节点：medium / complex 任务先生成执行计划
        if complexity in ("medium", "complex"):
            steps, plan_text = self._plan_node(user_input, messages, active_model)
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
                kwargs = dict(
                    model=active_model,
                    messages=messages,
                    tools=self.tool_definitions if self.tool_definitions else None,
                    stream=True,
                    temperature=agent_cfg.get("temperature", 0.7),
                    top_p=agent_cfg.get("top_p", 0.8),
                    stream_options={"include_usage": True},
                )
                extra_body = self._build_extra_body(active_model)
                if extra_body:
                    kwargs["extra_body"] = extra_body

                stream = self._client.chat.completions.create(**kwargs)

                full_content = ""
                tool_calls_acc = []
                finish_reason = None
                in_reasoning = False

                for chunk in stream:
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        _acc_input += getattr(chunk.usage, "prompt_tokens", 0) or 0
                        _acc_output += getattr(chunk.usage, "completion_tokens", 0) or 0

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
            for i, step_desc in enumerate(steps, 1):
                self.console.print(f"\n[bold cyan]► Step {i}/{len(steps)}：[/bold cyan]{step_desc}")
                # C+D：记录插入位置，注入含 [STEP_DONE] 协议的单步约束消息
                step_msg_idx = len(messages)
                messages.append({
                    "role": "system",
                    "content": (
                        f"【执行第 {i} 步，共 {len(steps)} 步】{step_desc}\n"
                        f"你当前只负责执行第 {i} 步。其余步骤会由系统在后续轮次单独调用，不要跨步骤执行。"
                        "完成本步骤后立即在回复末尾输出 [STEP_DONE] 停止。"
                    ),
                })
                last_content, reason = _run_tool_loop(step_desc, silent=silent_steps)
                # C：步骤结束后移除约束消息，工具调用链保留
                messages.pop(step_msg_idx)
                step_contents.append(last_content)
                self.console.print(f"[dim green]✓ Step {i} 完成[/dim green]")
                if _task_aborted:
                    self.console.print("[bold red]任务已中止。[/bold red]")
                    break
        else:
            # simple 模式：单层循环（无 step 分割，step_desc 为空，re-planner 不介入）
            last_content, _ = _run_tool_loop("")

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
            history = self.session_manager.get_history(self.session_id)
            turn_count = len(history)
            if self._memory_manager.should_extract(turn_count, last_content):
                recent = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
                self._memory_manager.extract_async(recent)
        return last_content

    def get_token_summary(self) -> dict:
        return self._token_tracker.get_session_summary(
            self.session_manager, self.session_id, self.config.get("pricing", {})
        )

    def toggle_skill(self, name: str, enabled: bool) -> bool:
        """激活或停用指定 skill，返回是否找到该 skill。"""
        for skill in self._skills:
            if skill["name"] == name:
                skill["enabled"] = enabled
                return True
        return False

    def switch_session(self, new_session_id: str):
        """切换到指定会话：更新 session_id、重载历史摘要、重置 token 计数。"""
        self.session_id = new_session_id
        self._history_summary = self.session_manager.get_summary(new_session_id)
        self._token_tracker = TokenTracker()

    def toggle_auto_confirm(self):
        self.auto_confirm = not self.auto_confirm
        state = "开启（自动确认）" if self.auto_confirm else "关闭（需手动确认）"
        self.console.print(f"[bold green]命令自动确认已{state}[/bold green]")
