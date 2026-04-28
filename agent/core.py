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
        # 历史摘要：优先从 Redis 恢复，确保重启后上下文不丢失
        self._history_summary = session_manager.get_summary(session_id)

        self._client = OpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url") or None,
        )
        self.tool_definitions, self.tool_executors = discover_tools()

        # 注入 session_manager 到 memory 工具
        memory_module.set_session_manager(session_manager)

        # 初始化分层记忆管理器
        self._memory_manager = None
        if config.get("memory", {}).get("enabled", False):
            try:
                self._memory_manager = MemoryManager(
                    config["memory"],
                    config["redis"],
                    config["api_key"],
                    config.get("base_url"),
                )
                if self._memory_manager.available:
                    memory_module.set_memory_manager(self._memory_manager)
                    console.print("[dim]分层记忆已启用（mem0 + 向量库）[/dim]")
                else:
                    console.print("[dim yellow]分层记忆初始化失败，降级为 Redis KV 模式[/dim yellow]")
            except Exception as e:
                console.print(f"[dim yellow]分层记忆加载异常：{e}，降级为 Redis KV 模式[/dim yellow]")

        # 注入 search 配置到 web_search 工具
        web_search_module.set_config(config.get("search", {}))

        # Token 追踪器
        self._token_tracker = TokenTracker()
        if self._token_tracker.mode == "unavailable":
            console.print(
                "[dim yellow]Token 估算模式：unavailable（字符粗估，误差较大）[/dim yellow]"
            )

        # 模型路由器（注入 Redis 客户端，用于路由分类结果缓存）
        self._router = ModelRouter(
            config.get("routing", {}),
            redis_client=session_manager.redis_client,
            default_model=config["model"]["default"],
        )

        # 启动时扫描并加载 skills/ 目录下所有 skill
        _project_root = os.path.dirname(os.path.dirname(__file__))
        _skills_dir = os.path.join(_project_root, "skills")
        _enabled_names = config.get("skills", {}).get("enabled", [])
        self._skills: list = discover_skills(_skills_dir, _enabled_names)
        if self._skills:
            active = [s["name"] for s in self._skills if s["enabled"]]
            console.print(f"[dim]已加载 {len(self._skills)} 个 skill，激活：{active or '无'}[/dim]")

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
            # 纯机械压缩，只在内存中生效，不写入 Redis
            raw_messages = summarizer.compress_pipeline(
                raw_messages,
                self._token_tracker,
                context_window,
                threshold,
                tool_max_chars,
            )

        # 每轮动态构建 system prompt：base + 记忆层 + 历史摘要（不修改 config，避免累积）
        base_prompt = self.config["agent"].get("system_prompt", "").strip()

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

    def _build_extra_body(self, model: str):
        extra = dict(self.config["model"].get("extra_params", {}).get(model, {}))
        return extra if extra else None

    def chat(self, user_input: str) -> str:
        self.session_manager.append_message(self.session_id, "user", user_input)
        messages = self._get_messages()
        agent_cfg = self.config["agent"]
        tools_cfg = self.config.get("tools", {})
        tool_max_retries = tools_cfg.get("tool_max_retries", 2)

        # 模型路由
        history_len = len(self.session_manager.get_history(self.session_id))
        routed_model, route_reason = self._router.route(
            user_input, history_len, agent_cfg, self._client,
            manual_model=self._manual_model if self._routing_locked else None,
        )
        active_model = routed_model

        # Token 估算（在 API 调用前本地估算 input token 数）
        estimated_input = self._token_tracker.estimate(messages)

        if self.config.get("routing", {}).get("show_routing_decision", True) and route_reason != "manual":
            self.console.print(f"[dim]路由决策：{active_model}  ({route_reason})  预估输入 ~{estimated_input} tokens[/dim]")

        tool_retry_count = 0
        # 累加所有轮次工具调用的 token，避免中间轮消耗丢失
        _acc_input = 0
        _acc_output = 0

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
                # 累加每轮的 token 消耗
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    _acc_input += getattr(chunk.usage, "prompt_tokens", 0) or 0
                    _acc_output += getattr(chunk.usage, "completion_tokens", 0) or 0

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason or finish_reason

                # 思维链内容（qwen3 系列）
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    if not in_reasoning:
                        if self.config["agent"].get("show_thinking", True):
                            self.console.print("\n[dim italic]思考中...[/dim italic]")
                        in_reasoning = True
                    if self.config["agent"].get("show_thinking", True):
                        self.console.print(f"[dim]{rc}[/dim]", end="")

                # 正常回复内容（流式打印）
                if delta.content:
                    if in_reasoning:
                        self.console.print()  # 结束思维链换行
                        in_reasoning = False
                    self.console.print(delta.content, end="", markup=False, highlight=False)
                    full_content += delta.content

                # 累积 tool_calls 分片
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        while len(tool_calls_acc) <= idx:
                            tool_calls_acc.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            })
                        if tc_delta.id:
                            tool_calls_acc[idx]["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

            if full_content or in_reasoning:
                self.console.print()  # 流式结束后换行

            if finish_reason == "tool_calls":
                # 工具重试上限检查
                if tool_retry_count >= tool_max_retries:
                    self.console.print(
                        f"[bold red]工具调用已达最大重试次数（{tool_max_retries}），停止执行。[/bold red]"
                    )
                    self.session_manager.append_message(self.session_id, "assistant", full_content or "[TOOL_FAILED]")
                    break

                tool_retry_count += 1

                messages.append({
                    "role": "assistant",
                    "content": full_content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for tc in tool_calls_acc
                    ],
                })

                for tc in tool_calls_acc:
                    tool_name = tc["function"]["name"]
                    try:
                        tool_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}

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

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tool_name,
                        "content": tool_result,
                    })
            else:
                self.session_manager.append_message(self.session_id, "assistant", full_content)
                # 记录所有轮次累积的真实 token 消耗（含中间工具调用轮次）
                if _acc_input or _acc_output:
                    self._token_tracker.record_usage(
                        self.session_manager, self.session_id, active_model,
                        {"prompt_tokens": _acc_input, "completion_tokens": _acc_output},
                    )
                # 触发异步情节记忆提取
                if self._memory_manager and self._memory_manager.available:
                    history = self.session_manager.get_history(self.session_id)
                    turn_count = len(history)
                    if self._memory_manager.should_extract(turn_count, full_content):
                        recent = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
                        self._memory_manager.extract_async(recent)
                return full_content

        return full_content

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

    def toggle_auto_confirm(self):
        self.auto_confirm = not self.auto_confirm
        state = "开启（自动确认）" if self.auto_confirm else "关闭（需手动确认）"
        self.console.print(f"[bold green]命令自动确认已{state}[/bold green]")
