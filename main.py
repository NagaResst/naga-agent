import sys
import os
import threading

sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import ANSI, HTML

from agent.config import load_config
from agent.core import Agent
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
  /search on          开启模型联网搜索
  /search off         关闭模型联网搜索
  /model <名称>       切换模型，例如 /model qwen-plus（同时固定模型，跳过路由）
  /model auto         恢复自动路由模式
  /routing on         开启智能模型路由
  /routing off        关闭智能模型路由
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
  /history            显示最近 5 条对话摘要
  /clear              清除当前会话的所有历史消息
  /help               显示此帮助信息
"""


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
    is_new_session = session_manager.get_history(session_id) == []

    console.print(f"\n[dim]会话：{session_id}  模型：{model}  输入 /help 查看命令[/dim]\n")

    def _context_toolbar():
        """底部固定状态栏：显示当前模型和上下文窗口用量。"""
        try:
            history = session_manager.get_history(agent.session_id)
            messages = [{"role": m["role"], "content": m["content"]} for m in history]
            used = agent._token_tracker.estimate(messages)
            max_ctx = get_context_window(
                agent.model,
                agent.config["agent"].get("context_token_limit", 0),
            )
            pct = used / max_ctx * 100 if max_ctx > 0 else 0
            if pct < 60:
                color = "ansigreen"
            elif pct < 80:
                color = "ansiyellow"
            else:
                color = "ansired"
            max_k = f"{max_ctx // 1024}k" if max_ctx >= 1024 else str(max_ctx)
            return HTML(
                f" <b>模型:</b> {agent.model}  "
                f"<b>上下文:</b> <{color}>{used:,} / {max_k} ({pct:.1f}%)</{color}> "
            )
        except Exception:
            return ""

    while True:
        try:
            user_input = pt_prompt(f"{prompt_prefix} > ", bottom_toolbar=_context_toolbar).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]已退出。[/dim]")
            break

        if not user_input:
            continue

        if user_input in ("/quit", "/exit"):
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

        elif user_input == "/search on":
            console.print("[red]模型联网搜索已禁用（独立计费），请使用 web_search 工具进行搜索。[/red]")

        elif user_input == "/search off":
            console.print("[red]模型联网搜索已禁用（独立计费），请使用 web_search 工具进行搜索。[/red]")

        elif user_input.startswith("/model "):
            new_model = user_input[7:].strip()
            if new_model == "auto":
                agent._routing_locked = False
                console.print("[green]已恢复自动路由模式。[/green]")
            elif new_model in config["model"]["available"]:
                agent.model = new_model
                agent._manual_model = new_model
                agent._routing_locked = True
                console.print(f"[green]模型已固定为：{new_model}（已跳过自动路由）[/green]")
            else:
                console.print(f"[red]不支持的模型：{new_model}[/red]")
                console.print(f"可用模型：{', '.join(config['model']['available'])}")

        elif user_input == "/routing on":
            agent.config["routing"]["enabled"] = True
            console.print("[green]智能模型路由已开启[/green]")

        elif user_input == "/routing off":
            agent.config["routing"]["enabled"] = False
            console.print("[green]智能模型路由已关闭（固定使用当前模型）[/green]")

        elif user_input == "/token":
            summary = agent.get_token_summary()
            if summary.get("turns", 0) == 0:
                console.print("[dim]本会话暂无 token 记录。[/dim]")
            else:
                console.print(f"\n[bold]Token 用量明细[/bold]  (估算模式: {summary['tokenizer_mode']})")
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

        elif user_input == "/session new":
            new_id = session_manager.create_session("new")
            agent.switch_session(new_id)
            is_new_session = True
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
                    is_new_session = session_manager.get_history(target["id"]) == []
                    console.print(f"[green]已切换到会话：{target['name']}（{target['id']}）[/green]")
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
            console.print(f"模型路由：{'开启' if agent.config.get('routing', {}).get('enabled') else '关闭'}")
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
            confirm = Prompt.ask("确认清除当前会话所有消息？(y/N)", default="N", console=console).strip().lower()
            if confirm == "y":
                session_manager.clear_messages(session_id)
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

        else:
            try:
                agent.chat(user_input)
                if is_new_session:
                    is_new_session = False
                    threading.Thread(
                        target=_auto_rename_session,
                        args=(session_manager, session_id, user_input, agent),
                        daemon=True,
                    ).start()
            except Exception as e:
                console.print(f"[bold red]错误：{e}[/bold red]")


if __name__ == "__main__":
    main()
