import re
import subprocess

# 直接拦截：触发即拒绝执行
BLACKLIST_PATTERNS = [
    r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+/",
    r"rm\s+-[a-zA-Z]*f[a-zA-Z]*r?\s+/",
    r"rm\s+-rf\s+~",
    r"rm\s+-fr\s+~",
    r"mkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{.*:\|:.*\}",           # fork 炸弹
    r"chmod\s+-R\s+777\s+/",
    r"base64\s+.*\|.*\b(bash|sh)\b",   # base64 解码后执行
    r"curl\s+.*\|.*\b(bash|sh)\b",     # curl pipe bash
    r"wget\s+.*-O\s*-.*\|.*\b(bash|sh)\b",
    r"\beval\s*[\(\`\"]",              # eval 执行
    r"chmod\s+[ugo+]*s\s",            # setuid/setgid
    r">\s*/etc/passwd",
    r">\s*/etc/shadow",
]

# 强制确认：即使 auto_confirm=True 也需要手动确认
FORCE_CONFIRM_PATTERNS = [
    r"\brm\b",
    r"\bchmod\b",
    r"\bchown\b",
    r"\bdd\b",
    r"\bmkfs\b",
    r"\bsystemctl\s+(stop|disable|mask)\b",
    r"\bkill\b",
    r"\bpkill\b",
    r"\bdrop\s+(table|database)\b",
    r"\btruncate\b",
]


def _truncate_output(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[:max_chars - 600]
    tail = text[-500:]
    omitted_lines = text[max_chars - 600:-500].count("\n")
    return f"{head}\n[...截断 {omitted_lines} 行，输出过长已省略...]\n{tail}"

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "execute_command",
        "description": (
            "在本地终端执行一条 Shell 命令。"
            "仅在确实需要运行系统命令时使用，例如查看文件、安装依赖、运行脚本等。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的完整 Shell 命令",
                },
                "description": {
                    "type": "string",
                    "description": "对该命令意图的简短说明，便于用户理解",
                },
            },
            "required": ["command", "description"],
        },
    },
}


def execute(args: dict, auto_confirm: bool = False, console=None, timeout: int = 30, output_max_chars: int = 3000) -> str:
    command = args.get("command", "").strip()
    description = args.get("description", "")

    if not command:
        return "错误：命令为空。"

    for pattern in BLACKLIST_PATTERNS:
        if re.search(pattern, command):
            return f"拒绝执行：命令触发安全黑名单规则，已阻止。\n命令：{command}"

    # 判断是否需要强制确认（即使 auto_confirm=True）
    force_confirm = any(re.search(p, command) for p in FORCE_CONFIRM_PATTERNS)

    if not auto_confirm or force_confirm:
        if console:
            console.print(f"\n[bold yellow]待执行命令：[/bold yellow] {command}")
            console.print(f"[bold yellow]意图说明：[/bold yellow] {description}")
            if force_confirm and auto_confirm:
                console.print("[bold red]⚠ 该命令属于高风险操作，需手动确认（忽略 auto_confirm 设置）[/bold red]")
            confirm = console.input("[bold red]是否执行？(y/N): [/bold red]").strip().lower()
        else:
            print(f"\n待执行命令：{command}")
            print(f"意图说明：{description}")
            confirm = input("是否执行？(y/N): ").strip().lower()

        if confirm != "y":
            return "命令已取消，未执行。"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        status = "成功" if result.returncode == 0 else f"失败（返回码 {result.returncode}）"
        stdout = result.stdout.strip() if result.stdout else "（无输出）"
        stderr = result.stderr.strip() if result.stderr else "（无）"
        raw = (
            f"[命令执行结果]\n"
            f"命令：{command}\n"
            f"状态：{status}\n"
            f"--- 标准输出 ---\n"
            f"{stdout}\n"
            f"--- 错误输出 ---\n"
            f"{stderr}"
        )
        return _truncate_output(raw, output_max_chars)
    except subprocess.TimeoutExpired:
        return (
            f"[命令执行结果]\n"
            f"命令：{command}\n"
            f"状态：超时（>{timeout} 秒）已终止"
        )
