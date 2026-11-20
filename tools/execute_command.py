import locale
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
    marker = f"\n[...输出已截断，原始长度 {len(text)} 字符...]\n"
    available = max_chars - len(marker)
    if available < 1:
        return "错误：命令输出上限过小，请提高 tools.output_max_chars"
    tail_size = min(4000, available // 4)
    head = text[:available - tail_size]
    tail = text[-tail_size:] if tail_size else ""
    return head + marker + tail


def _decode_output(data: bytes, encoding: str) -> tuple[str, bool]:
    try:
        text = data.decode(encoding)
        replaced = False
    except UnicodeDecodeError:
        text = data.decode(encoding, errors="replace")
        replaced = True
    return text.replace("\r\n", "\n").replace("\r", "\n"), replaced


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "execute_command",
        "description": "执行本地 Shell 命令。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell 命令"},
                "description": {"type": "string", "description": "命令意图说明"},
            },
            "required": ["command", "description"],
        },
    },
}


def execute(args: dict, auto_confirm: bool = False, console=None, timeout: int = 30, output_max_chars: int = 32000) -> str:
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
            timeout=timeout,
        )
        status = "成功" if result.returncode == 0 else f"失败（返回码 {result.returncode}）"
        encoding = locale.getpreferredencoding(False)
        stdout, stdout_replaced = _decode_output(result.stdout, encoding)
        stderr, stderr_replaced = _decode_output(result.stderr, encoding)
        stdout = stdout.strip() if stdout else "（无输出）"
        stderr = stderr.strip() if stderr else "（无）"
        affected = [name for name, replaced in (
            ("标准输出", stdout_replaced), ("错误输出", stderr_replaced),
        ) if replaced]
        warning = (
            f"警告：{'、'.join(affected)}含无法按 {encoding} 解码的字节，已用替换字符表示，输出可能不完整。\n"
            if affected else ""
        )
        raw = (
            f"[命令执行结果]\n"
            f"命令：{command}\n"
            f"状态：{status}\n"
            f"{warning}"
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
