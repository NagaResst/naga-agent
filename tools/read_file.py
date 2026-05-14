import os

_FILE_TYPE_MAP = {
    ".py": "Python \u6e90\u4ee3\u7801",
    ".js": "JavaScript \u6e90\u4ee3\u7801",
    ".ts": "TypeScript \u6e90\u4ee3\u7801",
    ".sh": "Shell \u811a\u672c",
    ".bash": "Bash \u811a\u672c",
    ".yaml": "YAML \u914d\u7f6e",
    ".yml": "YAML \u914d\u7f6e",
    ".json": "JSON \u6570\u636e",
    ".toml": "TOML \u914d\u7f6e",
    ".ini": "INI \u914d\u7f6e",
    ".conf": "\u914d\u7f6e\u6587\u4ef6",
    ".cfg": "\u914d\u7f6e\u6587\u4ef6",
    ".xml": "XML \u6587\u6863",
    ".html": "HTML \u9875\u9762",
    ".md": "Markdown \u6587\u6863",
    ".txt": "\u7eaf\u6587\u672c",
    ".log": "\u65e5\u5fd7\u6587\u4ef6",
    ".sql": "SQL \u811a\u672c",
    ".go": "Go \u6e90\u4ee3\u7801",
    ".java": "Java \u6e90\u4ee3\u7801",
    ".rs": "Rust \u6e90\u4ee3\u7801",
    ".rb": "Ruby \u6e90\u4ee3\u7801",
    ".php": "PHP \u6e90\u4ee3\u7801",
    ".css": "CSS \u6837\u5f0f\u8868",
    ".tf": "Terraform \u914d\u7f6e",
    ".env": "\u73af\u5883\u53d8\u91cf\u6587\u4ef6",
    ".groovy": "Groovy \u811a\u672c",
    ".kt": "Kotlin \u6e90\u4ee3\u7801",
    ".dockerfile": "Dockerfile",
}


def _detect_file_type(path: str) -> str:
    name = os.path.basename(path).lower()
    if name == "dockerfile":
        return "Dockerfile"
    ext = os.path.splitext(name)[1]
    return _FILE_TYPE_MAP.get(ext, "\u6587\u4ef6")

TOOL_DEFINITION = {
    "type": "function",
    "tags": ["文件", "读取", "查看", "代码", "日志", "配置", "内容"],
    "function": {
        "name": "read_file",
        "description": "读取本地文件内容，支持按行范围读取。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "start_line": {"type": "integer", "description": "起始行号（从1开始）"},
                "end_line": {"type": "integer", "description": "结束行号（含）"},
            },
            "required": ["path"],
        },
    },
}

_OUTPUT_MAX_CHARS = 3000


def execute(args: dict) -> str:
    path = args.get("path", "").strip()
    start_line = args.get("start_line")
    end_line = args.get("end_line")

    if not path:
        return "错误：未提供文件路径。"

    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        return f"错误：文件不存在：{path}"

    if not os.path.isfile(path):
        return f"错误：路径不是文件：{path}"

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return f"错误：无法读取文件：{e}"

    total_lines = len(lines)

    # 行号处理（1-based → 0-based）
    start = max(0, (start_line - 1) if start_line else 0)
    end = min(total_lines, end_line if end_line else total_lines)

    selected = lines[start:end]
    content = "".join(selected)

    header_lines = [
        "[文件读取结果]",
        f"路径：{path}",
        f"类型：{_detect_file_type(path)}",
        f"行范围：{start + 1}-{end} / 共 {total_lines} 行",
    ]
    remaining = total_lines - end
    if remaining > 0:
        header_lines.append(f"剩余未读取：{remaining} 行（可调整 end_line 或继续读取）")
    header_lines.append("---")
    header = "\n".join(header_lines) + "\n"

    if len(content) > _OUTPUT_MAX_CHARS:
        head = content[:_OUTPUT_MAX_CHARS - 200]
        omitted = content[_OUTPUT_MAX_CHARS - 200:].count("\n")
        content = f"{head}\n[...后续 {omitted} 行已截断，请缩小 start_line/end_line 范围...]"

    return header + content
