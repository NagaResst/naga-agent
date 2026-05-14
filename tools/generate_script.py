import os

TOOL_DEFINITION = {
    "type": "function",
    "tags": ["脚本", "生成", "代码", "文件", "shell", "python", "写"],
    "function": {
        "name": "generate_script",
        "description": "在 generated_scripts/ 目录生成脚本文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "文件名，如 deploy.sh"},
                "content": {"type": "string", "description": "文件完整内容"},
                "language": {"type": "string", "description": "脚本语言，如 bash/python"},
            },
            "required": ["filename", "content", "language"],
        },
    },
}


def execute(args: dict) -> str:
    filename = args.get("filename", "script.sh")
    content = args.get("content", "")
    language = args.get("language", "").lower()

    scripts_dir = os.path.join(os.getcwd(), "generated_scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    file_path = os.path.join(scripts_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    if language in ("bash", "sh"):
        os.chmod(file_path, 0o755)

    return f"脚本已生成：{file_path}"
