import os

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "generate_script",
        "description": (
            "在当前工作目录下的 generated_scripts 子目录中生成一个脚本文件。"
            "适用于需要保存代码片段、Shell 脚本或其他可执行文件的场景。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "脚本文件名，例如 deploy.sh 或 cleanup.py",
                },
                "content": {
                    "type": "string",
                    "description": "脚本文件的完整内容",
                },
                "language": {
                    "type": "string",
                    "description": "脚本语言，例如 bash、sh、python、ruby 等",
                },
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
