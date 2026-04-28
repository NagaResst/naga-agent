import os

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "对本地文件进行精确的字符串替换编辑。"
            "提供 old_string（必须与文件中某段文本完全一致）和 new_string（替换后的内容）。"
            "比通过 execute_command 使用 sed 更安全可控，且无需命令确认。"
            "若 old_string 在文件中不存在或匹配多处，操作会被拒绝并返回明确错误。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件的绝对路径或相对于当前工作目录的路径",
                },
                "old_string": {
                    "type": "string",
                    "description": "要替换的原始文本，必须与文件中某段内容完全一致（含空格和换行）",
                },
                "new_string": {
                    "type": "string",
                    "description": "替换后的新内容",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
}


def execute(args: dict) -> str:
    path = args.get("path", "").strip()
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")

    if not path:
        return "错误：未提供文件路径。"

    if not old_string:
        return "错误：old_string 不能为空。"

    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        return f"错误：文件不存在：{path}"

    if not os.path.isfile(path):
        return f"错误：路径不是文件：{path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return f"错误：无法读取文件：{e}"

    count = content.count(old_string)
    if count == 0:
        return (
            f"错误：old_string 在文件中未找到，请检查内容是否与文件完全一致（含空格/换行）。\n"
            f"文件：{path}"
        )
    if count > 1:
        return (
            f"错误：old_string 在文件中匹配了 {count} 处，操作已拒绝（需唯一匹配）。\n"
            f"请在 old_string 中增加更多上下文以唯一定位目标位置。"
        )

    new_content = content.replace(old_string, new_string, 1)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except OSError as e:
        return f"错误：写入文件失败：{e}"

    return f"文件已成功编辑：{path}"
