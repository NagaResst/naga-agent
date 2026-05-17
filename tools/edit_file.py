import os

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "对文件做精确字符串替换（old_string 须唯一匹配，含空格换行）。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "old_string": {"type": "string", "description": "被替换的原始文本（须唯一匹配）"},
                "new_string": {"type": "string", "description": "替换后的新内容"},
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
