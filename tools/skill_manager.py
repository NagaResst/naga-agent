_agent = None


def set_agent(agent):
    global _agent
    _agent = agent


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "skill_manager",
        "description": "列出、启用、停用或重新扫描已安装的 Skill；不修改 Skill 文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "enable", "disable", "reload"],
                    "description": "操作类型"
                },
                "name": {
                    "type": "string",
                    "description": "enable/disable：skill 名称"
                }
            },
            "required": ["action"]
        }
    }
}


def execute(args: dict) -> str:
    if _agent is None:
        return "错误：skill_manager 未初始化，agent 引用未注入。"

    action = args.get("action", "")

    if action in ("enable", "disable"):
        name = args.get("name", "").strip()
        if not name:
            return f"错误：{action} 动作需要提供 name 参数。"
        skill = _agent.toggle_skill(name, action == "enable")
        if skill is None:
            return f"错误：未找到 skill：{name}"
        if action == "enable":
            return (
                f"已启用 skill：{skill['name']}\n入口文件：{skill['source_file']}\n"
                "本轮若需使用，请先用 read_file 读取入口文件；下一条用户消息起自动匹配。"
            )
        return f"已停用 skill：{skill['name']}"

    if action == "reload":
        _agent.reload_skills()
    if action in ("list", "reload"):
        skills_info = [
            f"  {'[激活]' if s['enabled'] else '[停用]'} {s['name']}"
            + (f" — {s['description']}" if s.get("description") else "")
            for s in _agent._skills
        ]
        if not skills_info:
            return "skills/ 目录为空。"
        return "当前 skill 列表：\n" + "\n".join(skills_info)

    return f"错误：未知 action：{action}，支持 list / enable / disable / reload。"
