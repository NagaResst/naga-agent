import os
import shutil

from agent.skill_registry import _parse_frontmatter

_agent = None


def set_agent(agent):
    global _agent
    _agent = agent


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "skill_manager",
        "description": (
            "管理 Agent 的 Skill（专项能力提示词）。支持从外部路径加载 skill 文件或目录、移除已有 skill、重新扫描 skills/ 目录。\n"
            "action 说明：\n"
            "  load   - 从指定路径复制 .md 文件或 skill 目录（含 SKILL.md）到 skills/ 目录，导入后自动激活，需提供 source_path\n"
            "  remove - 从 skills/ 目录删除指定名称的 skill 并立即生效，需提供 name\n"
            "  reload - 重新扫描 skills/ 目录，刷新 skill 列表"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["load", "remove", "reload"],
                    "description": "操作类型",
                },
                "source_path": {
                    "type": "string",
                    "description": "load 动作：外部 .md 文件的绝对或相对路径（支持 ~ 展开）",
                },
                "name": {
                    "type": "string",
                    "description": "remove 动作：要删除的 skill 名称（对应 skills/ 目录下的文件名，不含 .md）",
                },
            },
            "required": ["action"],
        },
    },
}


def execute(args: dict) -> str:
    if _agent is None:
        return "错误：skill_manager 未初始化，agent 引用未注入。"

    action = args.get("action", "")

    if action == "load":
        source_path = args.get("source_path", "").strip()
        if not source_path:
            return "错误：load 动作需要提供 source_path 参数。"

        source_path = os.path.expanduser(source_path)
        if not os.path.isabs(source_path):
            source_path = os.path.abspath(source_path)

        is_dir = os.path.isdir(source_path)
        is_file = os.path.isfile(source_path)

        if not is_dir and not is_file:
            return f"错误：路径不存在：{source_path}"
        if is_file and not source_path.endswith(".md"):
            return f"错误：只支持 .md 格式的 skill 文件或 skill 目录，当前文件：{source_path}"

        os.makedirs(_agent._skills_dir, exist_ok=True)

        if is_dir:
            skill_md = os.path.join(source_path, "SKILL.md")
            if not os.path.isfile(skill_md):
                return f"错误：目录中未找到 SKILL.md：{source_path}"
            dirname = os.path.basename(source_path.rstrip("/"))
            dest_path = os.path.join(_agent._skills_dir, dirname)
            overwritten = os.path.isdir(dest_path)
            if overwritten:
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)
            # 从 SKILL.md frontmatter 读取 skill name
            try:
                with open(skill_md, "r", encoding="utf-8") as f:
                    content = f.read()
                meta, _ = _parse_frontmatter(content)
                skill_name = meta.get("name", dirname)
            except Exception:
                skill_name = dirname
        else:
            filename = os.path.basename(source_path)
            dest_path = os.path.join(_agent._skills_dir, filename)
            overwritten = os.path.isfile(dest_path)
            shutil.copy2(source_path, dest_path)
            try:
                with open(dest_path, "r", encoding="utf-8") as f:
                    content = f.read()
                meta, _ = _parse_frontmatter(content)
                skill_name = meta.get("name", filename[:-3])
            except Exception:
                skill_name = filename[:-3]

        # 自动激活：加入 enabled_names 后 reload
        if skill_name not in _agent._skills_enabled_names:
            _agent._skills_enabled_names.append(skill_name)
        _agent.reload_skills()

        active = [s["name"] for s in _agent._skills if s["enabled"]]
        action_word = "已覆盖更新" if overwritten else "已加载"
        return (
            f"{action_word} skill：{skill_name}（来源：{source_path}）\n"
            f"当前已加载 {len(_agent._skills)} 个 skill，激活：{active or '无'}"
        )

    elif action == "remove":
        name = args.get("name", "").strip()
        if not name:
            return "错误：remove 动作需要提供 name 参数。"

        target_file = os.path.join(_agent._skills_dir, f"{name}.md")
        target_dir = os.path.join(_agent._skills_dir, name)

        if os.path.isfile(target_file):
            os.remove(target_file)
        elif os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        else:
            existing_files = [f[:-3] for f in os.listdir(_agent._skills_dir) if f.endswith(".md")]
            existing_dirs = [
                d for d in os.listdir(_agent._skills_dir)
                if os.path.isdir(os.path.join(_agent._skills_dir, d))
                and os.path.isfile(os.path.join(_agent._skills_dir, d, "SKILL.md"))
            ]
            existing = sorted(existing_files + existing_dirs)
            return f"错误：未找到 skill：{name}\n当前可用的 skill：{existing or '空'}"

        # 从 enabled_names 中移除
        if name in _agent._skills_enabled_names:
            _agent._skills_enabled_names.remove(name)
        _agent.reload_skills()

        active = [s["name"] for s in _agent._skills if s["enabled"]]
        return (
            f"已删除 skill：{name}\n"
            f"当前已加载 {len(_agent._skills)} 个 skill，激活：{active or '无'}"
        )

    elif action == "reload":
        _agent.reload_skills()
        skills_info = [
            f"  {'[激活]' if s['enabled'] else '[停用]'} {s['name']}"
            + (f" — {s['description']}" if s.get("description") else "")
            for s in _agent._skills
        ]
        if not skills_info:
            return "重新扫描完成，skills/ 目录为空。"
        return "重新扫描完成，当前 skill 列表：\n" + "\n".join(skills_info)

    else:
        return f"错误：未知 action：{action}，支持 load / remove / reload。"
