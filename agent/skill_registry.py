import os
import re


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 Markdown 文件的 YAML frontmatter。

    返回 (meta_dict, body_text)。
    若无 frontmatter，返回 ({}, 全文)。
    仅解析简单 key: value 格式，不引入外部依赖。
    """
    pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    match = pattern.match(text)
    if not match:
        return {}, text.strip()

    meta = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            # 解析布尔值
            if value.lower() == "true":
                meta[key] = True
            elif value.lower() == "false":
                meta[key] = False
            # 解析 TOML 风格列表：["a", "b"] 或 [a, b]
            elif value.startswith("[") and value.endswith("]"):
                inner = value[1:-1]
                items = [i.strip().strip('"').strip("'") for i in inner.split(",") if i.strip()]
                meta[key] = items
            else:
                meta[key] = value.strip('"').strip("'")

    body = text[match.end():].strip()
    return meta, body


def discover_skills(skills_dir: str, enabled_names: list) -> list:
    """扫描 skills/ 目录下所有 .md 文件及子目录 skill，返回 skill 列表。

    支持两种 skill 形式：
      - 单文件：skills/my-skill.md
      - 目录：skills/my-skill/SKILL.md（含 scripts/、references/ 等子资源）

    enabled_names：来自 config["skills"]["enabled"]。
      - 若非空列表：名单内的 skill 强制 enabled=True，其余 False
      - 若空列表：尊重每个文件 frontmatter 中的 enabled 字段（默认 False）

    返回 list[dict]，每项：
      { name, description, enabled, prompt, source_file }
    """
    if not os.path.isdir(skills_dir):
        return []

    skills = []
    use_override = bool(enabled_names)

    for entry in sorted(os.listdir(skills_dir)):
        entry_path = os.path.join(skills_dir, entry)

        # 单文件 skill
        if entry.endswith(".md") and os.path.isfile(entry_path):
            filepath = entry_path
            default_name = entry[:-3]
        # 目录 skill：目录内必须有 SKILL.md
        elif os.path.isdir(entry_path):
            skill_md = os.path.join(entry_path, "SKILL.md")
            if not os.path.isfile(skill_md):
                continue
            filepath = skill_md
            default_name = entry
        else:
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            print(f"警告：无法读取 skill 文件 {filepath}：{e}")
            continue

        meta, body = _parse_frontmatter(content)

        # name 优先用 frontmatter，回退到文件名/目录名
        name = meta.get("name", default_name)
        description = meta.get("description", "")

        if use_override:
            enabled = name in enabled_names
        else:
            enabled = meta.get("enabled", False)

        skills.append({
            "name": name,
            "description": description,
            "enabled": enabled,
            "prompt": body,
            "source_file": filepath,
        })

    return skills
