import os
import re


_SKILL_TERM_STOPWORDS = {
    "assistant", "skill", "skills", "agent", "mode", "tool", "tools",
    "帮助", "助手", "模式", "工具", "能力", "技能", "自动", "执行", "生成",
    "分析", "报告", "完整", "数据", "信息", "流程", "步骤", "要求", "脚本",
    "搜索", "联网", "处理", "使用", "支持", "相关", "用户", "任务",
}


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


def _normalize_skill_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r'[`*_#>\[\]\(\)"\']+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _coerce_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _dedupe_keep_order(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _extract_match_terms(text: str) -> list[str]:
    normalized = _normalize_skill_text(text)
    terms = []

    for token in re.findall(r'[a-z0-9][a-z0-9+._-]{1,}', normalized):
        if len(token) >= 3 and token not in _SKILL_TERM_STOPWORDS:
            terms.append(token)

    for chunk in re.findall(r'[\u4e00-\u9fff]{2,}', normalized):
        if len(chunk) <= 8 and chunk not in _SKILL_TERM_STOPWORDS:
            terms.append(chunk)
        for size in (2, 3, 4):
            if len(chunk) < size:
                continue
            for index in range(len(chunk) - size + 1):
                term = chunk[index:index + size]
                if term in _SKILL_TERM_STOPWORDS:
                    continue
                terms.append(term)

    return _dedupe_keep_order(terms)


def _collect_skill_resource_files(skill_dir: str, primary_file: str) -> list[str]:
    if not skill_dir or not os.path.isdir(skill_dir):
        return []

    resource_files = []
    for root, dirs, files in os.walk(skill_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for filename in sorted(files):
            if filename.startswith("."):
                continue
            filepath = os.path.join(root, filename)
            if os.path.normpath(filepath) == os.path.normpath(primary_file):
                continue
            rel_path = os.path.relpath(filepath, skill_dir)
            resource_files.append(rel_path)

    return resource_files


def discover_skills(skills_dir: str, enabled_names: list) -> list:
    """扫描 skills/ 目录下所有 .md 文件及子目录 skill，返回 skill 列表。

    支持两种 skill 形式：
      - 单文件：skills/my-skill.md
      - 目录：skills/my-skill/SKILL.md（含 scripts/、references/ 等子资源）

    enabled_names：来自 config["skills"]["enabled"]。
      - 若非空列表：名单内的 skill 强制 enabled=True，其余 False
      - 若空列表：尊重每个文件 frontmatter 中的 enabled 字段（默认 False）

        返回 list[dict]，每项：
            { name, description, enabled, prompt, source_file, keywords, examples, match_phrases, match_terms }
    """
    if not os.path.isdir(skills_dir):
        return []

    skills = []
    use_override = bool(enabled_names)

    for entry in sorted(os.listdir(skills_dir)):
        entry_path = os.path.join(skills_dir, entry)

        skill_dir = None

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
            skill_dir = entry_path
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
        keywords = _coerce_string_list(meta.get("keywords") or meta.get("tags"))
        examples = _coerce_string_list(meta.get("examples") or meta.get("queries") or meta.get("triggers"))
        match_phrases = _dedupe_keep_order([
            _normalize_skill_text(default_name),
            _normalize_skill_text(name),
            _normalize_skill_text(description),
            *[_normalize_skill_text(item) for item in keywords],
            *[_normalize_skill_text(item) for item in examples],
        ])

        match_terms = []
        for phrase in match_phrases:
            match_terms.extend(_extract_match_terms(phrase))
        match_terms = _dedupe_keep_order(match_terms)

        resource_files = _collect_skill_resource_files(skill_dir, filepath) if skill_dir else []

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
            "skill_dir": skill_dir,
            "resource_files": resource_files,
            "keywords": keywords,
            "examples": examples,
            "match_phrases": [p for p in match_phrases if p],
            "match_terms": match_terms,
        })

    return skills
