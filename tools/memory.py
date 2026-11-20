"""记忆工具。第一参数 layer（core/episodic），第二参数 action（save/recall/delete/list）。"""

_memory_manager = None


def set_memory_manager(mm):
    global _memory_manager
    _memory_manager = mm


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": "持久化记忆系统。layer=core（核心偏好，全量注入）或 layer=episodic（情景记忆，可检索）；检索方式取决于配置的后端。",
        "parameters": {
            "type": "object",
            "properties": {
                "layer": {
                    "type": "string",
                    "enum": ["core", "episodic"],
                    "description": "操作哪一层",
                },
                "action": {
                    "type": "string",
                    "enum": ["save", "recall", "delete", "list"],
                    "description": "操作类型",
                },
                "content": {
                    "type": "string",
                    "description": "自然语言内容（save 必填）",
                },
                "query": {
                    "type": "string",
                    "description": "自然语言查询词（recall 必填）",
                },
                "memory_id": {
                    "type": "string",
                    "description": "记忆 ID（delete 必填）",
                },
            },
            "required": ["layer", "action"],
        },
    },
}


def _err(msg: str) -> str:
    return f"[memory 参数错误] {msg}"


def execute(args: dict) -> str:
    if _memory_manager is None:
        return "错误：记忆系统未初始化"

    layer = (args.get("layer") or "").strip()
    action = (args.get("action") or "").strip()

    # ── 参数校验 ────────────────────────────────────────────

    if layer not in ("core", "episodic"):
        return _err('layer 必须是 "core" 或 "episodic"')

    if action not in ("save", "recall", "delete", "list"):
        return _err('action 必须是 save / recall / delete / list')

    if action == "save":
        content = (args.get("content") or "").strip()
        if not content:
            return _err("save 需要提供 content")

    if action == "recall":
        query = (args.get("query") or "").strip()
        if not query:
            return _err("recall 需要提供 query")

    if action == "delete":
        memory_id = (args.get("memory_id") or "").strip()
        if not memory_id:
            return _err("delete 需要提供 memory_id")

    # ── core 层 ────────────────────────────────────────────

    if layer == "core":
        if action == "save":
            return _memory_manager.save_core(content)

        elif action == "recall":
            return _err("核心记忆全量注入，无需 recall，请使用 list 查看")

        elif action == "delete":
            return _memory_manager.delete_core(memory_id)

        elif action == "list":
            items = _memory_manager.list_core()
            if not items:
                return "暂无核心记忆"
            lines = ["共 {} 条核心记忆：".format(len(items))]
            for it in items:
                lines.append(f"  [{it['id']}] {it['content']}")
            return "\n".join(lines)

    # ── episodic 层 ────────────────────────────────────────

    if layer == "episodic":
        if action == "save":
            return _memory_manager.save_episodic(content)

        elif action == "recall":
            items = _memory_manager.recall_episodic(query, top_k=10)
            if not items:
                return "未找到相关情景记忆"
            lines = ["搜索 \"{}\" 返回 {} 条结果：".format(query, len(items))]
            for it in items:
                lines.append("  [{id}] ({score:.2f}) {content}".format(**it))
            return "\n".join(lines)

        elif action == "delete":
            return _memory_manager.delete_episodic(memory_id)

        elif action == "list":
            items = _memory_manager.list_episodic(top_k=20)
            if not items:
                return "暂无情景记忆"
            lines = ["共 {} 条近期情景记忆：".format(len(items))]
            for it in items:
                lines.append("  [{id}] {content}".format(**it))
            return "\n".join(lines)

    return _err("未知操作")