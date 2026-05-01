import os

_memory_manager = None


def set_memory_manager(mm):
    global _memory_manager
    _memory_manager = mm


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": (
            "跨会话持久化记忆存储。拥有两个记忆来源：\n"
            "① 对话记忆（自动从对话中提取偏好、事实）\n"
            "② 手动知识库（用户主动导入的文档，按标题拆分存储）\n"
            "数据持久存储，不会随会话结束而丢失。\n"
            "action 说明：\n"
            "  save          - 保存一条核心记忆，需提供 key 和 value\n"
            "  recall        - 读取指定 key 的记忆\n"
            "  list          - 列出所有核心记忆\n"
            "  delete        - 删除指定 key 的核心记忆\n"
            "  add_document  - 导入一个文档到知识库，自动按标题拆分为多条记忆并存储\n"
            "  forget        - 从知识库中删除指定记忆（按标题或来源匹配）\n"
            "  search        - 语义搜索记忆（可指定搜索范围：core/manual/all）"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "recall", "list", "delete", "add_document", "forget", "search"],
                    "description": "操作类型",
                },
                "key": {
                    "type": "string",
                    "description": "记忆的键名（save/recall/delete 时需要）",
                },
                "value": {
                    "type": "string",
                    "description": "要保存的记忆内容（仅 save 时需要）",
                },
                "path": {
                    "type": "string",
                    "description": "文档文件路径（add_document 时需要）",
                },
                "content": {
                    "type": "string",
                    "description": "文档完整内容（add_document 时需要，与 path 二选一）",
                },
                "source": {
                    "type": "string",
                    "description": "文档来源标识，如文件名或路径（add_document 时需要）",
                },
                "title_or_source": {
                    "type": "string",
                    "description": "要删除的记忆标题或来源（forget 时需要）",
                },
                "query": {
                    "type": "string",
                    "description": "搜索关键词（search 时需要）",
                },
                "scope": {
                    "type": "string",
                    "enum": ["core", "manual", "all"],
                    "description": "搜索范围：core=核心记忆, manual=手动知识库, all=全部（默认 manual）",
                },
            },
            "required": ["action"],
        },
    },
}


def execute(args: dict) -> str:
    if _memory_manager is None:
        return "错误：记忆工具未初始化（memory_manager 未注入）。"

    action = args.get("action", "").strip()

    if action == "save":
        key = args.get("key", "").strip()
        value = args.get("value", "").strip()
        if not key:
            return "错误：save 操作需要提供 key。"
        if not value:
            return "错误：save 操作需要提供 value。"
        _memory_manager.save_core(key, value)
        return f"记忆已保存：{key} = {value}"

    elif action == "recall":
        key = args.get("key", "").strip()
        if not key:
            return "错误：recall 操作需要提供 key。"
        cache_val = _memory_manager._core_cache.get(key)
        if cache_val is not None:
            return f"{key} = {cache_val}"
        items = _memory_manager.search_episodic(key, top_k=5)
        if items:
            return "语义检索结果：\n" + "\n".join(f"  {m}" for m in items)
        return f"未找到记忆：{key}"

    elif action == "list":
        items = _memory_manager.search_core()
        if not items:
            return "暂无任何核心记忆。"
        return f"共 {len(items)} 条核心记忆：\n" + "\n".join(f"  {m}" for m in items)

    elif action == "delete":
        key = args.get("key", "").strip()
        if not key:
            return "错误：delete 操作需要提供 key。"
        _memory_manager.delete(key)
        return f"记忆已删除：{key}"

    elif action == "add_document":
        from tools.manual_memory import add_document as _add_doc

        content = args.get("content", "").strip()
        path = args.get("path", "").strip()
        source = args.get("source", "").strip()

        if not content and path:
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            if not os.path.exists(path):
                return f"错误：文件不存在：{path}"
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError as e:
                return f"错误：无法读取文件：{e}"
            if not source:
                source = path

        if not content:
            return "错误：add_document 需要提供 content 或 path。"
        if not source:
            return "错误：add_document 需要提供 source（文档来源标识）。"

        try:
            return _add_doc(content, source)
        except Exception as e:
            return f"导入文档失败：{e}"

    elif action == "forget":
        from tools.manual_memory import forget as _forget

        title_or_source = args.get("title_or_source", "").strip()
        if not title_or_source:
            return "错误：forget 操作需要提供 title_or_source。"
        try:
            return _forget(title_or_source)
        except Exception as e:
            return f"删除记忆失败：{e}"

    elif action == "search":
        from tools.manual_memory import search_manual as _search_manual

        query = args.get("query", "").strip()
        scope = args.get("scope", "manual").strip()

        if not query:
            return "错误：search 操作需要提供 query。"

        parts = []

        # 搜索手动知识库
        if scope in ("manual", "all"):
            try:
                manual_result = _search_manual(query, top_k=5)
                if "未找到" not in manual_result:
                    parts.append("【手动知识库】\n" + manual_result)
            except Exception as e:
                parts.append(f"【手动知识库】搜索失败：{e}")

        # 搜索核心记忆
        if scope in ("core", "all"):
            core_items = _memory_manager.search_core()
            if core_items:
                parts.append("【核心记忆】\n" + "\n".join(f"  {m}" for m in core_items))
            # 搜索对话情节记忆
            if _memory_manager.available:
                episodic_items = _memory_manager.search_episodic(query, top_k=3)
                if episodic_items:
                    parts.append("【对话记忆】\n" + "\n".join(f"  {m}" for m in episodic_items))

        return "\n\n".join(parts) if parts else f"未找到与 '{query}' 相关的记忆。"

    else:
        return f"错误：未知 action：{action}，可选值为 save/recall/list/delete/add_document/forget/search。"
