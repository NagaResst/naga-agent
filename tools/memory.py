import os

# 注意：memory 工具需要访问 Redis，通过全局注入的 session_manager 实现。
# 由于工具 execute() 目前只接收 args dict，Redis 客户端通过模块级变量注入。
_session_manager = None
_memory_manager = None


def set_session_manager(sm):
    global _session_manager
    _session_manager = sm


def set_memory_manager(mm):
    global _memory_manager
    _memory_manager = mm


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": (
            "跨会话持久化记忆存储。用于保存用户偏好、环境信息、常用命令等，"
            "下次对话时可以主动召回。数据持久存储于 Redis，不会随会话结束而丢失。\n"
            "action 说明：\n"
            "  save   - 保存一条记忆，需提供 key 和 value\n"
            "  recall - 读取指定 key 的记忆\n"
            "  list   - 列出所有记忆 key\n"
            "  delete - 删除指定 key 的记忆"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "recall", "list", "delete"],
                    "description": "操作类型",
                },
                "key": {
                    "type": "string",
                    "description": "记忆的键名，例如 'preferred_editor'、'k8s_namespace'",
                },
                "value": {
                    "type": "string",
                    "description": "要保存的记忆内容（仅 save 时需要）",
                },
            },
            "required": ["action"],
        },
    },
}


def execute(args: dict) -> str:
    if _session_manager is None:
        return "错误：记忆工具未初始化（session_manager 未注入）。"

    action = args.get("action", "").strip()
    key = args.get("key", "").strip()
    value = args.get("value", "").strip()

    if action == "save":
        if not key:
            return "错误：save 操作需要提供 key。"
        if not value:
            return "错误：save 操作需要提供 value。"
        if _memory_manager and _memory_manager.available:
            _memory_manager.save_core(key, value)
        elif _session_manager:
            _session_manager.set_memory(key, value)
        return f"记忆已保存：{key} = {value}"

    elif action == "recall":
        if not key:
            return "错误：recall 操作需要提供 key。"
        if _memory_manager and _memory_manager.available:
            # 先精确查 Layer1 Redis 缓存
            from memory.manager import _CORE_CACHE_PREFIX
            redis_val = None
            if _memory_manager._redis:
                try:
                    redis_val = _memory_manager._redis.get(f"{_CORE_CACHE_PREFIX}{key}")
                except Exception:
                    pass
            if redis_val is not None:
                return f"{key} = {redis_val}"
            # 回退到语义检索
            items = _memory_manager.search_episodic(key, top_k=5)
            if items:
                return "语义检索结果：\n" + "\n".join(f"  {m}" for m in items)
            return f"未找到记忆：{key}"
        # 降级：原 Redis KV 逻辑
        if _session_manager is None:
            return "错误：记忆工具未初始化。"
        result = _session_manager.get_memory(key)
        if result is None:
            all_memories = _session_manager.list_memories()
            matched = {k: v for k, v in all_memories.items() if k.startswith(key)}
            if matched:
                lines = [f"  {k} = {v}" for k, v in matched.items()]
                return f"前缀匹配到 {len(matched)} 条记忆：\n" + "\n".join(lines)
            return f"未找到记忆：{key}"
        return f"{key} = {result}"

    elif action == "list":
        if _memory_manager and _memory_manager.available:
            items = _memory_manager.search_core()
            if not items:
                return "暂无任何核心记忆。"
            return f"共 {len(items)} 条核心记忆：\n" + "\n".join(f"  {m}" for m in items)
        if _session_manager is None:
            return "错误：记忆工具未初始化。"
        all_memories = _session_manager.list_memories()
        if not all_memories:
            return "暂无任何记忆。"
        lines = [f"  {k} = {v}" for k, v in all_memories.items()]
        return f"共 {len(all_memories)} 条记忆：\n" + "\n".join(lines)

    elif action == "delete":
        if not key:
            return "错误：delete 操作需要提供 key。"
        if _memory_manager and _memory_manager.available:
            _memory_manager.delete(key)
        elif _session_manager:
            _session_manager.delete_memory(key)
        return f"记忆已删除：{key}"

    else:
        return f"错误：未知 action：{action}，可选值为 save/recall/list/delete。"
