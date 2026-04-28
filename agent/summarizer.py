_SUMMARIZE_SYSTEM = (
    "你是一个对话历史压缩助手。将以下对话历史压缩成一段简洁的摘要，"
    "保留关键事实、用户偏好和重要结论，忽略闲聊和重复内容。"
    "只输出摘要正文，不要添加任何标题或说明。"
)


def should_summarize(estimated_tokens: int, context_token_limit: int) -> bool:
    """当估算 token 数超过上下文窗口的 75% 时触发摘要压缩。"""
    return estimated_tokens >= int(context_token_limit * 0.75)


def summarize_old_messages(messages_to_compress: list, client, classifier_model: str) -> str:
    """将一批历史消息压缩成摘要字符串，使用轻量模型（低消费）节省成本。"""
    if not messages_to_compress:
        return ""

    dialogue_text = []
    for m in messages_to_compress:
        role = m.get("role", "unknown")
        content = m.get("content", "") or ""
        if role in ("user", "assistant") and content:
            label = "用户" if role == "user" else "助手"
            dialogue_text.append(f"{label}：{content[:300]}")  # 每条最多 300 字

    if not dialogue_text:
        return ""

    combined = "\n".join(dialogue_text)
    try:
        resp = client.chat.completions.create(
            model=classifier_model,
            messages=[
                {"role": "system", "content": _SUMMARIZE_SYSTEM},
                {"role": "user", "content": combined},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def compress_history(
    history: list,
    max_history: int,
    client,
    classifier_model: str,
    existing_summary: str = "",
) -> tuple[list, str]:
    """对超出阈值的历史执行摘要压缩。

    返回 (保留的消息列表, 更新后的摘要字符串)。
    保留最近 max_history // 2 条原始消息，其余压缩进摘要。
    """
    keep_count = max_history // 2
    to_compress = history[:-keep_count] if len(history) > keep_count else []
    kept = history[-keep_count:] if len(history) > keep_count else history

    if not to_compress:
        return history, existing_summary

    new_summary = summarize_old_messages(to_compress, client, classifier_model)

    if existing_summary and new_summary:
        merged = existing_summary + "\n" + new_summary
    else:
        merged = new_summary or existing_summary

    return kept, merged
