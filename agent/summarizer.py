"""上下文预算管理器（Context Budget Manager）。

职责：纯机械 token 管理，零 LLM 调用，零记忆干预。
记忆完全由 mem0 负责，本模块只做 token 数量控制。

压缩管线顺序：
  Step 1: slim_tool_outputs   — 工具结果瘦身（最高收益）
  Step 2: priority_trim       — 分级保留（精准丢弃低价值消息）
  Step 3: dual_track_compress — 双轨压缩（兜底）
"""


def should_compress(estimated_tokens: int, context_window: int, threshold: float = 0.60) -> bool:
    """当 token 估算值超过上下文窗口的 threshold 比例时触发压缩。"""
    return estimated_tokens >= int(context_window * threshold)


# 向后兼容别名
def should_summarize(estimated_tokens: int, context_token_limit: int) -> bool:
    return should_compress(estimated_tokens, context_token_limit, threshold=0.75)


def slim_tool_outputs(messages: list, keep_latest_turns: int = 3, max_chars: int = 300) -> list:
    """工具结果瘦身：对非最新 N 轮的工具相关消息截断内容。

    - role=tool 消息：content 截断为前 max_chars 字符
    - role=assistant 且含 tool_calls 的消息：arguments 截断
    不修改原列表，返回新列表。
    """
    if not messages:
        return messages

    # 找出最新 keep_latest_turns 轮的边界索引
    turn_boundaries = []
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            turn_boundaries.append(i)
    cutoff_idx = turn_boundaries[-keep_latest_turns] if len(turn_boundaries) >= keep_latest_turns else 0

    result = []
    for i, msg in enumerate(messages):
        if i >= cutoff_idx:
            result.append(msg)
            continue

        msg = dict(msg)

        if msg.get("role") == "tool":
            content = msg.get("content") or ""
            if len(content) > max_chars:
                msg["content"] = content[:max_chars] + f"\n[...工具输出已折叠，共 {len(content)} 字符]"

        elif msg.get("role") == "assistant" and msg.get("tool_calls"):
            new_tcs = []
            for tc in msg["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                args = fn.get("arguments", "")
                if len(args) > max_chars:
                    fn["arguments"] = args[:max_chars] + "...[已截断]"
                tc["function"] = fn
                new_tcs.append(tc)
            msg["tool_calls"] = new_tcs

        result.append(msg)

    return result


def _score_message(msg: dict) -> int:
    """消息重要性评分（纯规则）。分数越高越重要，越不应该被丢弃。

    P1=4: role=tool 或含 tool_calls 的 assistant 消息
    P2=3: user 消息且 content > 100 字
    P3=2: assistant 消息且 content > 200 字
    P4=1: 其余
    """
    role = msg.get("role", "")
    content = msg.get("content") or ""

    if role == "tool":
        return 4
    if role == "assistant" and msg.get("tool_calls"):
        return 4
    if role == "user" and len(content) > 100:
        return 3
    if role == "assistant" and len(content) > 200:
        return 2
    return 1


def priority_trim(messages: list, token_tracker, target_tokens: int) -> list:
    """分级保留：从最旧的低优先级消息开始丢弃，直到 token 估算 ≤ target_tokens。

    P1（分数=4）永不丢弃。
    """
    result = list(messages)

    for priority_threshold in (1, 2, 3):
        if token_tracker.estimate(result) <= target_tokens:
            break
        trimmed = []
        for msg in result:
            if _score_message(msg) <= priority_threshold:
                continue  # 丢弃
            trimmed.append(msg)
        result = trimmed

    return result


def dual_track_compress(messages: list, max_chars_mid: int = 100) -> list:
    """双轨压缩：三等分消息，对旧消息做不同程度的内容折叠。

    - 最旧 1/3：content 清空为 "[已折叠]"，保留 role/tool_calls 字段
    - 中间 1/3：content 截断至 max_chars_mid 字
    - 最新 1/3：原样保留
    """
    n = len(messages)
    if n < 6:
        return messages

    third = n // 3
    old_end = third
    mid_end = 2 * third

    result = []
    for i, msg in enumerate(messages):
        msg = dict(msg)
        if i < old_end:
            if msg.get("role") != "system":
                msg["content"] = "[已折叠]"
                if msg.get("tool_calls"):
                    new_tcs = []
                    for tc in msg["tool_calls"]:
                        tc = dict(tc)
                        fn = dict(tc.get("function", {}))
                        fn["arguments"] = "{}"
                        tc["function"] = fn
                        new_tcs.append(tc)
                    msg["tool_calls"] = new_tcs
        elif i < mid_end:
            content = msg.get("content") or ""
            if len(content) > max_chars_mid:
                msg["content"] = content[:max_chars_mid] + "...[已截断]"
        result.append(msg)

    return result


def compress_pipeline(
    messages: list,
    token_tracker,
    context_window: int,
    threshold: float = 0.60,
    tool_max_chars: int = 300,
) -> list:
    """完整压缩管线：按顺序执行三步压缩，每步后检查是否已满足目标。

    目标：token 数 < context_window * threshold * 0.90（留 10% 余量）
    返回压缩后的消息列表（内存中生效，不写入 Redis）。
    """
    target = int(context_window * threshold * 0.90)

    # Step 1: 工具结果瘦身
    result = slim_tool_outputs(messages, keep_latest_turns=3, max_chars=tool_max_chars)
    if token_tracker.estimate(result) <= target:
        return result

    # Step 2: 分级保留
    result = priority_trim(result, token_tracker, target)
    if token_tracker.estimate(result) <= target:
        return result

    # Step 3: 双轨压缩（兜底）
    result = dual_track_compress(result)
    return result
