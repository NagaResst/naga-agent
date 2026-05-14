"""上下文预算管理器（Context Budget Manager）。

职责：纯机械 token 管理，零 LLM 调用，零记忆干预。
记忆完全由 mem0 负责，本模块只做 token 数量控制。

Turn 定义：
  一个 turn = 上一条 user 消息之后的所有 tool/assistant 消息 + 当前 user 消息。
  即以 user 消息作为 turn 的结尾边界。

  示例：
    [assistant / tool / tool / assistant / user]  ← turn 1（完整）
    [assistant / tool / assistant / user]          ← turn 2（完整，必须保留）
    [assistant / tool / ...]                       ← in-progress turn（无 user 结尾，必须全部保留）

滑动窗口三档：
  token 使用率 < lo_threshold      → keep_turns = keep_lo
  lo_threshold ≤ 使用率 < threshold → keep_turns = keep_mid
  压缩管线内                        → keep_turns = keep_hi

压缩管线顺序：
  Step 1: slim_tool_outputs          — 工具输出文本截断（最高收益）
  Step 2: sliding_window_trim(hi)    — 收紧滑动窗口到最小档
  Step 3: sliding_window_drop_oldest — 整轮删除兜底（仍超则继续删最旧 turn）
"""


def should_compress(estimated_tokens: int, context_window: int, threshold: float = 0.60) -> bool:
    """当 token 估算值超过上下文窗口的 threshold 比例时触发压缩。"""
    return estimated_tokens >= int(context_window * threshold)


# 向后兼容别名
def should_summarize(estimated_tokens: int, context_token_limit: int) -> bool:
    return should_compress(estimated_tokens, context_token_limit, threshold=0.75)


def _split_turns(messages: list) -> tuple[list[list], list]:
    """按 user 消息结尾拆分完整 turn，返回 (turns, inprogress)。

    turns:      list of list，每个子列表是一个完整 turn（最后一条是 user）
    inprogress: 尾部尚无 user 结尾的消息列表（可能为空）

    system 消息独立剥离，不参与 turn 分组。
    """
    # 剥离 system 消息（由调用方单独处理）
    non_system = [m for m in messages if m.get("role") != "system"]

    turns: list[list] = []
    current: list = []
    for msg in non_system:
        current.append(msg)
        if msg.get("role") == "user":
            turns.append(current)
            current = []
    inprogress = current  # 剩余无 user 结尾的部分

    return turns, inprogress


def sliding_window_trim(messages: list, keep_turns: int) -> list:
    """滑动窗口裁剪：保留最近 keep_turns 个完整 turn + in-progress turn。

    - system 消息始终保留
    - in-progress turn（无 user 结尾）始终全部保留
    - 完整 turn 只保留最近 keep_turns 个，多余的整轮从头删除
    不修改原列表，返回新列表。
    """
    if not messages:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    turns, inprogress = _split_turns(messages)

    # 只保留最近 keep_turns 个完整 turn
    kept_turns = turns[-keep_turns:] if len(turns) > keep_turns else turns

    result = system_msgs[:]
    for turn in kept_turns:
        result.extend(turn)
    result.extend(inprogress)
    return result


def slim_tool_outputs(messages: list, keep_latest_turns: int = 2, max_chars: int = 300) -> list:
    """工具输出文本截断：对非最新 keep_latest_turns 个完整 turn 里的工具消息截断内容。

    - role=tool 消息：content 截断为前 max_chars 字符
    - role=assistant 且含 tool_calls：arguments 截断
    最新 keep_latest_turns 个完整 turn 和 in-progress turn 原样保留。
    不修改原列表，返回新列表。
    """
    if not messages:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    turns, inprogress = _split_turns(messages)

    # 最新 keep_latest_turns 个完整 turn 不截断
    protected_start = max(0, len(turns) - keep_latest_turns)
    old_turns = turns[:protected_start]
    new_turns = turns[protected_start:]

    def _slim_turn(turn: list) -> list:
        result = []
        for msg in turn:
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

    result = system_msgs[:]
    for turn in old_turns:
        result.extend(_slim_turn(turn))
    for turn in new_turns:
        result.extend(turn)
    result.extend(inprogress)
    return result


def sliding_window_drop_oldest(messages: list, token_tracker, target_tokens: int) -> list:
    """兜底：逐轮删除最旧的完整 turn，直到 token 估算 ≤ target_tokens 或无可删。

    turn 2（倒数第二个完整 turn）及更新的 turn 不删除。
    system 消息和 in-progress turn 永远保留。
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    turns, inprogress = _split_turns(messages)

    # 至少保留最后 2 个完整 turn（turn2 必保留）
    min_keep = 2

    while len(turns) > min_keep:
        if token_tracker.estimate(messages) <= target_tokens:
            break
        turns = turns[1:]  # 删掉最旧的一个 turn
        messages = system_msgs[:]
        for t in turns:
            messages.extend(t)
        messages.extend(inprogress)

    return messages


def compress_pipeline(
    messages: list,
    token_tracker,
    context_window: int,
    threshold: float = 0.60,
    tool_max_chars: int = 300,
    keep_hi: int = 3,
) -> list:
    """完整压缩管线：按顺序执行三步压缩，每步后检查是否已满足目标。

    目标：token 数 < context_window * threshold * 0.85（留 15% 余量）
    返回压缩后的消息列表（内存中生效，不写入持久化存储）。
    """
    target = int(context_window * threshold * 0.85)

    # Step 1: 工具输出截断（收益最高，不丢消息）
    result = slim_tool_outputs(messages, keep_latest_turns=2, max_chars=tool_max_chars)
    if token_tracker.estimate(result) <= target:
        return result

    # Step 2: 收紧滑动窗口到最小档（keep_hi）
    result = sliding_window_trim(result, keep_turns=keep_hi)
    if token_tracker.estimate(result) <= target:
        return result

    # Step 3: 兜底逐轮删除（保留至少 2 个完整 turn）
    result = sliding_window_drop_oldest(result, token_tracker, target)
    return result


def pick_keep_turns(
    usage_ratio: float,
    lo_threshold: float = 0.30,
    compress_threshold: float = 0.95,
    keep_lo: int = 10,
    keep_mid: int = 6,
    keep_hi: int = 3,
) -> int:
    """根据当前 token 使用率返回对应档位的滑动窗口轮数。

    usage_ratio = estimated_tokens / context_window
    """
    if usage_ratio < lo_threshold:
        return keep_lo
    if usage_ratio < compress_threshold:
        return keep_mid
    return keep_hi
