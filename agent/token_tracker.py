import json
from datetime import datetime


def _load_tiktoken():
    try:
        import tiktoken
        # 依次尝试各编码，优先使用已缓存的
        for enc_name in ("cl100k_base", "o200k_base", "p50k_base"):
            try:
                return tiktoken.get_encoding(enc_name)
            except Exception:
                continue
        return None
    except ImportError:
        return None


class TokenTracker:
    def __init__(self):
        self._tiktoken = None
        self._mode = "unavailable"
        self._estimate_multiplier = 1.0
        self._calibration_samples = 0
        self._init_tokenizer()

    def _init_tokenizer(self):
        tic = _load_tiktoken()
        if tic is not None:
            self._tiktoken = tic
            self._mode = "tiktoken"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def estimate_multiplier(self) -> float:
        return self._estimate_multiplier

    def _scale_estimate(self, value: int) -> int:
        if value <= 0:
            return 0
        return max(1, int(round(value * self._estimate_multiplier)))

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._mode == "tiktoken":
            return self._scale_estimate(len(self._tiktoken.encode(text)))
        # 粗略估算：平均每个字符 0.4 个 token（中英混合场景）
        return self._scale_estimate(max(1, int(len(text) * 0.4)))

    def estimate_tools(self, tools: list | None) -> int:
        if not tools:
            return 0
        payload = json.dumps(tools, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return self.count_tokens(payload)

    def estimate(self, messages: list) -> int:
        """估算一组 messages 的 input token 数（含角色字段开销）。"""
        total = 0
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "") or ""
            # 每条消息固定开销约 4 token（格式标记）
            total += 4 + self.count_tokens(role) + self.count_tokens(content)
            # tool_calls 内容也计入
            for tc in m.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                total += self.count_tokens(fn.get("name", ""))
                total += self.count_tokens(fn.get("arguments", ""))
        return total

    def calibrate_prompt_estimate(self, estimated_prompt_tokens: int, actual_prompt_tokens: int):
        """用真实 prompt_tokens 对估算器做渐进校准。"""
        if estimated_prompt_tokens <= 0 or actual_prompt_tokens <= 0:
            return

        ratio = actual_prompt_tokens / max(estimated_prompt_tokens, 1)
        ratio = min(1.8, max(0.7, ratio))
        alpha = 0.18 if self._calibration_samples < 6 else 0.10
        self._estimate_multiplier = round(
            max(0.6, min(2.0, ((1 - alpha) * self._estimate_multiplier) + (alpha * ratio))),
            4,
        )
        self._calibration_samples += 1

    def record_usage(self, session_manager, session_id: str, model: str, usage) -> dict:
        """从 API 返回的 usage 对象读取真实消耗并持久化记录。

        usage 可以是 openai.types.CompletionUsage 对象或 dict。
        返回记录的 dict。
        """
        if usage is None:
            return {}
        if hasattr(usage, "prompt_tokens"):
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0
        elif isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
        else:
            return {}

        record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "timestamp": datetime.now().isoformat(),
        }
        session_manager.append_token_usage(session_id, record)
        return record

    def get_session_summary(self, session_manager, session_id: str) -> dict:
        """汇总本会话累计 token 消耗。"""
        records = session_manager.get_token_usage(session_id)
        total_input = 0
        total_output = 0
        per_model: dict = {}

        for r in records:
            model = r.get("model", "unknown")
            inp = r.get("input_tokens", 0)
            out = r.get("output_tokens", 0)
            total_input += inp
            total_output += out

            if model not in per_model:
                per_model[model] = {"input": 0, "output": 0, "turns": 0}
            per_model[model]["input"] += inp
            per_model[model]["output"] += out
            per_model[model]["turns"] += 1

        return {
            "total_input": total_input,
            "total_output": total_output,
            "total_tokens": total_input + total_output,
            "per_model": per_model,
            "turns": len(records),
            "tokenizer_mode": self._mode,
        }
