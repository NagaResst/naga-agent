import re
import hashlib

# 触发 complex/high 路由的关键词
_COMPLEX_KEYWORDS = re.compile(
    r"(写代码|生成代码|编写|实现|开发|设计|架构|优化|重构|分析|调试|排查|解释原理"
    r"|深入|详细|完整|全面|系统|对比|评估|方案|部署|迁移|性能|安全|漏洞"
    r"|write code|implement|develop|design|architecture|optimize|refactor"
    r"|analyze|debug|explain|comprehensive|detailed|complete)",
    re.IGNORECASE,
)

# 触发 simple/low 路由的关键词（简单问候/查询）
_SIMPLE_KEYWORDS = re.compile(
    r"^(你好|hi|hello|嗨|早|帮我查|查一下|是什么|什么是|几点|今天|现在"
    r"|what is|who is|when|where|how many|tell me)[\s\S]{0,30}$",
    re.IGNORECASE,
)

_CLASSIFIER_SYSTEM = (
    "你是一个任务复杂度分类器。根据用户输入，只输出以下三个词之一，不要有任何其他内容：\n"
    "simple\n"
    "medium\n"
    "complex\n\n"
    "判断标准：\n"
    "- simple：简单问候、单一事实查询、短文本翻译\n"
    "- medium：需要一定推理或知识整合，但不涉及大量代码或深度分析\n"
    "- complex：代码生成、系统设计、深度分析、长文档处理、多步骤技术任务"
)


class ModelRouter:
    _CACHE_TTL = 7 * 24 * 3600          # 路由分类缓存 7 天
    _CACHE_PREFIX = "naga_agent:route_cache:"

    def __init__(self, routing_config: dict, redis_client=None, default_model: str = ""):
        self._cfg = routing_config
        self._tier_to_model: dict = routing_config.get("tier_to_model", {})
        self._model_map: dict = routing_config.get("model_map", {
            "simple": "low", "medium": "medium", "complex": "high"
        })
        self._redis = redis_client
        self._default_model = default_model

    def get_model_by_tier(self, tier: str) -> str:
        model = self._tier_to_model.get(tier)
        if model:
            return model
        # 降级：找最低可用 tier
        for fallback in ("low", "medium", "high"):
            m = self._tier_to_model.get(fallback)
            if m:
                return m
        return self._default_model

    def _rule_route(self, user_input: str, history_len: int, agent_cfg: dict) -> str | None:
        """规则层路由，返回模型名或 None（表示需要分类层）。"""
        text = user_input.strip()
        token_estimate = max(1, int(len(text) * 0.4))

        # 明显简单
        if token_estimate < 30 and _SIMPLE_KEYWORDS.match(text):
            return self.get_model_by_tier(self._model_map.get("simple", "low"))

        # 明显复杂
        if _COMPLEX_KEYWORDS.search(text):
            return self.get_model_by_tier(self._model_map.get("complex", "high"))

        # 长对话上下文 → 偏向 medium
        if history_len > 20:
            return self.get_model_by_tier(self._model_map.get("medium", "medium"))

        return None  # 交给分类层

    def _cache_key(self, text: str) -> str:
        """对标准化后的输入文本生成 Redis cache key。"""
        normalized = " ".join(text.strip().split())[:200]
        digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        return self._CACHE_PREFIX + digest

    def _classifier_route(self, user_input: str, client) -> tuple:
        """用 classifier_model 做轻量分类，返回 (model, label)。"""
        classifier_model = self._cfg.get("classifier_model", "qwen-turbo")
        try:
            resp = client.chat.completions.create(
                model=classifier_model,
                messages=[
                    {"role": "system", "content": _CLASSIFIER_SYSTEM},
                    {"role": "user", "content": user_input[:500]},  # 最多 500 字避免浪费
                ],
                temperature=0.0,
                max_tokens=5,
            )
            label = resp.choices[0].message.content.strip().lower()
        except Exception:
            label = "medium"

        if label not in ("simple", "medium", "complex"):
            label = "medium"

        tier = self._model_map.get(label, "medium")
        return self.get_model_by_tier(tier), label

    def _cached_classify(self, user_input: str, client) -> tuple:
        """带 Redis 缓存的分类路由，命中则跳过 API 调用。"""
        if self._redis is not None:
            try:
                key = self._cache_key(user_input)
                cached = self._redis.get(key)
                if cached and cached in ("simple", "medium", "complex"):
                    label = cached
                    tier = self._model_map.get(label, "medium")
                    return self.get_model_by_tier(tier), f"classifier:{label}(cached)"
                # 未命中，调用 API
                model, label = self._classifier_route(user_input, client)
                self._redis.setex(key, self._CACHE_TTL, label)
                return model, f"classifier:{label}"
            except Exception:
                pass  # Redis 异常降级
        return self._classifier_route(user_input, client)

    def route(
        self,
        user_input: str,
        history_len: int,
        agent_cfg: dict,
        client,
        manual_model: str | None = None,
    ) -> tuple[str, str]:
        """完整路由，返回 (model_name, reason)。

        manual_model: 用户通过 /model 手动指定的模型，非空时直接返回，跳过路由。
        """
        if not self._cfg.get("enabled", False):
            return manual_model or self.get_model_by_tier("medium"), "routing_disabled"

        if manual_model:
            return manual_model, "manual"

        # 第一层：规则
        rule_result = self._rule_route(user_input, history_len, agent_cfg)
        if rule_result:
            return rule_result, "rule"

        # 第二层：分类模型（带 Redis 缓存）
        model, reason = self._cached_classify(user_input, client)
        return model, reason
