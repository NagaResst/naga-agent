import re
import hashlib
import time
from typing import Optional, Tuple

# 触发 complex/high 路由的关键词
# 只保留「动作意图」明确的词，避免「架构/设计/分析」等单独出现时的假阳性
_COMPLEX_KEYWORDS = re.compile(
    r"(写代码|生成代码|编写|实现|重构|调试|排查|解释原理|帮我写|帮我实现"
    r"|部署|迁移|漏洞|攻击|注入"
    r"|write code|implement|refactor|debug|develop a|build a|create a"
    r"|step[- ]by[- ]step|step by step)",
    re.IGNORECASE,
)

# 触发 simple/low 路由的关键词（简单问候/查询）
_SIMPLE_KEYWORDS = re.compile(
    r"^(你好|hi|hello|嗨|早|帮我查|查一下|是什么|什么是|几点|今天|现在"
    r"|what is|who is|when|where|how many|tell me)[\s\S]{0,30}$",
    re.IGNORECASE,
)

_CLASSIFIER_SYSTEM = (
    "你是一个任务复杂度分类器。根据用户输入，只输出以下四个词之一，不要有任何其他内容：\n"
    "simple\n"
    "medium\n"
    "complex\n"
    "plan\n\n"
    "判断标准：\n"
    "- simple：简单问候、单一事实查询、短词翻译、是非题\n"
    "- medium：需要知识整合或一定推理，但一次回复即可完成；包括概念解释、对比分析、方案建议等\n"
    "- complex：需要深度推理或长篇回答，但仍是单次回复；如详细技术解析、多维度评估\n"
    "- plan：必须分多步骤执行的任务；如写代码并运行、调用多个工具完成、系统设计并验证\n\n"
    "示例（输入 → 输出）：\n"
    "你好 → simple\n"
    "什么是 TCP/IP？ → simple\n"
    "Redis 和 Memcached 有什么区别？ → medium\n"
    "比较这四种基金类型的特点和适用场景 → medium\n"
    "解释微服务架构的优缺点及常见陷阱 → complex\n"
    "帮我写一个 Python 脚本，定时从 MySQL 同步数据到 Redis → plan\n"
    "分析这段代码的性能瓶颈并重构 → plan\n"
    "设计一个高可用的消息队列系统，给出架构图和关键组件 → plan"
)


class ModelRouter:
    _CACHE_TTL = 7 * 24 * 3600          # 路由分类缓存 7 天（进程内）

    def __init__(self, routing_config: dict, default_model: str = ""):
        self._cfg = routing_config
        self._tier_to_model: dict = routing_config.get("tier_to_model", {})
        self._model_map: dict = routing_config.get("model_map", {
            "simple": "low", "medium": "medium", "complex": "high", "plan": "high"
        })
        # 进程内内存缓存：{cache_key: (label, expire_ts)}
        self._mem_cache: dict = {}
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

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """CJK 字符感知的 token 估算：中文约 1.5 char/token，ASCII 约 0.25 word/token。"""
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
                        or '\u3400' <= c <= '\u4dbf'
                        or '\uf900' <= c <= '\ufaff')
        ascii_words = len(re.findall(r'[a-zA-Z0-9]+', text))
        return max(1, int(cjk_count * 1.5 + ascii_words * 0.3 + (len(text) - cjk_count - ascii_words) * 0.4))

    def _rule_route(self, user_input: str, history_len: int, agent_cfg: dict) -> Optional[Tuple[str, str]]:
        """规则层路由，返回 (model, label) 或 None（表示需要分类层）。"""
        text = user_input.strip()
        token_estimate = self._estimate_tokens(text)

        # 超长输入必然是 complex，跳过分类器节省 API 调用
        if token_estimate > 400:
            return self.get_model_by_tier(self._model_map.get("complex", "high")), "complex"

        # 明显简单：仅在全新对话（无历史）时生效，follow-up 不降级
        if token_estimate < 30 and _SIMPLE_KEYWORDS.match(text) and history_len <= 1:
            return self.get_model_by_tier(self._model_map.get("simple", "low")), "simple"

        # 动作意图明确的复杂请求 → 关键词命中直接走 plan
        if _COMPLEX_KEYWORDS.search(text):
            return self.get_model_by_tier(self._model_map.get("plan", "high")), "plan"

        return None  # 交给分类层（会携带上下文消息）

    def _cache_key(self, text: str) -> str:
        """对标准化后的输入文本生成缓存 key。"""
        normalized = " ".join(text.strip().split())[:200]
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def _classifier_route(self, user_input: str, client, context_messages: list = None, session_tag: str = "") -> tuple:
        """用 classifier_model 做轻量分类，返回 (model, label)。
        session_tag: AI 提取的会话主题，注入为上下文提示让分类器感知领域。
        """
        classifier_model = self._cfg.get("classifier_model", "qwen-turbo")
        system_content = _CLASSIFIER_SYSTEM
        if session_tag:
            system_content += f"\n\n当前对话主题：{session_tag}"
        messages = [{"role": "system", "content": system_content}]
        if context_messages:
            ctx = next((m for m in reversed(context_messages) if m.get("role") == "assistant"), None)
            if ctx:
                snippet = (ctx.get("content") or "")[:300]
                messages.append({"role": "assistant", "content": snippet})
        messages.append({"role": "user", "content": user_input[:500]})
        try:
            resp = client.chat.completions.create(
                model=classifier_model,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=5,
            )
            label = resp.choices[0].message.content.strip().lower()
        except Exception:
            label = "medium"

        if label not in ("simple", "medium", "complex", "plan"):
            label = "medium"

        tier = self._model_map.get(label, "medium")
        return self.get_model_by_tier(tier), label

    def _cached_classify(self, user_input: str, client, context_messages: list = None, session_tag: str = "") -> tuple:
        """带内存缓存的分类路由，命中则跳过 API 调用。

        缓存 key = session_tag + user_input，同一问题在不同会话场景下独立缓存。
        session_tag：会话指纹（取首条 user 消息前50字），零成本捕获会话领域特征。
        """
        key = self._cache_key(session_tag + "||" + user_input)
        now = time.time()
        cached = self._mem_cache.get(key)
        if cached:
            label, expire_ts = cached
            if now < expire_ts and label in ("simple", "medium", "complex", "plan"):
                tier = self._model_map.get(label, "medium")
                return self.get_model_by_tier(tier), f"classifier:{label}(cached)"
        # 未命中，调用 API
        model, label = self._classifier_route(user_input, client, context_messages)
        self._mem_cache[key] = (label, now + self._CACHE_TTL)
        return model, f"classifier:{label}"

    def route(
        self,
        user_input: str,
        history_len: int,
        agent_cfg: dict,
        client,
        manual_model: Optional[str] = None,
        context_messages: list = None,
        session_tag: str = "",
    ) -> Tuple[str, str, str]:
        """完整路由，返回 (model_name, reason, complexity)。

        complexity: simple / medium / complex，供 plan_node 判断是否触发规划。
        manual_model: 用户通过 /model 手动指定的模型，非空时直接返回，跳过路由。
        context_messages: 最近几条对话消息，传给分类器提升上下文理解准确度。
        """
        if not self._cfg.get("enabled", False):
            return manual_model or self.get_model_by_tier("medium"), "routing_disabled", "simple"

        if manual_model:
            return manual_model, "manual", "simple"

        # 第一层：规则
        rule_result = self._rule_route(user_input, history_len, agent_cfg)
        if rule_result:
            model, label = rule_result
            return model, "rule", label

        # 第二层：分类模型（带进程内缓存，session_tag 区隔不同会话场景）
        model, reason = self._cached_classify(user_input, client, context_messages, session_tag)
        # reason 格式："classifier:complex" / "classifier:complex(cached)"
        label = reason.split(":", 1)[-1].split("(")[0].strip()
        if label not in ("simple", "medium", "complex", "plan"):
            label = "medium"
        return model, reason, label
