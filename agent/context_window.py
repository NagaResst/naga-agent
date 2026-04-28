"""模型上下文窗口大小映射表。

查找优先级：
  1. config 手动指定（context_token_limit > 0）
  2. 精确模型名匹配
  3. 前缀匹配（处理带日期后缀的模型名）
  4. 默认值 131072（128k）
"""

_DEFAULT = 131072  # 128k，保守但足够的通用默认值

_KNOWN_CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI
    "gpt-4.1":                    1047576,
    "gpt-4.1-mini":               1047576,
    "gpt-4.1-nano":               1047576,
    "gpt-4o":                      128000,
    "gpt-4o-mini":                 128000,
    "gpt-4-turbo":                 128000,
    "gpt-4":                         8192,
    "gpt-3.5-turbo":               16385,
    "o1":                          200000,
    "o1-mini":                     128000,
    "o3":                          200000,
    "o3-mini":                     200000,
    "o4-mini":                     200000,
    # Anthropic Claude
    "claude-3-5-sonnet":           200000,
    "claude-3-5-haiku":            200000,
    "claude-3-opus":               200000,
    "claude-3-sonnet":             200000,
    "claude-3-haiku":              200000,
    # DeepSeek
    "deepseek-chat":               128000,
    "deepseek-reasoner":           128000,
    "deepseek-r1":                 128000,
    "deepseek-v3":                 128000,
    # Qwen3 系列
    "qwen3-235b-a22b":             131072,
    "qwen3-32b":                   131072,
    "qwen3-14b":                   131072,
    "qwen3-8b":                    131072,
    "qwen3-4b":                    131072,
    "qwen3.6-flash":               131072,
    "qwen3.6-plus":                131072,
    "qwen3.6-max":                 131072,
    # Qwen2.5 系列
    "qwen2.5-72b-instruct":        131072,
    "qwen2.5-32b-instruct":        131072,
    "qwen2.5-14b-instruct":        131072,
    "qwen2.5-7b-instruct":         131072,
    "qwen-max":                    131072,
    "qwen-plus":                   131072,
    "qwen-turbo":                   32000,
    # Gemini
    "gemini-2.0-flash":           1048576,
    "gemini-1.5-pro":             2097152,
    "gemini-1.5-flash":           1048576,
    # Mistral
    "mistral-large":               131072,
    "mistral-small":               131072,
    "mixtral-8x22b":                65536,
    # Meta Llama（Ollama / 本地）
    "llama3.3":                    131072,
    "llama3.2":                    131072,
    "llama3.1":                    131072,
    "llama3":                        8192,
    "llama2":                        4096,
    # Ollama 其他常用模型
    "mistral":                      32768,
    "codellama":                    16384,
    "phi4":                        131072,
    "phi3":                        131072,
    "gemma3":                      131072,
    "gemma2":                        8192,
    "qwq":                         131072,
}


def get_context_window(model: str, config_override: int = 0) -> int:
    """返回指定模型的上下文窗口大小（token 数）。

    Args:
        model: 模型名称，如 "gpt-4o" 或 "qwen3.6-flash-2026-04-16"
        config_override: config.toml 中手动指定的值；> 0 时直接返回此值

    Returns:
        上下文窗口大小（token 数）
    """
    if config_override and config_override > 0:
        return config_override

    # 精确匹配
    if model in _KNOWN_CONTEXT_WINDOWS:
        return _KNOWN_CONTEXT_WINDOWS[model]

    # 前缀匹配（处理带日期/版本后缀的模型名，如 qwen3.6-flash-2026-04-16）
    model_lower = model.lower()
    for known, size in _KNOWN_CONTEXT_WINDOWS.items():
        if model_lower.startswith(known.lower()):
            return size

    return _DEFAULT
