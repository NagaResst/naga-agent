import os
import sys

from dotenv import load_dotenv

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_config() -> dict:
    load_dotenv()

    toml_config = {}
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.toml")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            toml_config = tomllib.load(f)

    storage_section = toml_config.get("storage", {})
    db_path_default = os.path.join(os.path.dirname(os.path.dirname(__file__)), "naga.db")
    db_path = os.environ.get("NAGA_DB_PATH", storage_section.get("db_path", db_path_default))
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), db_path)

    model_section = toml_config.get("model", {})
    agent_section = toml_config.get("agent", {})

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("警告：未设置 OPENAI_API_KEY 环境变量，请在 .env 文件或环境变量中配置。")

    base_url = os.environ.get("OPENAI_BASE_URL") or model_section.get("base_url", "") or None

    project_root = os.path.dirname(os.path.dirname(__file__))

    # system_prompt 优先级：env var > system_prompt_file > system_prompt inline
    system_prompt = os.environ.get("AGENT_SYSTEM_PROMPT", "")
    if not system_prompt:
        prompt_file = agent_section.get("system_prompt_file", "")
        if prompt_file:
            prompt_file_path = os.path.join(project_root, prompt_file)
            if os.path.exists(prompt_file_path):
                with open(prompt_file_path, "r", encoding="utf-8") as f:
                    system_prompt = f.read().strip()
            else:
                print(f"警告：system_prompt_file 指定的文件不存在：{prompt_file_path}")
    if not system_prompt:
        system_prompt = agent_section.get("system_prompt", "You are a helpful assistant.")

    # 构建 tier 双向映射
    available_models = model_section.get("available", ["gpt-4o-mini", "gpt-4o"])
    tiers_raw = model_section.get("tiers", {})
    model_to_tier = {m: tiers_raw.get(m, "medium") for m in available_models}
    # 反向映射：同 tier 多个模型时，取 available 中靠后的（用户可通过排序控制优先级）
    tier_to_model: dict = {}
    for m in available_models:
        tier = model_to_tier.get(m, "medium")
        tier_to_model[tier] = m

    extra_params = model_section.get("extra_params", {})
    extra_params_default = dict(model_section.get("extra_params_default", {}))

    delegation_section = toml_config.get("delegation", {})

    tools_section = toml_config.get("tools", {})

    skills_section = toml_config.get("skills", {})

    memory_section = dict(toml_config.get("memory", {}))

    return {
        "api_key": api_key,
        "base_url": base_url,
        "storage": {
            "db_path": db_path,
        },
        "model": {
            "default": model_section.get("default", "gpt-4o-mini"),
            "available": available_models,
            "tiers": model_to_tier,
            "tier_to_model": tier_to_model,
            "extra_headers": model_section.get("extra_headers", {}),
            "extra_params": extra_params,
            "extra_params_default": extra_params_default,
        },
        "delegation": {
            "enabled": delegation_section.get("enabled", False),
            "model": os.environ.get("SUBAGENT_MODEL") or delegation_section.get("model", ""),
            "base_url": os.environ.get("SUBAGENT_BASE_URL") or delegation_section.get("base_url", ""),
            "api_key": os.environ.get("SUBAGENT_API_KEY", ""),
            "extra_headers": delegation_section.get("extra_headers", {}),
            "max_tasks": max(1, min(10, delegation_section.get("max_tasks", 3))),
            "max_rounds": max(0, min(10, delegation_section.get("max_rounds", 3))),
            "max_calls": max(1, min(10, delegation_section.get("max_calls", 2))),
            "context_token_limit": delegation_section.get("context_token_limit", 0),
        },
        "tools": {
            "command_timeout": tools_section.get("command_timeout", 30),
            "output_max_chars": tools_section.get("output_max_chars", 32000),
            "tool_max_retries": tools_section.get("tool_max_retries", 2),
            "tool_max_rounds": tools_section.get("tool_max_rounds", 10),
            "tool_max_errors": tools_section.get("tool_max_errors", 3),
        },
        "agent": {
            "max_history": agent_section.get("max_history", 150),
            "auto_confirm": agent_section.get("auto_confirm", False),
            "scripts_subdir": agent_section.get("scripts_subdir", "generated_scripts"),
            "temperature": agent_section.get("temperature", 0.7),
            "top_p": agent_section.get("top_p", 0.8),
            "show_thinking": agent_section.get("show_thinking", True),
            "enable_search": False,  # 锁死：独立计费禁止使用 DashScope 自带联网搜索
            "prompt_prefix": os.environ.get("AGENT_PROMPT_PREFIX", agent_section.get("prompt_prefix", "🤖")),
            "system_prompt": system_prompt,
            "context_token_limit": agent_section.get("context_token_limit", 0),
            "context_compact_ratio": min(0.9, max(0.5, agent_section.get("context_compact_ratio", 0.75))),
            "tool_result_keep_rounds": max(0, agent_section.get("tool_result_keep_rounds", 30)),
            "runtime_state_max_chars": agent_section.get("runtime_state_max_chars", 1800),
            "runtime_tool_events_max_items": agent_section.get("runtime_tool_events_max_items", 6),
            "runtime_findings_max_items": agent_section.get("runtime_findings_max_items", 6),
            "runtime_read_skill_files_max_items": agent_section.get("runtime_read_skill_files_max_items", 8),
            "runtime_working_set_max_items": agent_section.get("runtime_working_set_max_items", 8),
            "runtime_completed_steps_max_items": agent_section.get("runtime_completed_steps_max_items", 8),
            "runtime_constraints_max_items": agent_section.get("runtime_constraints_max_items", 10),
            "runtime_required_outputs_max_items": agent_section.get("runtime_required_outputs_max_items", 8),
        },
        "mcp": toml_config.get("mcp", {}),
        "skills": {
            "enabled": skills_section.get("enabled", []),
            "match_top_k": skills_section.get("match_top_k", 2),
            "summary_top_k": skills_section.get("summary_top_k", 6),
            "min_match_score": skills_section.get("min_match_score", 2.2),
        },
        "memory": memory_section,
        "embedding_daemon": toml_config.get("embedding_daemon", {}),
    }
