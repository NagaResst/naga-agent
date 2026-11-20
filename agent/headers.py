import os
import re


def resolve_headers(headers: dict, label: str) -> dict[str, str]:
    if not isinstance(headers, dict):
        raise ValueError(f"{label} 必须是 Header 名称到字符串的映射")

    def replace_env(match):
        name = match.group(1)
        if not os.environ.get(name):
            raise ValueError(f"缺少 {label} 环境变量：{name}")
        return os.environ[name]

    resolved = {}
    for name, value in headers.items():
        if not isinstance(name, str) or not name or not isinstance(value, str):
            raise ValueError(f"{label} 的 Header 名称和值必须是字符串，名称不能为空")
        resolved[name] = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace_env, value)
    return resolved
