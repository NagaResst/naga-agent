import re

try:
    import requests as _requests
except ImportError:
    _requests = None

try:
    from ddgs import DDGS as _DDGS
except ImportError:
    _DDGS = None

_config: dict = {}

# 域名可信度白名单
_OFFICIAL_DOMAINS = {
    "kubernetes.io", "docs.docker.com", "docs.python.org", "docs.aliyuncs.com",
    "help.aliyun.com", "cloud.google.com", "docs.microsoft.com", "learn.microsoft.com",
    "docs.aws.amazon.com", "nginx.org", "redis.io", "postgresql.org",
    "docs.github.com", "prometheus.io", "grafana.com",
}
_KNOWN_DOMAINS = {
    "github.com", "stackoverflow.com", "juejin.cn", "zhihu.com",
    "segmentfault.com", "cnblogs.com", "csdn.net", "jianshu.com",
    "infoq.cn", "oschina.net", "v2ex.com", "linux.do",
}

BOCHA_SEARCH_URL = "https://api.bochaai.com/v1/web-search"

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "联网搜索。摘要不足时用 fetch_url 读原文，勿连续重复搜索同一主题。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "count": {"type": "integer", "description": "返回数量（1-10，默认5）"},
                "freshness": {"type": "string", "description": "时效：oneDay/oneWeek/oneMonth/noLimit"},
            },
            "required": ["query"],
        },
    },
}


def set_config(cfg: dict):
    global _config
    _config = cfg


def _classify_domain(url: str) -> str:
    try:
        domain = re.sub(r"^https?://", "", url).split("/")[0].lower()
        # 去掉 www. 前缀
        domain_root = domain[4:] if domain.startswith("www.") else domain
        if domain_root in _OFFICIAL_DOMAINS or domain.endswith(tuple(
            f".{d}" for d in _OFFICIAL_DOMAINS
        )):
            return "官方文档"
        if domain_root in _KNOWN_DOMAINS or domain.endswith(tuple(
            f".{d}" for d in _KNOWN_DOMAINS
        )):
            return "知名来源"
        return "未知来源"
    except Exception:
        return "未知来源"


def _ddg_search(query: str, count: int) -> str:
    """使用 DuckDuckGo 搜索，作为 Bocha 不可用时的 fallback。"""
    if _DDGS is None:
        return "[web_search 错误] duckduckgo-search 未安装，请运行：pip install duckduckgo-search"

    try:
        results = list(_DDGS().text(query, max_results=count))
    except Exception as e:
        return f"[web_search 错误] DuckDuckGo 搜索失败：{e}"

    if not results:
        return (
            f"[网络搜索结果]（DuckDuckGo）\n"
            f"查询词：{query}\n"
            f"结果数量：0 条\n\n"
            f"未找到相关结果，建议更换关键词重新搜索。"
        )

    lines = [
        "[网络搜索结果]（DuckDuckGo · Bocha 不可用已自动切换）",
        f"查询词：{query}",
        f"结果数量：{len(results)} 条",
        "",
    ]
    for i, item in enumerate(results, 1):
        title = item.get("title") or "（无标题）"
        url = item.get("href") or item.get("url") or ""
        snippet = item.get("body") or item.get("snippet") or "（无摘要）"
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        credibility = _classify_domain(url)
        lines.append(f"[{i}] 标题：{title}")
        lines.append(f"    URL：{url}")
        lines.append(f"    摘要：{snippet}")
        lines.append(f"    可信度：{credibility}")
        lines.append("")

    lines.append("提示：如摘要不足以判断，请先用 fetch_url 读取上述推荐结果的原文，再决定是否需要重新搜索。")
    return "\n".join(lines)


def execute(args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return "[web_search 错误] 查询词不能为空。"

    count = min(max(int(args.get("count", 5)), 1), 10)

    api_key = _config.get("api_key", "")
    # 无 Bocha key 时直接走 DDG
    if not api_key:
        return _ddg_search(query, count)

    if _requests is None:
        return _ddg_search(query, count)

    freshness = args.get("freshness", _config.get("freshness", "noLimit"))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "count": count,
        "freshness": freshness,
        "summary": False,
    }

    try:
        resp = _requests.post(
            BOCHA_SEARCH_URL,
            json=payload,
            headers=headers,
            timeout=_config.get("timeout", 10),
        )
    except _requests.exceptions.Timeout:
        return _ddg_search(query, count)
    except _requests.exceptions.ConnectionError:
        return _ddg_search(query, count)

    if resp.status_code == 401:
        return "[web_search 错误] Bocha API Key 无效或已过期，请检查 BOCHA_API_KEY。"
    if resp.status_code in (402, 429):
        # 配额耗尽或限速 → 自动降级至 DDG
        return _ddg_search(query, count)
    if resp.status_code != 200:
        return f"[web_search 错误] HTTP {resp.status_code}：{resp.text[:200]}"

    try:
        data = resp.json()
    except Exception:
        return "[web_search 错误] 响应解析失败，返回内容非 JSON 格式。"

    # 兼容博查 API 两种响应结构
    results = []
    if "data" in data and "webPages" in data["data"]:
        results = data["data"]["webPages"].get("value", [])
    elif "webPages" in data:
        results = data["webPages"].get("value", [])
    elif isinstance(data.get("results"), list):
        results = data["results"]

    if not results:
        return (
            f"[网络搜索结果]\n"
            f"查询词：{query}\n"
            f"结果数量：0 条\n\n"
            f"未找到相关结果，建议：\n"
            f"  1. 如果有已知 URL，可尝试直接使用 fetch_url 访问\n"
            f"  2. 更换关键词重新搜索\n"
            f"  3. 尝试更简短或更通用的词语"
        )

    lines = [
        "[网络搜索结果]",
        f"查询词：{query}",
        f"结果数量：{len(results)} 条  时效过滤：{freshness}",
        "",
    ]

    for i, item in enumerate(results, 1):
        title = item.get("name") or item.get("title") or "（无标题）"
        url = item.get("url") or item.get("link") or ""
        snippet = item.get("snippet") or item.get("description") or "（无摘要）"
        site_name = item.get("siteName") or item.get("displayUrl") or ""
        date_published = item.get("datePublished") or item.get("date") or ""

        # 截断摘要
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."

        credibility = _classify_domain(url)

        lines.append(f"[{i}] 标题：{title}")
        lines.append(f"    URL：{url}")
        if site_name:
            lines.append(f"    来源：{site_name}")
        if date_published:
            lines.append(f"    发布时间：{date_published[:10]}")
        lines.append(f"    摘要：{snippet}")
        lines.append(f"    可信度：{credibility}")
        lines.append("")

    # 推荐优先读取的 URL（取可信度最高的前 2 个有 URL 的结果）
    recommended = [(i+1, url) for i, item in enumerate(results)
                   if (url := item.get("url") or item.get("link") or "")
                   and _classify_domain(url) in ("官方文档", "知名来源")][:2]
    if recommended:
        rec_lines = "  ".join(f"[{idx}] {url}" for idx, url in recommended)
        lines.append(f"推荐优先读取：{rec_lines}")

    lines.append("提示：如摘要不足以判断，请先用 fetch_url 读取上述推荐结果的原文，再决定是否需要重新搜索。")

    return "\n".join(lines)
