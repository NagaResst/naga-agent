import os
import re

try:
    import requests as _requests
except ImportError:
    _requests = None

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

# 文件类型映射（扩展名 → 中文描述）
_FILE_TYPE_MAP = {
    ".py": "Python 源代码",
    ".js": "JavaScript 源代码",
    ".ts": "TypeScript 源代码",
    ".sh": "Shell 脚本",
    ".bash": "Bash 脚本",
    ".yaml": "YAML 配置",
    ".yml": "YAML 配置",
    ".json": "JSON 数据",
    ".toml": "TOML 配置",
    ".ini": "INI 配置",
    ".conf": "配置文件",
    ".cfg": "配置文件",
    ".xml": "XML 文档",
    ".html": "HTML 页面",
    ".md": "Markdown 文档",
    ".txt": "纯文本",
    ".log": "日志文件",
    ".sql": "SQL 脚本",
    ".go": "Go 源代码",
    ".java": "Java 源代码",
    ".rs": "Rust 源代码",
    ".rb": "Ruby 源代码",
    ".php": "PHP 源代码",
    ".css": "CSS 样式表",
    ".dockerfile": "Dockerfile",
    ".tf": "Terraform 配置",
    ".env": "环境变量文件",
    ".gradle": "Gradle 构建文件",
    ".kt": "Kotlin 源代码",
    ".groovy": "Groovy 脚本",
}

# 移除这些标签及其内容（噪音）
_NOISE_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "form", "iframe", "noscript", "advertisement", "ads",
    "banner", "cookie", "popup", "modal",
]

# 优先提取正文的选择器（按优先级排列）
_CONTENT_SELECTORS = [
    "article",
    "main",
    "[role=main]",
    "#content",
    ".content",
    "#main-content",
    ".main-content",
    ".post-content",
    ".article-content",
    ".entry-content",
    "#article",
    ".article",
    ".markdown-body",   # GitHub
    ".doc-content",
    ".documentation",
]

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

_HTTP_STATUS_MESSAGES = {
    400: "请求格式错误",
    401: "需要身份验证",
    403: "访问被拒绝（可能需要登录或被反爬）",
    404: "页面不存在",
    429: "请求频率超限，请稍后重试",
    500: "服务器内部错误",
    502: "网关错误",
    503: "服务暂时不可用",
}

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "访问 URL，返回页面正文纯文本（自动过滤噪音）。",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "完整 URL（含 http/https）"},
                "max_chars": {"type": "integer", "description": "正文最大字符数（默认4000，最大8000）"},
            },
            "required": ["url"],
        },
    },
}


def _extract_text(html: str) -> tuple[str, str]:
    """提取页面正文，返回 (正文文本, 页面标题)。"""
    if not _BS4_AVAILABLE:
        # 降级：简单去除标签
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text, ""

    soup = BeautifulSoup(html, "lxml")

    # 提取标题
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # 移除噪音标签
    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # 尝试按优先级选择器提取正文区域
    content_element = None
    for selector in _CONTENT_SELECTORS:
        content_element = soup.select_one(selector)
        if content_element:
            break

    # 回退到 body
    if not content_element:
        content_element = soup.find("body") or soup

    # 提取文本段落
    paragraphs = []
    for element in content_element.find_all(
        ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "code", "td", "th", "dt", "dd"]
    ):
        text = element.get_text(separator=" ", strip=True)
        # 过滤过短的碎片（< 10 字）
        if len(text) >= 10:
            # 标题加前缀标记
            if element.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(element.name[1])
                prefix = "#" * level
                paragraphs.append(f"{prefix} {text}")
            elif element.name == "pre":
                paragraphs.append(f"```\n{text}\n```")
            else:
                paragraphs.append(text)

    # 去重相邻重复段落
    deduped = []
    prev = None
    for p in paragraphs:
        if p != prev:
            deduped.append(p)
        prev = p

    return "\n\n".join(deduped), title


def execute(args: dict) -> str:
    if _requests is None:
        return "[fetch_url 错误] requests 库未安装，请运行：pip install requests"

    url = args.get("url", "").strip()
    if not url:
        return "[fetch_url 错误] URL 不能为空。"
    if not url.startswith(("http://", "https://")):
        return "[fetch_url 错误] URL 必须以 http:// 或 https:// 开头。"

    max_chars = min(max(int(args.get("max_chars", 4000)), 500), 8000)

    try:
        resp = _requests.get(
            url,
            headers=_DEFAULT_HEADERS,
            timeout=15,
            allow_redirects=True,
        )
    except _requests.exceptions.Timeout:
        return f"[fetch_url 错误] 请求超时（>15s）：{url}"
    except _requests.exceptions.SSLError:
        return f"[fetch_url 错误] SSL 证书验证失败：{url}"
    except _requests.exceptions.ConnectionError:
        return f"[fetch_url 错误] 无法连接到目标服务器：{url}"
    except Exception as e:
        return f"[fetch_url 错误] 请求异常：{e}"

    actual_url = resp.url
    status_code = resp.status_code
    status_msg = _HTTP_STATUS_MESSAGES.get(status_code, "")
    status_label = f"{status_code} OK" if status_code == 200 else f"{status_code} {status_msg or '错误'}"

    if status_code != 200:
        return (
            f"[网页内容]\n"
            f"URL：{actual_url}\n"
            f"HTTP 状态：{status_label}\n\n"
            f"无法获取页面内容。{('建议：' + status_msg) if status_msg else ''}"
        )

    # 处理编码
    content_type = resp.headers.get("Content-Type", "")
    if "charset=" in content_type:
        encoding = content_type.split("charset=")[-1].split(";")[0].strip()
        try:
            html = resp.content.decode(encoding, errors="replace")
        except (LookupError, UnicodeDecodeError):
            html = resp.text
    else:
        resp.encoding = resp.apparent_encoding or "utf-8"
        html = resp.text

    # 检查是否为纯文本/JSON（非 HTML）
    is_html = "<html" in html[:500].lower() or "<!doctype" in html[:200].lower()
    if not is_html:
        # 直接截断返回
        raw = html.strip()
        char_count = len(raw)
        truncated = raw[:max_chars]
        suffix = f"\n\n[...已截断，完整内容约 {char_count} 字，当前显示前 {max_chars} 字]" if char_count > max_chars else ""
        return (
            f"[网页内容]\n"
            f"URL：{actual_url}\n"
            f"HTTP 状态：{status_label}\n"
            f"内容类型：纯文本/非 HTML\n"
            f"正文字数：约 {char_count} 字\n"
            f"---\n"
            f"{truncated}{suffix}"
        )

    body_text, page_title = _extract_text(html)

    if not body_text or len(body_text) < 50:
        return (
            f"[网页内容]\n"
            f"URL：{actual_url}\n"
            f"HTTP 状态：{status_label}\n"
            f"页面标题：{page_title or '（无标题）'}\n\n"
            f"页面正文内容为空或极短，可能需要 JavaScript 渲染（动态页面），无法直接抓取。"
        )

    total_chars = len(body_text)
    truncated_text = body_text[:max_chars]
    is_truncated = total_chars > max_chars

    header_lines = [
        "[网页内容]",
        f"URL：{actual_url}",
        f"HTTP 状态：{status_label}",
        f"页面标题：{page_title or '（无标题）'}",
        f"正文字数：约 {total_chars} 字",
    ]
    if is_truncated:
        header_lines.append(f"读取范围：前 {max_chars} 字（完整内容约 {total_chars} 字）")
    header_lines.append("---")

    footer = f"\n---\n提示：如需读取更多内容，可增大 max_chars 参数（最大 8000）。" if is_truncated else ""

    return "\n".join(header_lines) + "\n" + truncated_text + footer
