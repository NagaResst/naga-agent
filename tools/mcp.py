import asyncio
import json
import os
import re

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


_servers = {}
_timeout = 20
_output_max_chars = 16000


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "mcp",
        "description": "列出 MCP server 的工具；list 指定 tool 可查看参数 schema，再用 call 调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "call"]},
                "server": {"type": "string", "description": "已配置的 MCP server 名称；list 可省略以查看所有 server"},
                "tool": {"type": "string", "description": "list 时查看指定工具的 schema；call 时调用该工具"},
                "arguments": {"type": "object", "description": "传给 MCP 工具的参数"},
            },
            "required": ["action"],
        },
    },
}


def set_config(config: dict):
    global _servers, _timeout, _output_max_chars
    _servers = config.get("servers", {})
    _timeout = config.get("timeout", 20)
    _output_max_chars = config.get("output_max_chars", 16000)


def configured() -> bool:
    return any(server.get("enabled", True) for server in _servers.values())


def _resolve_headers(headers: dict) -> dict:
    def replace_env(match):
        name = match.group(1)
        if not os.environ.get(name):
            raise ValueError(f"缺少 MCP 环境变量：{name}")
        return os.environ[name]

    return {
        name: re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace_env, value)
        for name, value in headers.items()
    }


async def _exchange(read_stream, write_stream, action: str, tool: str, arguments: dict, timeout: int) -> str:
    async with ClientSession(read_stream, write_stream) as session:
        await asyncio.wait_for(session.initialize(), timeout=timeout)
        if action == "list":
            tools = await asyncio.wait_for(session.list_tools(), timeout=timeout)
            if tool:
                for item in tools.tools:
                    if item.name == tool:
                        return json.dumps({
                            "name": item.name, "description": item.description,
                            "inputSchema": item.inputSchema,
                        }, ensure_ascii=False)
                return f"错误：未找到 MCP 工具：{tool}"
            return json.dumps([
                {"name": item.name, "description": (item.description or "")[:160]}
                for item in tools.tools
            ], ensure_ascii=False)
        result = await asyncio.wait_for(session.call_tool(tool, arguments), timeout=timeout)
        if result.structuredContent is not None:
            text = json.dumps(result.structuredContent, ensure_ascii=False)
        else:
            text = "\n".join(
                item.text if item.type == "text" else f"[{item.type} 内容未作为文本返回]"
                for item in result.content
            )
        return ("错误：" if result.isError else "") + text


async def _request(server: dict, action: str, tool: str, arguments: dict) -> str:
    timeout = max(_timeout, server.get("connect_timeout", _timeout))
    if "url" in server:
        headers = _resolve_headers(server.get("headers", {}))
        async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(_timeout, connect=timeout)) as client:
            async with streamable_http_client(server["url"], http_client=client) as (read_stream, write_stream, _):
                return await _exchange(read_stream, write_stream, action, tool, arguments, timeout)

    params = StdioServerParameters(
        command=server["command"],
        args=server.get("args", []),
        env={**os.environ, **server.get("env", {})},
        cwd=server.get("cwd"),
    )
    async with stdio_client(params) as (read_stream, write_stream):
        return await _exchange(read_stream, write_stream, action, tool, arguments, timeout)


def execute(args: dict) -> str:
    action = args.get("action")
    if action not in ("list", "call"):
        return "错误：action 必须是 list 或 call"

    name = args.get("server")
    if action == "list" and not name:
        return json.dumps({"servers": [key for key, server in _servers.items() if server.get("enabled", True)]}, ensure_ascii=False)
    if name not in _servers or not _servers[name].get("enabled", True):
        return f"错误：未配置 MCP server：{name}"

    tool = args.get("tool")
    arguments = args.get("arguments", {})
    if action == "call" and (not isinstance(tool, str) or not tool or not isinstance(arguments, dict)):
        return "错误：call 需要 tool 名称和对象类型的 arguments"

    server = _servers[name]
    if not isinstance(server, dict) or not (server.get("command") or server.get("url")):
        return f"错误：MCP server {name} 缺少 command 或 url"
    try:
        result = asyncio.run(_request(server, action, tool, arguments))
    except Exception as exc:
        return f"错误：MCP server {name} 请求失败：{exc}"
    if action == "list" and len(result) > _output_max_chars:
        return "错误：MCP 工具列表超过输出上限，请提高 mcp.output_max_chars"
    return result[:_output_max_chars]