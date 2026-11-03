# -*- coding: utf-8 -*-
"""SLC 真实工具的调用后端。

两种落地方式，均遵循"失败即未知，绝不伪造"的原则：
  1. RealMcpBackend —— 通过真实 mcp.ClientSession.call_tool 调用 SLC 的 4 个 stdio MCP 服务
     （需 SLC 项目环境、DB/浏览器/R 依赖，缺失时返回结构化 unavailable）。
  2. UnavailableBackend —— 默认兜底，任何工具都返回 unavailable，供离线开发/单测使用。

上层 retrieval_tools / computation_tools 只依赖本文件的 `call(tool, kwargs) -> dict` 协议。
"""
from __future__ import annotations

import json
import os
from typing import Any


class Backend:
    """工具调用后端协议。"""

    def call(self, tool: str, kwargs: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - 接口
        raise NotImplementedError

    async def acall(self, tool: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """异步入口；RealMcpBackend 覆写，默认后端直接返回 unavailable。"""
        return self.call(tool, kwargs)


class UnavailableBackend(Backend):
    """离线兜底后端：一律返回 unavailable，不触碰任何外部依赖。"""

    def call(self, tool: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {"status": "unavailable", "error": f"[offline] backend not configured for tool '{tool}'"}


class RealMcpBackend(Backend):
    """连接 SLC multi_agent 真实 MCP 服务的后端（同步封装 + 异步优先）。

    通过环境变量定位 SLC 包根目录与各 server 脚本：
      SLC_ROOT                      -> SLC_fresh 根目录
      SLC_MCP_PUBLIC_INPUTS         -> 公开题目 JSON（project_knowledge / graph_computation 需要）
      NEO4J_URI/USER/PASSWORD, MONGO_URI/DB 等 -> 由各 server 自行读取
    """

    # 工具名 -> (server 脚本相对路径, 该 server 内的真实 MCP 工具名)
    _ROUTES = {
        "project_knowledge": ("multi_agent/mcp_servers/slc_benchmark_context_mcp.py", "project_knowledge"),
        "project_graph_computation": ("multi_agent/mcp_servers/slc_benchmark_context_mcp.py", "project_graph_computation"),
        "kg_schema": ("multi_agent/mcp_servers/slc_kg_mcp.py", "kg_schema"),
        "kg_neighbors": ("multi_agent/mcp_servers/slc_kg_mcp.py", "kg_gene_neighbors"),
        "kg_cypher": ("multi_agent/mcp_servers/slc_kg_mcp.py", "kg_cypher"),
        "timer": ("multi_agent/mcp_servers/bio_web_mcp.py", "timer_pancancer_expression"),
        "km": ("multi_agent/mcp_servers/bio_web_mcp.py", "km_survival"),
        "de": ("multi_agent/mcp_servers/slc_r_mcp.py", "slc_differential_expression"),
    }

    def __init__(self, slc_root: str | None = None):
        self.slc_root = slc_root or os.environ.get("SLC_ROOT")
        self._sessions: dict[str, Any] = {}
        self._ctx: dict[str, Any] = {}

    def _server_path(self, tool: str) -> str:
        rel, _ = self._ROUTES[tool]
        if not self.slc_root:
            raise RuntimeError("SLC_ROOT not set")
        return os.path.join(self.slc_root, rel)

    async def acall(self, tool: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        if tool not in self._ROUTES:
            return {"status": "error", "error": f"unknown tool {tool}"}
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            return {"status": "unavailable", "error": "mcp client library not installed"}
        if not self.slc_root:
            return {"status": "unavailable", "error": "SLC_ROOT not set"}

        _, real_name = self._ROUTES[tool]
        # 显式把 os.environ 传给 stdio server（mcp 库若收到 env=None 会构造
        # 一个不含我们变量的最小环境，导致 SLC_KMTIMER_ROOT / SLC_TOOLS_CACHE_ROOT
        # 等在子进程里丢失，进而被 safe_error 判为 unavailable）
        params = StdioServerParameters(
            command=_sys_python(),
            args=[self._server_path(tool)],
            env=dict(os.environ),
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(real_name, kwargs)
                    return _decode(result)
        except Exception as e:  # noqa BLE001 —— 传输/执行失败按未知处理，绝不伪造
            return {"status": "unavailable", "error": f"{type(e).__name__}: {e}"}

    def call(self, tool: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        return asyncio.run(self.acall(tool, kwargs))


def _sys_python() -> str:
    return os.environ.get("SLC_MCP_PYTHON") or "python"


def _decode(result: Any) -> dict[str, Any]:
    """把 MCP CallToolResult 解码为 dict；优先 JSON 文本，否则原始串。"""
    content = getattr(result, "content", None) or []
    texts = [c.text for c in content if hasattr(c, "text")]
    joined = "\n".join(texts).strip()
    if not joined:
        return {"status": "unavailable", "error": "empty MCP response"}
    try:
        return json.loads(joined)
    except json.JSONDecodeError:
        return {"status": "ok", "raw": joined}


def get_default_backend() -> Backend:
    """按环境自动选择后端：配了 SLC_ROOT 且装了 mcp 用真实后端，否则兜底。"""
    if os.environ.get("SLC_ROOT"):
        try:
            import mcp  # noqa F401

            return RealMcpBackend()
        except ImportError:
            pass
    return UnavailableBackend()
