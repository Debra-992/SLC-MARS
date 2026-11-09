# -*- coding: utf-8 -*-
"""Invocation backend for the real SLC tools.

Two implementations, both obeying the rule "a failure is unknown, never fabricated":
  1. RealMcpBackend -- calls the four SLC stdio MCP servers through a real
     mcp.ClientSession.call_tool (needs the SLC project environment and its DB/browser/R
     dependencies; returns a structured unavailable when they are missing).
  2. UnavailableBackend -- the default fallback, returns unavailable for every tool, used for
     offline development and unit tests.

The layers above (retrieval_tools / computation_tools) depend only on this file's
`call(tool, kwargs) -> dict` protocol.
"""
from __future__ import annotations

import json
import os
from typing import Any


class Backend:
    """Tool-invocation backend protocol."""

    def call(self, tool: str, kwargs: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    async def acall(self, tool: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Async entry point; RealMcpBackend overrides it, the default backend returns unavailable."""
        return self.call(tool, kwargs)


class UnavailableBackend(Backend):
    """Offline fallback backend: always returns unavailable and touches no external dependency."""

    def call(self, tool: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {"status": "unavailable", "error": f"[offline] backend not configured for tool '{tool}'"}


class RealMcpBackend(Backend):
    """Backend connected to the real SLC multi_agent MCP services (sync wrapper + async first).

    The SLC package root and each server script are located through environment variables:
      SLC_ROOT                      -> the SLC_fresh root directory
      SLC_MCP_PUBLIC_INPUTS         -> public question JSON (needed by project_knowledge /
                                       graph_computation)
      NEO4J_URI/USER/PASSWORD, MONGO_URI/DB ... -> read by each server itself
    """

    # tool name -> (server script relative path, real MCP tool name inside that server)
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
        # Pass os.environ to the stdio server explicitly: if the mcp library gets env=None it
        # builds a minimal environment without our variables, so SLC_KMTIMER_ROOT /
        # SLC_TOOLS_CACHE_ROOT and friends vanish in the child process and safe_error then
        # reports the tool as unavailable.
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
        except Exception as e:  # noqa BLE001 -- transport/execution failure counts as unknown, never fabricated
            return {"status": "unavailable", "error": f"{type(e).__name__}: {e}"}

    def call(self, tool: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        return asyncio.run(self.acall(tool, kwargs))


def _sys_python() -> str:
    return os.environ.get("SLC_MCP_PYTHON") or "python"


def _decode(result: Any) -> dict[str, Any]:
    """Decode an MCP CallToolResult into a dict; JSON text first, otherwise the raw string."""
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
    """Pick the backend from the environment: the real one when SLC_ROOT is set and mcp is
    installed, the fallback otherwise."""
    if os.environ.get("SLC_ROOT"):
        try:
            import mcp  # noqa F401

            return RealMcpBackend()
        except ImportError:
            pass
    return UnavailableBackend()
