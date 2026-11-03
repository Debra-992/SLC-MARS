# -*- coding: utf-8 -*-
"""知识检索类工具适配层：对接 project_knowledge / kg_schema / kg_gene_neighbors / kg_cypher。

对应 SLC 知识智能体供给背景知识的 4 个真实工具。全部经 slc_backend 调用，
返回结构化状态（ok/unavailable/rejected/error），绝不把失败粉饰成空成功。
"""
from __future__ import annotations

from typing import Any

from knowledge_agents.schema import BackgroundKnowledge, KGEvidence, ToolStatus
from knowledge_agents.tools.cypher_guard import CypherRejected, validate_cypher
from knowledge_agents.tools.slc_backend import Backend, get_default_backend


class RetrievalTools:
    """知识检索智能体的工具集。"""

    def __init__(self, backend: Backend | None = None):
        self.backend = backend or get_default_backend()

    # ---- project_knowledge：Neo4j 图谱证据 + MongoDB 文献 ----
    async def project_knowledge(self, qid: str) -> BackgroundKnowledge:
        resp = await self.backend.acall("project_knowledge", {"qid": qid})
        status = ToolStatus(resp.get("status", "error"))
        kg = BackgroundKnowledge(
            question=f"qid={qid}",
            status=status,
            source=resp.get("source"),
            candidates=list(resp.get("candidates") or []),
            requested_context=dict(resp.get("requested_context") or {}),
            public_web_used=bool(resp.get("public_web_used", False)),
            source_independence=str(resp.get("source_independence", "NOT_ESTABLISHED")),
            error=resp.get("error"),
        )
        for receipt in resp.get("graph_queries") or []:
            kg.evidence.append(
                KGEvidence(
                    operation=receipt.get("operation", "graph"),
                    rows=list(receipt.get("rows") or []),
                    cypher=receipt.get("cypher"),
                    status=ToolStatus(receipt.get("status", "ok")) if "status" in receipt else ToolStatus.OK,
                )
            )
        lit = resp.get("literature") or []
        if lit:
            kg.evidence.append(KGEvidence(operation="literature", rows=list(lit)))
        return kg

    # ---- kg_schema：仅 text2cypher 上下文，不作生物学证据 ----
    async def kg_schema(self) -> dict[str, Any]:
        resp = await self.backend.acall("kg_schema", {})
        if ToolStatus(resp.get("status", "error")) is ToolStatus.OK:
            return {
                "labels": resp.get("labels") or [],
                "rel_types": resp.get("rel_types") or [],
                "property_keys": resp.get("property_keys") or [],
            }
        return {"labels": [], "rel_types": [], "property_keys": [], "status": resp.get("status"), "error": resp.get("error")}

    # ---- kg_gene_neighbors：候选缺覆盖时补直接邻居 ----
    async def kg_gene_neighbors(self, gene: str, limit: int = 50) -> dict[str, Any]:
        resp = await self.backend.acall("kg_neighbors", {"gene": gene, "limit": limit})
        return {
            "gene": gene,
            "status": resp.get("status", "error"),
            "count": resp.get("count"),
            "neighbors": resp.get("neighbors") or [],
            "error": resp.get("error"),
        }

    # ---- kg_cypher：text2cypher 执行后端（只读守卫） ----
    async def kg_cypher(self, query: str, limit: int = 100) -> dict[str, Any]:
        try:
            safe = validate_cypher(query)
        except (CypherRejected, ValueError) as e:
            return {"status": "rejected", "count": None, "rows": [], "error": str(e)}
        resp = await self.backend.acall("kg_cypher", {"query": safe, "limit": limit})
        return {
            "query": safe,
            "status": resp.get("status", "error"),
            "count": resp.get("count"),
            "rows": resp.get("rows") or [],
            "error": resp.get("error"),
        }
