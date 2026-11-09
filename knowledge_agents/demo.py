# -*- coding: utf-8 -*-
"""Demo entry points for the knowledge-retrieval and computation agents.

Backend selection (automatic, driven by the environment):
  - SLC_ROOT unset -> UnavailableBackend (offline, everything unknown, proving nothing is fabricated)
  - SLC_ROOT set + mcp installed -> RealMcpBackend (the real SLC stdio MCP servers)
Before connecting to real MCP, set: SLC_ROOT / SLC_MCP_PYTHON / SLC_KMTIMER_ROOT / the database env.

Demo coverage:
  demo_compute, pairwise bucket -> TIMER + KM hit the cache; DESeq2 returns download_required
  demo_compute_topk, topk bucket -> additionally triggers PPR (needs a reachable remote Neo4j)
  demo_knowledge on a whitelisted qid -> project_knowledge (KG receipts + literature)
"""
from __future__ import annotations

import asyncio

from metagpt.logs import logger

from knowledge_agents.roles.compute_role import ComputeRole
from knowledge_agents.roles.knowledge_query_role import KnowledgeQueryRole
from knowledge_agents.schema import parse_knowledge


async def demo_compute():
    """pairwise routing: TIMER + KM hit the cache; DESeq2 without counts returns download_required."""
    role = ComputeRole()
    # SLC25A1/COAD: km_plots holds SLC25A1_colon_km_stats.txt and timer_cache holds boxplot_SLC25A1.png
    role.set_task(gene="SLC25A1", cancer="COAD", task_bucket="pairwise")
    await role.run("gene=SLC25A1 cancer=COAD give the differential expression evidence and the patient survival association")
    ck = parse_knowledge(role.rc.memory.get()[-1].content)
    logger.info(f"[candidate knowledge pairwise] status={ck.status.value}")
    if ck.expression:
        logger.info(f"  TIMER: status={ck.expression.status.value} scope={ck.expression.evidence_scope} "
                    f"boxplot={ck.expression.boxplot_path}")
    if ck.survival:
        stats = (ck.survival.stats_text or "").splitlines()
        stats_head = stats[0] if stats else ""
        logger.info(f"  KM: status={ck.survival.status.value} scope={ck.survival.evidence_scope} "
                    f"stats_head={stats_head!r}")
    logger.info(f"  {ck.summary}")


async def demo_compute_topk():
    """topk routing: additionally triggers PPR, which needs a reachable remote Neo4j; SLC16A1 has no TIMER cache,
    so a cache miss is expected and shows that evidence integrity does not fabricate."""
    role = ComputeRole()
    role.set_task(qid="pairwise_negative_CRC_Human_HCT116_SLC16A1_KO_00179",
                  gene="SLC16A1", cancer="COAD", task_bucket="topk")
    await role.run("gene=SLC16A1 cancer=COAD topk rank the SLC family candidates, assessing proximity, differential expression and survival association")
    ck = parse_knowledge(role.rc.memory.get()[-1].content)
    logger.info(f"[candidate knowledge topk] status={ck.status.value}")
    if ck.graph_computation:
        gc = ck.graph_computation
        top = (gc.candidates or [])[:5]
        top_str = ", ".join(f"{c.gene}={c.degree_corrected_score}" for c in top) if top else "none"
        logger.info(f"  PPR: status={gc.status.value} seeds={gc.seed_count} "
                    f"nodes={gc.node_count} edges={gc.edge_count}  top5=[{top_str}]")
    if ck.expression:
        logger.info(f"  TIMER: status={ck.expression.status.value}")
    if ck.survival:
        stats = (ck.survival.stats_text or "").splitlines()
        logger.info(f"  KM: status={ck.survival.status.value} stats_head={stats[0] if stats else ''!r}")
    logger.info(f"  {ck.summary}")


async def demo_knowledge():
    """Whitelisted qid: goes through project_knowledge. With Neo4j up, the three KG receipts should be ok;
    with Mongo papers available the literature receipt should be ok too."""
    role = KnowledgeQueryRole()
    await role.run("qid=pairwise_negative_CRC_Human_HCT116_SLC16A1_KO_00179 retrieve SLC16A1 colorectal cancer background knowledge")
    kg = parse_knowledge(role.rc.memory.get()[-1].content)
    logger.info(f"[background knowledge] status={kg.status.value} candidates={kg.candidates} evidence={len(kg.evidence)}")
    for ev in kg.evidence:
        rows = len(ev.rows) if ev.rows else 0
        logger.info(f"  - op={ev.operation:<20} status={ev.status.value:<15} rows={rows}")
    logger.info(f"           {kg.summary}")


async def main():
    await demo_compute()
    await demo_compute_topk()
    await demo_knowledge()


if __name__ == "__main__":
    asyncio.run(main())
