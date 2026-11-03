# -*- coding: utf-8 -*-
"""知识检索 / 计算 两个智能体的演示入口。

后端选择（按 env 自动）：
  - 未设 SLC_ROOT                -> UnavailableBackend（离线，全未知，验证不伪造）
  - 设 SLC_ROOT + 装 mcp         -> RealMcpBackend（走 SLC 真实 stdio MCP 服务）
接真实 MCP 前需要设好：SLC_ROOT / SLC_MCP_PYTHON / SLC_KMTIMER_ROOT / 数据库 env。

演示覆盖：
  demo_compute      pairwise 桶 → TIMER + KM 命中缓存；DESeq2 download_required
  demo_compute_topk topk 桶     → 追加触发 PPR（需远程 Neo4j 可达）
  demo_knowledge    白名单 qid  → project_knowledge（KG receipts + literature）
"""
from __future__ import annotations

import asyncio

from metagpt.logs import logger

from knowledge_agents.roles.compute_role import ComputeRole
from knowledge_agents.roles.knowledge_query_role import KnowledgeQueryRole
from knowledge_agents.schema import parse_knowledge


async def demo_compute():
    """pairwise 路由：TIMER + KM 命中缓存；DESeq2 无计数返回 download_required。"""
    role = ComputeRole()
    # SLC25A1/COAD：km_plots 里有 SLC25A1_colon_km_stats.txt，timer_cache 里有 boxplot_SLC25A1.png
    role.set_task(gene="SLC25A1", cancer="COAD", task_bucket="pairwise")
    await role.run("gene=SLC25A1 cancer=COAD 请给出表达差异证据与患者生存关联")
    ck = parse_knowledge(role.rc.memory.get()[-1].content)
    logger.info(f"[候选知识 · pairwise] status={ck.status.value}")
    if ck.expression:
        logger.info(f"  TIMER: status={ck.expression.status.value} scope={ck.expression.evidence_scope} "
                    f"boxplot={ck.expression.boxplot_path}")
    if ck.survival:
        stats = (ck.survival.stats_text or "").splitlines()
        stats_head = stats[0] if stats else ""
        logger.info(f"  KM    : status={ck.survival.status.value} scope={ck.survival.evidence_scope} "
                    f"stats_head={stats_head!r}")
    logger.info(f"  {ck.summary}")


async def demo_compute_topk():
    """topk 路由：追加触发 PPR，需远程 Neo4j 可达；SLC16A1 无 TIMER 缓存，
    命中 cache-miss 属预期，验证 evidence-integrity 不伪造。"""
    role = ComputeRole()
    role.set_task(qid="pairwise_negative_CRC_Human_HCT116_SLC16A1_KO_00179",
                  gene="SLC16A1", cancer="COAD", task_bucket="topk")
    await role.run("gene=SLC16A1 cancer=COAD topk 排序 SLC 家族候选，评估邻近度、表达差异与生存关联")
    ck = parse_knowledge(role.rc.memory.get()[-1].content)
    logger.info(f"[候选知识 · topk] status={ck.status.value}")
    if ck.graph_computation:
        gc = ck.graph_computation
        top = (gc.candidates or [])[:5]
        top_str = ", ".join(f"{c.gene}={c.degree_corrected_score}" for c in top) if top else "无"
        logger.info(f"  PPR   : status={gc.status.value} seeds={gc.seed_count} "
                    f"nodes={gc.node_count} edges={gc.edge_count}  top5=[{top_str}]")
    if ck.expression:
        logger.info(f"  TIMER : status={ck.expression.status.value}")
    if ck.survival:
        stats = (ck.survival.stats_text or "").splitlines()
        logger.info(f"  KM    : status={ck.survival.status.value} stats_head={stats[0] if stats else ''!r}")
    logger.info(f"  {ck.summary}")


async def demo_knowledge():
    """白名单 qid：走 project_knowledge。有 Neo4j 时 3 个 KG receipts 应 ok；
    有 Mongo papers 时 literature 也应 ok。"""
    role = KnowledgeQueryRole()
    await role.run("qid=pairwise_negative_CRC_Human_HCT116_SLC16A1_KO_00179 检索 SLC16A1 结直肠癌背景知识")
    kg = parse_knowledge(role.rc.memory.get()[-1].content)
    logger.info(f"[背景知识] status={kg.status.value} 候选={kg.candidates} 证据数={len(kg.evidence)}")
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
