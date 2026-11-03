# -*- coding: utf-8 -*-
"""快速验证 8 个 SLC 工具全部可达（绕过 MetaGPT 层，直接打 RealMcpBackend）。"""
from __future__ import annotations

import asyncio
import json

from knowledge_agents.tools.slc_backend import get_default_backend


async def probe(backend, tool: str, kwargs: dict):
    try:
        r = await backend.acall(tool, kwargs)
        status = r.get("status") if isinstance(r, dict) else "?"
        sample = json.dumps(r, ensure_ascii=False, default=str)
        if len(sample) > 220:
            sample = sample[:220] + "..."
        print(f"  [{tool:<26}] status={status:<20} {sample}")
        return status
    except Exception as e:
        print(f"  [{tool:<26}] EXCEPTION: {type(e).__name__}: {e}")
        return "exception"


async def main():
    backend = get_default_backend()
    print(f"backend: {type(backend).__name__}")
    print("=== 8 SLC tools connectivity ===")
    # 4 retrieval
    await probe(backend, "project_knowledge",        {"qid": "pairwise_negative_CRC_Human_HCT116_SLC16A1_KO_00179"})
    await probe(backend, "kg_schema",                {})
    await probe(backend, "kg_neighbors",             {"gene": "SLC16A1", "limit": 5})
    await probe(backend, "kg_cypher",                {
        "query": "MATCH (g:SLCGene {name:'SLC16A1'})-[r]->(x) RETURN type(r), labels(x), x.name LIMIT 5",
        "limit": 5,
    })
    # 4 compute
    await probe(backend, "project_graph_computation",{"qid": "pairwise_negative_CRC_Human_HCT116_SLC16A1_KO_00179"})
    await probe(backend, "timer",                    {"gene": "SLC25A1", "gene_family": "SLC"})
    await probe(backend, "km",                       {"gene": "SLC25A1", "cancer_type": "COAD"})
    await probe(backend, "de",                       {"cancer_type": "COAD", "gene_family": "SLC", "top_n": 5})


if __name__ == "__main__":
    asyncio.run(main())
