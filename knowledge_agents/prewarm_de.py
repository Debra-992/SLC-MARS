# -*- coding: utf-8 -*-
"""DESeq2 结果预热：把"每癌种只付一次代价"的那一次代价，放到正式评测之前完成。

背景（证据）：
  - slc_backend.RealMcpBackend 每次工具调用都新起一个 stdio MCP 进程，R 侧 DESeq2 结果
    没有跨题缓存；GDCdata 里只有 BRCA / COAD 具备 raw_counts.csv + metadata.csv（肿瘤/正常配对），
    其余癌种会很快返回 download_required / unsupported_cancer。
  - ComputationTools.differential_expression 已加磁盘缓存（de_workspace_cache/*.json），
    只缓存确定性终态。本脚本触发每个癌种的首次计算把缓存填满，正式评测即可即时复用 -> DE 通道真正参与。

用法（必须在 run_route_b 设好 R / GDCdata / cache root 等环境后运行）：
  python -m knowledge_agents.prewarm_de                # 预热 clean 集出现过的全部癌种
  python -m knowledge_agents.prewarm_de --cancers "Breast Cancer,Colon Cancer"
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os

from knowledge_agents.tools.computation_tools import ComputationTools


def _dataset_cancers() -> list[str]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "questions_clean.jsonl")
    seen: "collections.Counter[str]" = collections.Counter()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            cancer = (rec.get("constraints") or {}).get("cancer", "")
            if cancer:
                seen[cancer] += 1
    return list(seen)


async def _prewarm(cancers: list[str]) -> None:
    tools = ComputationTools()
    top_n = int(os.environ.get("SLC_DE_TOPN", "400"))
    real_runs = 0
    for cancer in cancers:
        try:
            de = await tools.differential_expression(cancer, top_n=top_n)
        except Exception as e:  # noqa BLE001
            print(f"[prewarm] {cancer:<28} EXC {type(e).__name__}: {e}")
            continue
        n_top = len(de.top_genes or [])
        hit = "REAL" if de.status == "ok" else de.status
        if de.status == "ok":
            real_runs += 1
        note = ""
        if de.status != "ok":
            # 失败时直接取后端原始回执里的 error（本地已有计数文件，重算很快），便于定位真实原因
            try:
                raw = await tools.backend.acall("de", {"cancer_type": cancer, "gene_family": "SLC", "top_n": top_n})
                note = " | err={}".format(str(raw.get("error"))[:160])
            except Exception as e:  # noqa BLE001
                note = " | err<{}>".format(type(e).__name__)
        print(f"[prewarm] {cancer:<28} status={hit:<20} significant={de.significant_count} top_genes={n_top}{note}")
    print(f"[prewarm] 完成：{real_runs} 个癌种产生了真实 DESeq2 数值（已落盘缓存，评测即时复用）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancers", default="", help="逗号分隔的癌种列表；留空则用 clean 数据集出现过的全部癌种")
    args = ap.parse_args()
    cancers = [c.strip() for c in args.cancers.split(",") if c.strip()] or _dataset_cancers()
    print(f"[prewarm] 待预热癌种 {len(cancers)} 个；DE 缓存开关 SLC_DE_CACHE={os.environ.get('SLC_DE_CACHE','1')}, "
          f"DE 超时 SLC_DE_TIMEOUT={os.environ.get('SLC_DE_TIMEOUT','900')}s")
    asyncio.run(_prewarm(cancers))


if __name__ == "__main__":
    main()
