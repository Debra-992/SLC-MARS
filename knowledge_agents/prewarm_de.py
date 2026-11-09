# -*- coding: utf-8 -*-
"""DESeq2 warm-up: pay the "once per cancer" cost before the scored run instead of during it.

Why this exists (the evidence):
  - slc_backend.RealMcpBackend starts a fresh stdio MCP process for every tool call, so nothing
    caches the R-side DESeq2 result across questions; of the GDCdata trees only BRCA and COAD ship
    raw_counts.csv + metadata.csv (tumour/normal pairs), so every other cancer returns
    download_required / unsupported_cancer quickly.
  - ComputationTools.differential_expression already has a disk cache (de_workspace_cache/*.json)
    that stores only deterministic terminal states. This script triggers the first computation for
    each cancer and fills that cache, so the scored run reuses it instantly and the DE channel
    actually participates.

Usage (run only after run_route_b has set the R / GDCdata / cache-root environment):
  python -m knowledge_agents.prewarm_de                # every cancer present in the clean set
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
            # On failure, read the backend's own error string (the count files are already local, so
            # recomputing is cheap) to make the real cause diagnosable.
            try:
                raw = await tools.backend.acall("de", {"cancer_type": cancer, "gene_family": "SLC", "top_n": top_n})
                note = " | err={}".format(str(raw.get("error"))[:160])
            except Exception as e:  # noqa BLE001
                note = " | err={}".format(type(e).__name__)
        print(f"[prewarm] {cancer:<28} status={hit:<20} significant={de.significant_count} top_genes={n_top}{note}")
    print(f"[prewarm] done: {real_runs} cancer(s) produced real DESeq2 numbers (cached on disk, reused by the scored run)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancers", default="", help="comma-separated cancer list; empty = every cancer in the clean dataset")
    args = ap.parse_args()
    cancers = [c.strip() for c in args.cancers.split(",") if c.strip()] or _dataset_cancers()
    print(f"[prewarm] {len(cancers)} cancer(s) to warm; DE cache switch SLC_DE_CACHE={os.environ.get('SLC_DE_CACHE','1')}, "
          f"DE timeout SLC_DE_TIMEOUT={os.environ.get('SLC_DE_TIMEOUT','900')}s")
    asyncio.run(_prewarm(cancers))


if __name__ == "__main__":
    main()
