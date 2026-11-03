# -*- coding: utf-8 -*-
"""R2 可行性体检（离线·免端点）：必需性相关信号对 53 道 membership 有没有【区分度】。

用持久化 tracestore 里 v16 membership 每条 answer 的 basis(通道名集合)+gene_scores(净分)
+ GT(answer bool=必需)，交叉表看：
 - 净分符号(net>0) 与 必需 的相关（启发式就是靠它，20% 说明可能反向）；
 - 极性矛盾(kg_direction & kg_direction_against 并存) 在 必需/非必需 组的占比；
 - paralog 信号 / ppr / timer / de 在各组的占比。
若某特征在两组占比差异大 → 有区分度、R2 有料；若处处近乎恒定 → R2 退化成先验，否。
用法: .venv\\Scripts\\python.exe tools\\_r2_discrim.py
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from knowledge_agents.baselines.common import TRACES  # noqa: E402
from knowledge_agents.baselines.merge_baselines import load_jsonl  # noqa: E402


def main():
    lat = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        if str((r.get("provenance") or {}).get("trace_version") or "") != "16":
            continue
        lat[r["qid"]] = r            # 同 qid 后写覆盖，去重
    pos = defaultdict(int)   # feature -> count where GT essential
    neg = defaultdict(int)   # feature -> count where GT non-essential
    netpos = {"essential": [0, 0], "nonessential": [0, 0]}  # [net>0, net<0]
    npos = nneg = 0
    for r in lat.values():
        if r.get("bucket") != "membership":
            continue
        s = (r.get("systems", {}) or {}).get("pipeline_auto") or {}
        a = s.get("answer") or {}
        gt = r.get("ground_truth") or {}
        ess = bool(gt.get("answer"))
        basis = set(a.get("basis") or [])
        gs = a.get("gene_scores") or {}
        net = max(gs.values()) if gs else 0.0
        npos += 1 if ess else 0
        nneg += 0 if ess else 1
        feats = set(basis)
        if {"kg_direction", "kg_direction_against"} <= basis:
            feats.add("POLARITY_CONTRADICTION")
        if {"paralog_redundancy", "paralog_buffered"} & basis:
            feats.add("HAS_PARALOG")
        if "ppr" in basis or "ppr_rank" in basis:
            feats.add("HAS_PPR")
        feats |= {f for f in ("timer", "de", "km") if f in basis}
        bucket = "essential" if ess else "nonessential"
        for f in feats:
            (pos if ess else neg)[f] += 1
        netpos[bucket][0] += 1 if net > 0 else 0
        netpos[bucket][1] += 1 if net < 0 else 0

    print(f"# membership n={npos+nneg}  必需(True)={npos}  非必需(False)={nneg}")
    print(f"\n净分符号分布（启发式即按 net>0 判必需）：")
    for grp, (p, m) in netpos.items():
        print(f"  GT={grp:12s} net>0:{p:2d}  net<0:{m:2d}")
    print(f"\n{'feature':24s} {'必需组占比':>12} {'非必需组占比':>12} {'差':>8}")
    allf = sorted(set(pos) | set(neg))
    for f in allf:
        pp = pos[f] / npos if npos else 0
        pn = neg[f] / nneg if nneg else 0
        print(f"  {f:22s} {pp:>11.0%} {pn:>12.0%} {pp-pn:>+8.0%}")
    print("\n判读：|差| 越大 → 该特征越能区分必需/非必需；全 ~0 → R2 无料、退化先验。")


if __name__ == "__main__":
    main()
