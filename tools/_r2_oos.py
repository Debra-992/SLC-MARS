# -*- coding: utf-8 -*-
"""v20 扩 500 后的 R2 out-of-sample 迁移判决 + 500 pipeline 终表（离线·免端点）。

R2 系数是在 fit100 里的 53 道 membership 拟合的。本脚本用全池 tv20 行：
 1) 把 membership 分成 fit(∈fit100) vs OOS(∉) —— OOS 是从未参与拟合的新题；
 2) 用烘焙 R2 对每题现算必需性答案，分别报两组的：类别地板(恒答多数)、R2 单独 acc、
    区分度(预测非必需数+命中)。**OOS R2 acc > OOS 地板 且仍判非必需 → R2 迁移成立**；
    ≈ 地板/退化 → in-dist 过拟合，R2 迁移存疑（诚实标注）。
 3) 另报 OOS 里"真走了启发式兜底(used_llm=False 无 judge)"的题——R2 在那些题上是真实决策，
    看其实测 acc（≈ 冠军里 R2 的 realized 收益）。
 4) 500 终表：pipeline_auto(v16+R2) / pipeline_heuristic / native baseline 的 acc(全体) 分桶。
用法: .venv\\Scripts\\python.exe tools\\_r2_oos.py
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from _r2_fit import feats  # noqa: E402
from knowledge_agents.baselines.common import TRACES  # noqa: E402
from knowledge_agents.baselines.merge_baselines import load_jsonl  # noqa: E402

RW = [1.732673, -1.733301, 2.634866, -0.619402, 1.934390]
RB = -0.408812


def r2_ess(basis: set, strength: float) -> bool:
    z = sum(w * f for w, f in zip(RW, feats(basis, strength))) + RB
    return (1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))) > 0.5


def load_tv(version: str):
    lat = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        if str((r.get("provenance") or {}).get("trace_version") or "") == version:
            lat[r["qid"]] = r
    return lat


def acc_block(rows):
    n = len(rows)
    if not n:
        return None
    ess = sum(1 for r in rows if r["gt"])
    floor = max(ess, n - ess) / n
    r2_ok = sum(1 for r in rows if r["r2"] == r["gt"])
    pred_neg = sum(1 for r in rows if not r["r2"])
    neg_ok = sum(1 for r in rows if not r["r2"] and not r["gt"])
    return dict(n=n, ess=ess, floor=floor, r2acc=r2_ok / n,
                pred_neg=pred_neg, neg_ok=neg_ok,
                pos_ok=sum(1 for r in rows if r["r2"] and r["gt"]),
                pos_pred=sum(1 for r in rows if r["r2"]))


def main():
    fit = set(json.loads((TRACES / "fit100_qids.json").read_text(encoding="utf-8"))["qids"])
    lat = load_tv("20")
    M = []
    for q, r in lat.items():
        if r.get("bucket") != "membership":
            continue
        a = ((r.get("systems") or {}).get("pipeline_auto") or {}).get("answer") or {}
        gs = a.get("gene_scores") or {}
        strength = max(gs.values()) if gs else 0.0
        basis = set(a.get("basis") or [])
        dm = str(a.get("decision_mode") or "")
        M.append(dict(q=q, gt=bool((r.get("ground_truth") or {}).get("answer")),
                      r2=r2_ess(basis, strength),
                      fallback=(not a.get("used_llm")) and ("judge" not in dm) and ("法官" not in str(a.get("rationale") or ""))))
    fit_m = [r for r in M if r["q"] in fit]
    oos_m = [r for r in M if r["q"] not in fit]
    fb_oos = [r for r in oos_m if r["fallback"]]

    print(f"# tv20 membership 全池 n={len(M)}  (fit={len(fit_m)}  OOS={len(oos_m)}  OOS启发式兜底={len(fb_oos)})")
    print(f"# 注：fit 拟合自 v16@100 的 53 题；若 OOS 行数偏少=500 臂尚未跑完/未覆盖\n")

    for label, blk in (("fit(已参与拟合)", acc_block(fit_m)),
                       ("OOS(全新题)", acc_block(oos_m)),
                       ("OOS·R2真兜底题", acc_block(fb_oos))):
        if not blk:
            print(f"  {label:16s} 无数据"); continue
        print(f"  {label:16s} n={blk['n']:3d}  地板(恒多数)={blk['floor']:.1%}  "
              f"R2 acc={blk['r2acc']:.1%}  Δ地板={blk['r2acc']-blk['floor']:+.1%}  "
              f"判非必需 {blk['pred_neg']} 中命中 {blk['neg_ok']} | 判必需 {blk['pos_pred']} 中命中 {blk['pos_ok']}")

    fb = acc_block(oos_m)
    if fb:
        print()
        if fb["r2acc"] > fb["floor"] and fb["neg_ok"] >= 3:
            print(">>> 迁移判定：PASS —— R2 在 OOS 新题上仍超类别地板且能正确判非必需（区分度迁移，非过拟合先验）")
        elif fb["r2acc"] <= fb["floor"]:
            print(">>> 迁移判定：NEG —— R2 OOS acc ≤ 地板，in-dist 66.8% 过拟合，泛化不成立")
        else:
            print(">>> 迁移判定：边缘 —— R2 OOS 略超地板但区分度弱，需更大 n 复核")

    # 500 pipeline 终表（acc 全体 = correct/total）
    print("\n===== 500 池 pipeline 终表（acc全体, 分母=tv20 行）=====")
    for sysname in ("pipeline_auto", "pipeline_heuristic", "baseline"):
        per = {}
        for q, r in lat.items():
            o = ((r.get("systems") or {}).get(sysname) or {}).get("outcome") or "abstain"
            per[q] = o if o in ("correct", "wrong", "abstain") else "abstain"
        for b in ("ALL", "pairwise", "membership"):
            keep = [q for q, r in lat.items() if b == "ALL" or r.get("bucket") == b]
            if not keep:
                continue
            c = sum(1 for q in keep if per[q] == "correct")
            print(f"  {sysname:18s} {b:10s} n={len(keep):3d}  acc全体={c/len(keep):.1%}")


if __name__ == "__main__":
    main()
