# -*- coding: utf-8 -*-
"""R2 零噪声隔离评估（离线·免端点·确定性）：
R2 只改 used_llm=False 的启发式 membership 行；这些行的 basis+gene_scores+v16答案 全持久化。
=> 拿 v16 自己的启发式 membership 行，离线用烘焙 R2 重算答案，逐题对比 v16答案 vs R2答案 vs GT。
无任何 LLM 参与 => 无重跑抖动，得到 R2 在它目标子集上的真实净效应。
用法: .venv\\Scripts\\python.exe tools\\_r2_isolate.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from _r2_fit import feats  # noqa: E402  (conflict,dir_present,solo,buffered,mag)
from _v16_diag import load_jsonl  # noqa: E402
from knowledge_agents.baselines.common import TRACES  # noqa: E402

RW = [1.732673, -1.733301, 2.634866, -0.619402, 1.934390]
RB = -0.408812


def r2_answer(basis: set, strength: float) -> bool:
    z = sum(w * f for w, f in zip(RW, feats(basis, strength))) + RB
    return (1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))) > 0.5


def main():
    lat = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        if str((r.get("provenance") or {}).get("trace_version") or "") == "16":
            lat[r["qid"]] = r

    def is_heur(a):
        dm = str(a.get("decision_mode") or "")
        return (not a.get("used_llm")) and ("judge" not in dm) and ("法官" not in str(a.get("rationale") or ""))

    memb_heur = []
    for q, r in lat.items():
        if r.get("bucket") != "membership":
            continue
        a = ((r.get("systems") or {}).get("pipeline_auto") or {}).get("answer") or {}
        if is_heur(a):
            memb_heur.append((r, a))

    gt_ess = lambda r: bool((r.get("ground_truth") or {}).get("answer"))
    w2c = c2w = still_wrong = abstain_saved = abstain_to_wrong = 0
    rows = []
    for r, a in memb_heur:
        basis = set(a.get("basis") or [])
        gs = a.get("gene_scores") or {}
        strength = max(gs.values()) if gs else 0.0
        gt = gt_ess(r)
        v16_ans = a.get("answer")            # True/False/None(弃)
        abstained = bool(a.get("abstained"))
        new = r2_answer(basis, strength)
        v16_ok = (not abstained) and (v16_ans is not None) and (bool(v16_ans) == gt)
        new_ok = (new == gt)
        if v16_ok and not new_ok:
            c2w += 1
        elif (not v16_ok) and new_ok:
            w2c += 1
            if abstained:
                abstain_saved += 1
        elif not v16_ok and not new_ok:
            still_wrong += 1
        if abstained and not new_ok:
            abstain_to_wrong += 1
        rows.append((r["qid"][:44], abstained, v16_ans, new, gt, v16_ok, new_ok))

    n = len(memb_heur)
    print(f"# v16 启发式 membership 行（R2 会接管的全部题） n={n}")
    print(f"  wrong/abstain -> correct (w2c) = {w2c}  [其中救回原弃权 {abstain_saved}]")
    print(f"  correct -> wrong        (c2w) = {c2w}")
    print(f"  仍错                       = {still_wrong}")
    print(f"  R2 净效应（membership 计数，±题数）= {w2c - c2w:+d}")
    print(f"  换算 acc(全体,分母100) 增量 ≈ {(w2c - c2w)/100:+.1%}  "
          f"（acc(已答)口径另计：R2 还消除 {abstain_saved - abstain_to_wrong} 道弃权）")
    print()
    print(f"  {'qid':44s} {'v16弃':>5} {'v16答':>5} {'R2答':>5} {'GT':>4} {'v16✓':>5} {'R2✓':>4}")
    for qid, ab, va, new, gt, vok, nok in rows:
        print(f"  {qid:44} {'弃' if ab else '  ':>5} {str(va):>5} {str(new):>5} {str(gt):>4} {'Y' if vok else '.':>5} {'Y' if nok else '.':>4}")


if __name__ == "__main__":
    main()
