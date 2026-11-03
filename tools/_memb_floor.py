# -*- coding: utf-8 -*-
"""量 membership 的真实"地板"：GT 类别偏度 → 恒答多数类的准确率（无信号可达的 acc 下限），
再对每条决策路径的实测 acc 比较，判断哪条低于地板、哪条是反向预测器。
用法: .venv\\Scripts\\python.exe tools\\_memb_floor.py
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from _v16_diag import build_signals  # noqa: E402


def path(r):
    if r["judge"]:
        return "judge"
    return "heuristic" if not r["used_llm"] else "llm-chief"


def main():
    rows, _ = build_signals()          # v16 tv=16
    memb = [r for r in rows if r["bucket"] == "membership"]
    # GT 类别分布（正=essential True）
    pos = sum(1 for r in memb if bool((r["gt"] or {}).get("answer")))
    tot = len(memb)
    print(f"# membership 题数={tot}  GT True={pos} ({pos/tot:.1%})  False={tot-pos} ({(tot-pos)/tot:.1%})")
    floor = max(pos, tot - pos) / tot
    print(f"# 无信号地板：恒答多数类 acc = {floor:.1%}（{'True' if pos>=tot-pos else 'False'}-majority）")

    g = defaultdict(lambda: [0, 0])    # path -> [correct, answered]
    for r in memb:
        if r["outcome"] == "abstain":
            continue
        g[path(r)][1] += 1
        g[path(r)][0] += 1 if r["outcome"] == "correct" else 0
    print(f"\n{'path':10s} {'n':>3} {'acc':>6}  vs地板")
    for p, (c, n) in sorted(g.items(), key=lambda kv: -(kv[1][0]/kv[1][1] if kv[1][1] else 0)):
        a = c / n
        flag = "↑高于地板" if a > floor else ("≈chance" if abs(a-0.5) < 0.06 else "↓低于地板(可反向)")
        print(f"{p:10s} {n:>3} {a:>6.1%}  {flag}")

    pw = [r for r in rows if r["bucket"] == "pairwise"]
    pwc = sum(1 for r in pw if r["outcome"] == "correct")
    pwa = sum(1 for r in pw if r["outcome"] in ("correct", "wrong"))
    print(f"\n# pairwise：地板=50%（二选一）；管线实测 acc(已答)={pwc/pwa:.1%} n={pwa}")


if __name__ == "__main__":
    main()
