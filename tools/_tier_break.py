# -*- coding: utf-8 -*-
"""按决策 tier 分组看真实正确率与 margin，验证 A_TIER<0 是不是数据里真有此规律。"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from _v16_diag import build_signals  # noqa: E402


def tiername(r):
    if r["agree"]:
        return "agree"
    if r["judge"]:
        return "judge"
    return "heuristic" if not r["used_llm"] else "llm-chief"


def main():
    rows, _ = build_signals()
    g = defaultdict(lambda: [0, 0, 0.0])  # tier -> [correct, total, sum_margin]
    for r in rows:
        if r["outcome"] == "abstain":
            continue
        t = tiername(r)
        g[t][1] += 1
        g[t][0] += 1 if r["outcome"] == "correct" else 0
        g[t][2] += r["margin"]
    print(f"{'tier':10s} {'n':>3} {'acc':>6} {'mean_margin':>11}")
    for t, (c, tot, sm) in sorted(g.items(), key=lambda kv: -kv[1][0]/kv[1][1]):
        print(f"{t:10s} {tot:>3} {c/tot:>6.1%} {sm/tot:>11.2f}")


if __name__ == "__main__":
    main()
