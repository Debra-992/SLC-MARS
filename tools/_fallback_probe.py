# -*- coding: utf-8 -*-
"""离线验证"证据不足→改用 LLM 架构能力作答"能否拉高 acc(全体)。

思路：管线里【启发式独判 membership】(used_llm=False, 非judge) 那批题 = 证据不足/最烂路径(实测20%)。
同一批 qid 上，autogen / mdagents / pure-llm 的答案已持久化，直接算它们在这些题上的正确率，
即"把这条路的最终裁决让给 LLM 架构"的期望 uplift。全程只读，免端点。
用法: .venv\\Scripts\\python.exe tools\\_fallback_probe.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from _v16_diag import build_signals  # noqa: E402
from knowledge_agents.baselines.common import TRACES, GT_PATH, INPUT_PATH  # noqa: E402
from knowledge_agents.baselines.merge_baselines import load_jsonl, outcome_of  # noqa: E402

GT = json.loads(GT_PATH.read_text(encoding="utf-8"))
INPUTS = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
BUCKET = {i["qid"]: i["bucket"] for i in INPUTS}


def baseline_answers(name):
    seen = {}
    p = TRACES / f"answers_{name.replace('-', '_')}.jsonl"
    if p.exists():
        for rr in load_jsonl(p):
            seen[rr["qid"]] = rr
    return seen


def acc_on(name, qids):
    seen = baseline_answers(name)
    c = w = ab = miss = 0
    for q in qids:
        rr = seen.get(q)
        if rr is None:
            miss += 1
            continue
        o = outcome_of(rr, GT.get(q, {}), BUCKET.get(q, ""))
        if o == "correct":
            c += 1
        elif o == "wrong":
            w += 1
        else:
            ab += 1
    answered = c + w
    return c, w, ab, miss, answered


def main():
    rows, _ = build_signals()
    memb = [r for r in rows if r["bucket"] == "membership"]

    heur_q = [r["q"] for r in memb if (not r["used_llm"]) and (not r["judge"]) and r["outcome"] != "abstain"]
    heur_correct = sum(1 for r in memb if r["q"] in heur_q and r["outcome"] == "correct")
    pip_wrong_q = [r["q"] for r in memb if r["outcome"] == "wrong"]
    all_memb_q = [r["q"] for r in memb]

    print(f"# 目标子群：管线·启发式独判 membership n={len(heur_q)}  (管线自己只对 {heur_correct}/{len(heur_q)})")
    for name in ("autogen", "mdagents", "pure-llm", "lit-rag"):
        c, w, ab, miss, answered = acc_on(name, heur_q)
        acc_a = c / answered if answered else 0.0
        print(f"    LLM架构[{name:9s}] 在该{len(heur_q)}题: 对={c} 错={w} 弃={ab} 缺={miss}  | acc(已答)={acc_a:.1%}")

    print(f"\n# 对照·更宽的子群")
    print(f"    管线 membership 答错 {len(pip_wrong_q)} 题 → LLM 能救回多少：")
    for name in ("autogen", "mdagents", "pure-llm"):
        c, w, ab, miss, answered = acc_on(name, pip_wrong_q)
        print(f"      [{name:9s}] 对={c}/{len(pip_wrong_q)} ({c/len(pip_wrong_q):.0%})  ← 这些是管线已答错的题")
    print(f"\n    全部 membership {len(all_memb_q)} 题各系统 acc(已答)：")
    for name in ("autogen", "mdagents", "pure-llm"):
        c, w, ab, miss, answered = acc_on(name, all_memb_q)
        acc_a = c / answered if answered else float("nan")
        print(f"      [{name:9s}] acc(已答)={acc_a:.1%} 覆盖率={answered/len(all_memb_q):.1%} 缺={miss}")


if __name__ == "__main__":
    main()
