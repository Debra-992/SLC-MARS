# -*- coding: utf-8 -*-
"""分析 tracestore 里答错的题目：方向偏差 / 证据通道 / 置信度 / 类比把弃答变错答 等。"""
from __future__ import annotations
import json
from collections import Counter

st = [json.loads(l) for l in open('outputs/traces/tracestore.jsonl', encoding='utf-8') if l.strip()]
SYS = ("pipeline_heuristic", "pipeline_auto", "baseline")


def ans(e, s):
    return (e.get("systems", {}).get(s, {}) or {}).get("answer") or {}


def outcome(e, s):
    return (e.get("systems", {}).get(s, {}) or {}).get("outcome", "abstain")


print("total traces:", len(st), " buckets:", Counter(e['bucket'] for e in st))

for s in SYS:
    wrong = [e for e in st if outcome(e, s) == "wrong"]
    print(f"\n{'='*70}\n系统 {s}: 答错 {len(wrong)} / {len(st)}")
    print("  wrong by bucket:", Counter(e['bucket'] for e in wrong))

    # membership 预测方向偏差
    mem_w = [e for e in wrong if e['bucket'] == 'membership']
    pred_true = sum(1 for e in mem_w if ans(e, s).get('answer') is True)
    pred_false = sum(1 for e in mem_w if ans(e, s).get('answer') is False)
    print(f"  membership 错答方向: 预测True={pred_true} 预测False={pred_false}")

    # 证据通道（basis）在错答里的构成
    chan = Counter()
    for e in wrong:
        b = ans(e, s).get('basis') or []
        chan['(无证据通道)'] += 1 if not b else 0
        for t in b:
            chan[t] += 1
    print("  错答用到的证据通道计数:", dict(chan))

    # 置信度分布（错 vs 对）
    correct_c = [ans(e, s).get('confidence', 0) for e in st if outcome(e, s) == 'correct']
    wrong_c = [ans(e, s).get('confidence', 0) for e in wrong]
    def stat(v):
        return f"n={len(v)} mean={sum(v)/len(v):.2f}" if v else "n=0"
    print(f"  置信度 对:{stat(correct_c)}  错:{stat(wrong_c)}")

# 针对管线 auto：类比把 heuristic 弃答 -> auto 错答 的数量
print("\n" + "=" * 70)
flip_wrong = [e for e in st if outcome(e, 'pipeline_heuristic') == 'abstain'
              and outcome(e, 'pipeline_auto') == 'wrong']
flip_correct = [e for e in st if outcome(e, 'pipeline_heuristic') == 'abstain'
                and outcome(e, 'pipeline_auto') == 'correct']
print(f"auto 类比救回弃答: 变对={len(flip_correct)} 变错={len(flip_wrong)}")

# auto 错答里 decision_mode 分布（是接地启发式错的，还是 LLM 类比错的）
auto_w = [e for e in st if outcome(e, 'pipeline_auto') == 'wrong']
print("auto 错答的 decision_mode:", Counter(ans(e, 'pipeline_auto').get('decision_mode') for e in auto_w))
print("auto 错答里 used_llm 比例:", sum(1 for e in auto_w if ans(e, 'pipeline_auto').get('used_llm')), "/", len(auto_w))

# 错答里：目标基因是否根本没进 PPR 候选（纯靠 kg_neighbor 弱信号）
print("\n--- 深挖 auto 错答的 gene_scores（前 12）---")
for e in auto_w[:12]:
    a = ans(e, 'pipeline_auto')
    gs = a.get('gene_scores') or {}
    top = max(gs, key=gs.get) if gs else None
    pred = a.get('winner') or a.get('answer')
    gt = e['ground_truth']
    print(f"  {e['qid'][:46]:<46} {e['bucket']:<9} pred={pred} gt={gt} scores={gs} basis={a.get('basis')} mode={a.get('decision_mode')}")
