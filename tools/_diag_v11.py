# -*- coding: utf-8 -*-
"""v11 诊断：证据通道 vs prior 通道的条件正确率 / 冲突裁决复盘。

回答三个问题：
  1) 一致/冲突/fallback 三种状态下的准确率各是多少（分桶）？
  2) 冲突时"证据优先"规则损失了多少题（如果听 prior 会怎样）？
  3) prior 通道缺席率（真实先验覆盖 vs 名义 fallback 次数差）。
"""
import glob
import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rows = []
for p in glob.glob(r"<PROJECT_ROOT>\outputs\traces\by_question\*.json"):
    d = json.load(open(p, encoding="utf-8"))
    if str((d.get("provenance") or {}).get("trace_version")) != "11":
        continue
    for s in ("pipeline_heuristic", "pipeline_auto"):
        a = (d["systems"][s].get("answer") or {})
        r = a.get("rationale") or ""
        dm = a.get("decision_mode") or ""
        outcome = d["systems"][s]["outcome"]
        if "+prior" in dm:
            state = "fallback"          # 证据弃答，采纳prior
        elif "prior·v11 一致" in r:
            state = "agree"             # 证据&prior同向
        elif "冲突未仲裁" in r:
            state = "conflict_evid_wins"
        elif "专家辩论·v11 确认" in r:
            state = "agree_debate_conf"  # 已答+辩论确认（prior可能缺席）
        else:
            state = "no_prior"
        rows.append((s, d["bucket"], state, outcome))

def acc(sel):
    c = Counter(o for st, o in sel if o in ("correct", "wrong", "abstain"))
    n = sum(c.values()); ans = c["correct"] + c["wrong"]
    return f"n={n:>2} 答={ans:>2} 对={c['correct']:>2} 错={c['wrong']:>2} 弃={c['abstain']:>1} acc_all={(c['correct']/n if n else 0):.1%}"

# 简化重写
from collections import defaultdict
agg = defaultdict(list)
for s, b, state, outcome in rows:
    agg[(s, state)].append((b, outcome))
    agg[("ALL", state)].append((b, outcome))

print("\n== 融合状态复盘（两臂合并 + 分臂） ==")
for key in sorted(agg, key=lambda k: (k[0], k[1])):
    s, st = key
    items = agg[key]
    n = len(items)
    per_bucket = {}
    for b in ("pairwise", "membership"):
        sub = [o for bb, o in items if bb == b]
        if sub:
            cc = sub.count("correct"); ww = sub.count("wrong")
            per_bucket[b] = f"{cc}/{cc+ww}" if (cc+ww) else "-"
    c = sum(1 for _, o in items if o == "correct")
    w = sum(1 for _, o in items if o == "wrong")
    ab = n - c - w
    print(f"{s:<18} {st:<19} n={n:>2} 对={c:>2} 错={w:>2} 弃={ab:>1}  pairwise对/答={per_bucket.get('pairwise','-'):<6} membership={per_bucket.get('membership','-')}")

# fallback 的 prior 质量参照：pure-llm 在同题集上的表现
pure = {}
for line in open(r"<PROJECT_ROOT>\outputs\traces\answers_pure_llm.jsonl", encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        pure[r["qid"]] = r   # 后写覆盖先写（同 baseline 口径）
gt = json.load(open(r"<PROJECT_ROOT>\outputs\traces\baselines_gt.json", encoding="utf-8"))
inputs = {r["qid"]: r for r in json.load(open(r"<PROJECT_ROOT>\outputs\traces\baselines_input.json", encoding="utf-8"))}

# fallback 题上：同一题 pure-llm baseline 对不对？（同一记忆体、不同置信阈值路径）
fb_qids = []
for p in glob.glob(r"<PROJECT_ROOT>\outputs\traces\by_question\*.json"):
    d = json.load(open(p, encoding="utf-8"))
    if str((d.get("provenance") or {}).get("trace_version")) != "11":
        continue
    for s in ("pipeline_heuristic", "pipeline_auto"):
        a = (d["systems"][s].get("answer") or {})
        if "+prior" in (a.get("decision_mode") or ""):
            fb_qids.append((s, d["qid"], d["bucket"], d["systems"][s]["outcome"]))
c2 = Counter()
for s, q, b, o in fb_qids:
    grow = pure.get(q)
    if not grow:
        continue
    if b == "pairwise":
        p_ok = str(grow.get("winner", "")).upper() == str(gt[q].get("winner", "")).upper()
    else:
        p_ok = bool(grow.get("answer")) == bool(gt[q].get("answer"))
    c2[(s, "pure_llm_same_q_" + ("ok" if p_ok else "bad",)[0], o)] += 1
print("\n== fallback 题 × pure-llm 同题表现 × v11 fallback outcome ==")
tot = defaultdict(lambda: [0, 0])
for (s, pk, o), n in c2.items():
    key = (s, pk)
    tot[key][0] += n
    if o == "correct":
        tot[key][1] += n
for k, (n, c) in sorted(tot.items()):
    print(f"{k[0]:<18} {k[1]:<20} fallback_n={n:>2} 其中v11对={c}")
