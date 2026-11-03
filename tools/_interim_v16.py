# -*- coding: utf-8 -*-
"""v16 中期读数：按融合路径分解 auto/heuristic 正确率 + 缺席率（对比 v15 两大死穴）。"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
rows = [json.loads(l) for l in open(r"outputs/traces/tracestore.jsonl", encoding="utf-8") if l.strip()]
v16 = {}
for r in rows:
    if str(r.get("provenance", {}).get("trace_version")) == "16":
        v16[r["qid"]] = r
print("v16 done:", len(v16), "/40")


def ans_of(r, arm):
    return (r.get("systems", {}).get(arm, {}).get("answer", {}) or {})


def cls(r):
    a = ans_of(r, "pipeline_auto")
    rat = a.get("rationale", "") or ""
    dm = a.get("decision_mode", "") or ""
    if "三通道·v15 一致" in rat:
        return "agree直通(零成本)"
    if "表决+交叉批评·v16" in rat:
        return "v16辩论表决(含批评)"
    if "三通道表决·v15" in rat:
        return "表决(法官v15头)"
    if "纯LLM独立票·v15" in rat:
        return "多数退回纯LLM票"
    if "与证据通道一致" in rat:
        return "多数退回证据侧"
    if "无多数退v12规则" in rat:
        return "无多数退prior"
    if "+triad" not in dm:
        return "非三通道(v12直通/缺席)"
    return "其他"


for arm in ("pipeline_auto", "pipeline_heuristic"):
    print(f"--- {arm} ---")
    agg, tc, tt = {}, 0, 0
    for q, r in v16.items():
        s = r.get("systems", {}).get(arm, {})
        if (s.get("answer") or {}).get("error") is not None:
            continue
        k = cls(r) if arm == "pipeline_auto" else "融合层"
        c, t = agg.get(k, (0, 0))
        agg[k] = (c + (s.get("outcome") == "correct"), t + 1)
    for k, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1]):
        tc += c; tt += t
        print(f"  {k:26s} {c:2d}/{t:2d} = {100 * c / t:5.1f}%")
    print(f"  合计 {tc}/{tt} = {100 * tc / tt:.1f}%" if tt else "  (无)")

miss = [q for q, r in v16.items() if cls(r) == "非三通道(v12直通/缺席)"]
print("缺席/非三通道 qids:", len(miss), miss[:10])

# B 集分布（b_qids 子集）
bq = set(json.load(open(r"outputs/traces/dataset_b_qids.json", encoding="utf-8"))["b_qids"])
b = [q for q in v16 if q in bq]
a_ = [q for q in v16 if q not in bq]


def arm_split(qs, arm):
    agg = {}
    for q in qs:
        s = v16[q].get("systems", {}).get(arm, {})
        if (s.get("answer") or {}).get("error") is not None:
            continue
        k = cls(v16[q]) if arm == "pipeline_auto" else "融合层"
        c, t = agg.get(k, (0, 0))
        agg[k] = (c + (s.get("outcome") == "correct"), t + 1)
    return agg


for label, qs in (("A 集", a_), ("B 集(防记忆·判决题)", b)):
    print(f"=== {label} 已跑 {len(qs)} 题 · auto 分路径 ===")
    tc = tt = 0
    for k, (c, t) in sorted(arm_split(qs, "pipeline_auto").items(), key=lambda x: -x[1][1]):
        tc += c; tt += t
        print(f"  {k:26s} {c:2d}/{t:2d} = {100 * c / t:5.1f}%")
    if tt:
        print(f"  {label} auto 小计 {tc}/{tt} = {100 * tc / tt:.1f}%")
    hb = v16[qs[0]] if qs else None
    bc2 = sum(1 for q in qs if v16[q]["systems"]["pipeline_heuristic"].get("outcome") == "correct")
    if qs:
        print(f"  {label} heuristic {bc2}/{len(qs)} = {100 * bc2 / len(qs):.1f}%")
