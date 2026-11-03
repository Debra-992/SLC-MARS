# -*- coding: utf-8 -*-
"""v15 终局分析：干净行(ts>=复跑起点) 双口径(标题/机制) × A/B × 两臂，对照 v12。"""
import json
import collections
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = Path(__file__).resolve().parents[1] / "outputs" / "traces"
rows = [json.loads(l) for l in (BASE / "tracestore.jsonl").open(encoding="utf-8") if l.strip()]
bq = set(json.loads((BASE / "dataset_b_qids.json").read_text(encoding="utf-8"))["b_qids"])
latest = {}
for r in rows:
    p = r.get("provenance") or {}
    v = str(p.get("trace_version"))
    if v in ("12", "15"):
        key = (v, r["qid"])
        latest[key] = r  # 同 qid 后写覆盖

TAGS = (("三通道·v15 一致", "agree直通"), ("三通道表决·v15", "智能体辩论表决"),
        ("纯LLM独立票·v15", "多数票L1/L2胜"), ("与证据通道一致", "多数票E胜"),
        ("无多数退v12规则", "无多数退prior"), ("维持证据通道原判", "全弃权维持E"))


def cls(a):
    rt = a.get("rationale") or ""
    for t, lab in TAGS:
        if t in rt:
            return lab
    return "prior缺席直通"


def v12rows(keep):
    return [r for (v, q), r in latest.items() if v == "12" and (keep is None or q in keep)]


def v15rows(keep):
    out = []
    for (v, q), r in latest.items():
        if v == "15" and (r["provenance"].get("ts") or "") >= "2026-10-23T04:50":
            if keep is None or q in keep:
                out.append(r)
    return out


def armstats(rr, arm, mech_only):
    c = n = 0
    for r in rr:
        s = r["systems"][arm]
        if mech_only and cls(s["answer"]) == "prior缺席直通":
            continue
        n += 1
        c += (s.get("outcome") == "correct")
    return c, n, (round(100 * c / n, 1) if n else 0)


A = {q for (v, q) in latest if v == "15" and (latest[(v, q)]["provenance"].get("ts") or "") >= "2026-10-23T04:50"}
B = A & bq
print(f"干净v15题数: A={len(A)}  B={len(B)}")
r15 = v15rows(A); r15b = v15rows(B); r12 = v12rows(A); r12b = v12rows(B)
for arm in ("pipeline_auto", "pipeline_heuristic"):
    m12 = armstats(r12, arm, False)
    h15 = armstats(r15, arm, False)
    k15 = armstats(r15, arm, True)
    m12b = armstats(r12b, arm, False)
    h15b = armstats(r15b, arm, False)
    k15b = armstats(r15b, arm, True)
    print(f"\n=== {arm} ===")
    print(f"  A集: v12标题={m12}  v15标题={h15}  v15机制(剔除缺席直通)={k15}")
    print(f"  B集: v12标题={m12b}  v15标题={h15b}  v15机制={k15b}")

print("\n=== v15 分路径 (A / B) ===")
for arm in ("pipeline_auto", "pipeline_heuristic"):
    for tag, keep in (("A", r15), ("B", r15b)):
        by = collections.defaultdict(lambda: [0, 0])
        for r in keep:
            s = r["systems"][arm]
            k = cls(s["answer"])
            by[k][1] += 1
            by[k][0] += (s.get("outcome") == "correct")
        print(f"  {arm} {tag}:")
        for k, v in sorted(by.items(), key=lambda x: -x[1][1]):
            print(f"    {k:12s} {v[0]:2d}/{v[1]:2d} acc={round(100*v[0]/v[1],1) if v[1] else 0}%")

# 同题翻转对照 v12->v15(标题口径)
flip = collections.Counter()
d12 = {r["qid"]: r for r in r12}
for r in r15:
    o15 = r["systems"]["pipeline_auto"].get("outcome")
    o12 = d12.get(r["qid"], {}).get("systems", {}).get("pipeline_auto", {}).get("outcome")
    if o12:
        flip[f"{o12}->{o15}"] += 1
print("\n=== 同题 auto 臂 v12->v15 翻转 ===", dict(flip))
miss = [r["qid"] for r in r15 if cls(r["systems"]["pipeline_auto"]["answer"]) == "prior缺席直通"]
print("缺席直通题:", miss)
