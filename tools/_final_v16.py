# -*- coding: utf-8 -*-
"""v16 终局分析：v16 与 v12/v15 同题配对，标题/机制双口径 × A(全seed40)/B(19防记忆)/机制(批评真跑) 三集，
两臂(auto/heuristic) + 分路径 + v12->v16 同题翻转。"""
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
    v = str((r.get("provenance") or {}).get("trace_version"))
    if v in ("12", "16"):
        latest[(v, r["qid"])] = r  # 后写覆盖

TAGS = (("三通道·v15 一致", "agree直通"), ("表决+交叉批评·v16", "v16辩论表决(含批评)"),
        ("三通道表决·v15", "v15表决(无批评)"), ("纯LLM独立票·v15", "多数票L1/L2胜"),
        ("与证据通道一致", "多数票E胜"), ("无多数退v12规则", "无多数退prior"),
        ("维持证据通道原判", "全弃权维持E"))


def cls(a):
    rt = a.get("rationale") or ""
    for t, lab in TAGS:
        if t in rt:
            return lab
    return "非三通道(缺席/直通v12)"


def rows_of(v, keep):
    return [r for (ver, q), r in latest.items() if ver == v and (keep is None or q in keep)]


def stats(rr, arm, mech_only=False):
    c = n = 0
    for r in rr:
        s = r["systems"][arm]
        if (s.get("answer") or {}).get("error") is not None:
            continue
        if mech_only and cls(s["answer"]) not in ("v16辩论表决(含批评)",):
            continue
        n += 1
        c += (s.get("outcome") == "correct")
    return c, n, (round(100 * c / n, 1) if n else 0)


ALL = {q for (v, q) in latest if v == "16"}
Bset = ALL & bq
crit = {r["qid"] for r in rows_of("16", ALL) if cls(r["systems"]["pipeline_auto"]["answer"]) == "v16辩论表决(含批评)"}
print(f"v16: 全{len(ALL)}  B{len(Bset)}  机制(批评真跑){len(crit)}")
r16, r16b, r16c = rows_of("16", ALL), rows_of("16", Bset), rows_of("16", crit)
r12, r12b, r12c = rows_of("12", ALL), rows_of("12", Bset), rows_of("12", crit)

for arm in ("pipeline_auto", "pipeline_heuristic"):
    print(f"\n=== {arm} ===")
    print(f"  全集 v12={stats(r12, arm)}  v16={stats(r16, arm)}")
    print(f"  B集  v12={stats(r12b, arm)}  v16={stats(r16b, arm)}")
    print(f"  机制(批评题) v12(同题)={stats(r12c, arm)}  v16={stats(r16c, arm)}")

print("\n=== v16 分路径 (全 / B) auto ===")
for tag, keep in (("全", r16), ("B", r16b)):
    by = collections.defaultdict(lambda: [0, 0])
    for r in keep:
        s = r["systems"]["pipeline_auto"]
        by[cls(s["answer"])][1] += 1
        by[cls(s["answer"])][0] += (s.get("outcome") == "correct")
    print(f"  {tag}:")
    for k, v in sorted(by.items(), key=lambda x: -x[1][1]):
        print(f"    {k:22s} {v[0]:2d}/{v[1]:2d} = {round(100*v[0]/v[1],1) if v[1] else 0}%")

print("\n=== 同题 auto v12->v16 翻转 ===")
flip = collections.Counter()
d12 = {r["qid"]: r for r in r12}
for r in r16:
    o16 = r["systems"]["pipeline_auto"].get("outcome")
    o12 = d12.get(r["qid"], {}).get("systems", {}).get("pipeline_auto", {}).get("outcome")
    if o12:
        flip[f"{o12}->{o16}"] += 1
print("  ", dict(flip))
w2c = [r["qid"] for r in r16 if d12.get(r["qid"], {}).get("systems", {}).get("pipeline_auto", {}).get("outcome") == "wrong" and r["systems"]["pipeline_auto"].get("outcome") == "correct"]
c2w = [r["qid"] for r in r16 if d12.get(r["qid"], {}).get("systems", {}).get("pipeline_auto", {}).get("outcome") == "correct" and r["systems"]["pipeline_auto"].get("outcome") == "wrong"]
print("  wrong->correct:", w2c)
print("  correct->wrong:", c2w)
miss = [r["qid"] for r in r16 if cls(r["systems"]["pipeline_auto"]["answer"]).startswith("非三通道")]
print("  非三通道(缺席/直通):", len(miss), miss[:8])
