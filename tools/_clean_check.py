# -*- coding: utf-8 -*-
"""临时：v17 端点故障剔除后的 LLM-可用配对快照（只读，跑完可删）。"""
import json, collections, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rows = [json.loads(l) for l in open("outputs/traces/tracestore.jsonl", encoding="utf-8") if l.strip()]

def arm(v):
    d = {}
    for r in rows:
        if str((r.get("provenance") or {}).get("trace_version")) == v:
            d[r["qid"]] = r
    return d

a17, a16 = arm("17"), arm("16")
ans = lambda r: (r.get("systems", {}).get("pipeline_auto") or {}).get("answer") or {}
out = lambda r: ((r.get("systems", {}).get("pipeline_auto") or {}).get("outcome") or "abstain")

flag16 = sum(1 for r in a16.values() if ans(r).get("used_llm") is False)
text16 = sum(1 for r in a16.values() if "LLM 不可用" in str(ans(r).get("rationale")))
ab16 = sum(1 for r in a16.values() if out(r) == "abstain")
print(f"[标定] v16 100 行: used_llm=False {flag16} | rationale含'LLM不可用' {text16} | abstain {ab16}")
print(f"[标定] v17 现 {len(a17)} 行: used_llm=False {sum(1 for r in a17.values() if ans(r).get('used_llm') is False)}")

# 污染 = 端点失败的直接指纹：abstain 或 rationale 含 LLM不可用
bad = lambda r: out(r) == "abstain" or "LLM 不可用" in str(ans(r).get("rationale"))
clean = [q for q, r in a17.items() if q in a16 and not bad(r)]
print(f"\n严格 LLM-可用配对行数: {len(clean)}")
c = collections.Counter()
for q in clean:
    c[(out(a16[q]), out(a17[q]))] += 1
for k, n in sorted(c.items(), key=lambda x: -x[1]):
    print(f"  v16 {k[0]:8s} -> v17 {k[1]:8s}: {n}")
if clean:
    n = len(clean)
    a = 100 * sum(1 for q in clean if out(a16[q]) == "correct") / n
    b = 100 * sum(1 for q in clean if out(a17[q]) == "correct") / n
    print(f"同 {n} 题配对: v16={a:.1f}%  v17={b:.1f}%")
    pw = [q for q in clean if q.startswith("pairwise")]
    if pw:
        a2 = 100 * sum(1 for q in pw if out(a16[q]) == "correct") / len(pw)
        b2 = 100 * sum(1 for q in pw if out(a17[q]) == "correct") / len(pw)
        print(f"pairwise n={len(pw)}: v16={a2:.1f}%  v17={b2:.1f}%")
        w2c = [q for q in pw if out(a16[q]) == "wrong" and out(a17[q]) == "correct"]
        c2w = [q for q in pw if out(a16[q]) == "correct" and out(a17[q]) == "wrong"]
        print(f"pairwise 翻转: w2c {len(w2c)} vs c2w {len(c2w)}")
