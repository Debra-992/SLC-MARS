# -*- coding: utf-8 -*-
"""扩测规划探针：题池大小 / v12 缓存覆盖 / 各基线覆盖 / 不同 n 的缓存命中率。"""
import io
import json
import random
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from knowledge_agents import answer_pipeline as ap

qs = [q for q in ap.load_jsonl(ap.QUESTIONS) if q.get("task_bucket") in ("pairwise", "membership")]
print("pool(pairwise+membership):", len(qs))
gt = ap._gt_index()
withgt = [q for q in qs if q["qid"] in gt]
print("  其中带GT:", len(withgt))

rows = [json.loads(l) for l in open(r"outputs/traces/tracestore.jsonl", encoding="utf-8") if l.strip()]
v12 = {r["qid"] for r in rows if str((r.get("provenance") or {}).get("trace_version")) == "12"}
print("v12 rows qids:", len(v12))

for name in ("mdagents", "autogen", "pure_llm", "lit_rag"):
    cov = {json.loads(l).get("qid") for l in open(rf"outputs/traces/answers_{name}.jsonl", encoding="utf-8") if l.strip()}
    print(f"{name}: {len(cov)} qids, 与v12交叠 {len(cov & v12)}")

for n in (60, 80, 100, 120):
    samp = [r["qid"] for r in random.Random(5).sample(withgt, min(n, len(withgt)))]
    md = {json.loads(l).get("qid") for l in open(r"outputs/traces/answers_mdagents.jsonl", encoding="utf-8") if l.strip()}
    print(f"seed5 n={n}: 命中v12缓存 {len([q for q in samp if q in v12])}/{n}, mdagents已答 {len([q for q in samp if q in md])}/{n}")
