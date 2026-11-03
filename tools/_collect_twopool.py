# -*- coding: utf-8 -*-
"""Collect verified NDCG@5 / hit@5 for the two-pool TopK matrix (scratch).

Metric = mean over non-error rows, matching _topk_baselines.report() and
_v20_big_table.eval_topk(). Prints both the graph pool and the LLM pool for
every system so Table 2 can be rebuilt from real numbers only.
"""
import json
from pathlib import Path

TRACES = Path(__file__).resolve().parents[1] / "outputs" / "traces"

ROWS = [
    ("pure_llm",          "topk_eval_pure_llm.json",       "topk_eval_pure_llm_llmpool.json"),
    ("MetaGPT",           "topk_eval_metagpt.json",        "topk_eval_metagpt_llmpool.json"),
    ("AutoGen",           "topk_eval_autogen.json",        "topk_eval_autogen_llmpool.json"),
    ("MDAgents",          "topk_eval_mdagents.json",       "topk_eval_mdagents_llmpool.json"),
    ("PPR->LLM",          "topk_eval_pipeline.json",       "topk_eval_pipeline_llmpool.json"),
    ("per-gene (Kira)",   "topk_eval_v20pergene.json",     "topk_eval_v20pergene_llmpool.json"),
    ("RRF (Kira+Cael)",   "topk_eval_v20rrf_llm.json",     "topk_eval_v20rrf_llm_llmpool.json"),
    ("graph+PPR (Cael)",  "topk_eval_v20graph_ppr.json",   "topk_eval_v20graph_ppr_llmpool.json"),
    ("graph only (Cael)", "topk_eval_v20graph.json",       "topk_eval_v20graph_llmpool.json"),
]


def metrics(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    res = d.get("results", [])
    ok = [r for r in res if not r.get("error")]
    if not ok:
        return (len(res), 0, 0.0, 0.0)
    ndcg = sum(r.get("ndcg5", 0.0) for r in ok) / len(ok)
    hit = sum(r.get("hit5", 0.0) for r in ok) / len(ok)
    return (len(res), len(ok), ndcg, hit)


def fmt(m):
    if m is None:
        return "  MISSING        "
    n, nok, ndcg, hit = m
    return f"{ndcg:.4f}/{hit:.3f} n={nok}/{n}"


print(f"{'System':<18s} | {'GRAPH pool':<22s} | {'LLM pool':<22s}")
print("-" * 70)
for label, gf, lf in ROWS:
    g = metrics(TRACES / gf)
    l = metrics(TRACES / lf)
    print(f"{label:<18s} | {fmt(g):<22s} | {fmt(l):<22s}")
