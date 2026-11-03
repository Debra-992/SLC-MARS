# -*- coding: utf-8 -*-
"""Quick RRF fusion + comparison for LLM pool experiment."""
import json, math
from pathlib import Path

TRACES = Path(__file__).resolve().parents[1] / "outputs" / "traces"

pack = json.loads((TRACES / "topk_eval_pack_llmpool.json").read_text(encoding="utf-8"))
pure = json.loads((TRACES / "topk_eval_pure_llm_llmpool.json").read_text(encoding="utf-8"))
vg_ppr = json.loads((TRACES / "topk_eval_v20graph_ppr_llmpool.json").read_text(encoding="utf-8"))

pure_map = {r["qid"]: r for r in pure["results"]}
vg_map = {r["qid"]: r for r in vg_ppr["results"]}


def ndcg_at_k(pred, gt, k=5):
    gt_rank = {g: i for i, g in enumerate(gt)}
    rel = {g: (1.0 / math.log2(gt_rank[g] + 2)) for g in gt}
    dcg = sum(rel.get(g, 0) / math.log2(i + 2) for i, g in enumerate(pred[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(pred, gt, k=5):
    return 1.0 if set(pred[:k]) & set(gt[:k]) else 0.0


K = 60
rows = []
for p in pack["packs"]:
    qid = p["qid"]
    pool = [g.upper() for g in p["eval_pool"]]
    gt = [g.upper() for g in p["gt_ranking"]]
    vg_pred = (vg_map.get(qid) or {}).get("pred") or []
    llm_pred = (pure_map.get(qid) or {}).get("pred") or []

    vg_r, llm_r = {}, {}
    for i, g in enumerate(vg_pred):
        vg_r[g] = i
    for i, g in enumerate(llm_pred):
        llm_r[g] = i
    for g in pool:
        if g not in vg_r:
            vg_r[g] = len(pool)
        if g not in llm_r:
            llm_r[g] = len(pool)

    fused = sorted(pool, key=lambda g: (-(1.0 / (K + vg_r[g]) + 1.0 / (K + llm_r[g])), g))
    pred = fused[:5]
    rows.append(dict(qid=qid, pred=pred, ndcg5=ndcg_at_k(pred, gt),
                     hit5=hit_at_k(pred, gt), error=""))

ok = [r for r in rows if not r["error"]]
out = dict(system="v20rrf_llm_llmpool", n=len(rows), n_ok=len(ok),
           n_err=len(rows) - len(ok), results=rows)
json.dump(out, open(TRACES / "topk_eval_v20rrf_llm_llmpool.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
ndcg = sum(r["ndcg5"] for r in ok) / len(ok)
hit = sum(r["hit5"] for r in ok) / len(ok)
print(f"v20rrf_llm_llmpool: NDCG@5={ndcg:.4f}  hit@5={hit:.3f}")

# Compare
def load_ndcg(name):
    p = TRACES / f"topk_eval_{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    ok_r = [r for r in d["results"] if not r["error"]]
    return sum(r["ndcg5"] for r in ok_r) / len(ok_r), sum(r["hit5"] for r in ok_r) / len(ok_r)

print()
print(f"{'Arm':<30s} {'NDCG@5':>8s} {'hit@5':>8s}  {'Pool':>10s}")
print("-" * 62)
all_rows = []
for name, pool_label in [
    ("pure_llm", "PPR"), ("pipeline", "PPR"), ("v20graph", "PPR"),
    ("v20graph_ppr", "PPR"), ("v20rrf_llm", "PPR"),
    ("pure_llm_llmpool", "LLM"), ("pipeline_llmpool", "LLM"),
    ("v20graph_llmpool", "LLM"), ("v20graph_ppr_llmpool", "LLM"),
    ("v20rrf_llm_llmpool", "LLM"),
]:
    r = load_ndcg(name)
    if r:
        ndcg, hit = r
        all_rows.append((name, ndcg, hit, pool_label))

all_rows.sort(key=lambda x: -x[1])
for name, ndcg, hit, pool in all_rows:
    print(f"{name:<30s} {ndcg:>8.4f} {hit:>8.3f}  {pool:>10s}")