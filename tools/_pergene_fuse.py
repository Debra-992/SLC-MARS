# -*- coding: utf-8 -*-
"""pergene + pure_llm RRF 融合（互补最强两臂）。"""
import json, math
from pathlib import Path

TRACES = Path(__file__).resolve().parents[1] / "outputs" / "traces"
K = 60


def ndcg_at_k(pred, gt, k=5):
    gt_rank = {g: i for i, g in enumerate(gt)}
    rel = {g: (1.0 / math.log2(gt_rank[g] + 2)) for g in gt}
    dcg = sum(rel.get(g, 0) / math.log2(i + 2) for i, g in enumerate(pred[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(pred, gt, k=5):
    return 1.0 if set(pred[:k]) & set(gt[:k]) else 0.0


# Load data
pack = json.loads((TRACES / "topk_eval_pack.json").read_text(encoding="utf-8"))
pure = json.loads((TRACES / "topk_eval_pure_llm.json").read_text(encoding="utf-8"))
pg_scores = json.loads((TRACES / "pergene_llm_scores.json").read_text(encoding="utf-8"))
pure_map = {r["qid"]: r for r in pure["results"]}
pack_map = {p["qid"]: p for p in pack["packs"]}

rows = []
for p in pack["packs"]:
    qid = p["qid"]
    pool = [g.upper() for g in p["eval_pool"]]
    gt = [g.upper() for g in p["gt_ranking"]]
    scores = pg_scores.get(qid, {})
    llm_pred = (pure_map.get(qid) or {}).get("pred") or []

    # v20pergene: full ranking from scores
    pg_ranked = sorted(pool, key=lambda g: (-scores.get(g, 0), g))
    pg_r = {g: i for i, g in enumerate(pg_ranked)}

    # pure_llm: top-5 known, rest = pool_size
    llm_r = {}
    for i, g in enumerate(llm_pred):
        llm_r[g] = i
    for g in pool:
        if g not in llm_r:
            llm_r[g] = len(pool)

    # RRF
    fused = sorted(pool, key=lambda g: (
        -(1.0 / (K + pg_r[g]) + 1.0 / (K + llm_r[g])), g))
    pred = fused[:5]
    ndcg = ndcg_at_k(pred, gt)
    hit = hit_at_k(pred, gt)
    rows.append(dict(qid=qid, pred=pred, ndcg5=ndcg, hit5=hit, error=""))

ok = [r for r in rows if not r["error"]]
out = dict(system="v20rrf_pergene_purellm", n=len(rows), n_ok=len(ok),
           n_err=len(rows) - len(ok), results=rows)
json.dump(out, open(TRACES / "topk_eval_v20rrf_pergene_purellm.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
ndcg = sum(r["ndcg5"] for r in ok) / len(ok)
hit = sum(r["hit5"] for r in ok) / len(ok)
print(f"v20rrf_pergene_purellm: NDCG@5={ndcg:.4f}  hit@5={hit:.3f}")

# Compare all arms
def load_ndcg(name):
    p = TRACES / f"topk_eval_{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    ok_r = [r for r in d["results"] if not r["error"]]
    return sum(r["ndcg5"] for r in ok_r) / len(ok_r), sum(r["hit5"] for r in ok_r) / len(ok_r)

print()
print(f"{'Arm':<32s} {'NDCG@5':>8s} {'hit@5':>8s}")
print("-" * 50)
all_rows = [("v20rrf_pergene_purellm", ndcg, hit)]
for name in ("v20pergene", "v20rrf_llm", "v20graph_ppr", "pure_llm", "v20graph", "pipeline"):
    r = load_ndcg(name)
    if r:
        all_rows.append((name, r[0], r[1]))
all_rows.sort(key=lambda x: -x[1])
for name, ndcg_val, hit_val in all_rows:
    new_flag = " <- NEW" if name == "v20rrf_pergene_purellm" else ""
    print(f"{name:<32s} {ndcg_val:>8.4f} {hit_val:>8.3f}{new_flag}")

# Per-question breakdown
print(f"\n{'qid':<26s} {'pergene':>8s} {'pure_llm':>8s} {'fused':>8s}  {'delta':>7s}")
pg_data = load_ndcg("v20pergene")
for r in rows:
    qid_short = r["qid"][-24:]
    pg_ndcg = r["ndcg5"]  # fused value computed above... wait no
    # Need per-question pg and pure NDCG
    pass

# More accurate per-question
print(f"\n{'qid':<26s} {'pergene':>8s} {'pure_llm':>8s} {'fused':>8s}  win")
for p in pack["packs"]:
    qid = p["qid"]
    short = qid[-24:]
    pg_r = [r2 for r2 in rows if r2["qid"] == qid][0]
    pure_r = pure_map.get(qid, {})
    pg_ndcg = [r2 for r2 in json.loads((TRACES / "topk_eval_v20pergene.json").read_text(encoding="utf-8"))["results"] if r2["qid"] == qid][0]["ndcg5"]
    pure_ndcg = pure_r.get("ndcg5", 0)
    fused_ndcg = pg_r["ndcg5"]
    best = max(pg_ndcg, pure_ndcg, fused_ndcg)
    winner = "fused" if fused_ndcg == best else ("pergene" if pg_ndcg == best else "pure")
    print(f"{short:<26s} {pg_ndcg:>8.3f} {pure_ndcg:>8.3f} {fused_ndcg:>8.3f}  {winner}")