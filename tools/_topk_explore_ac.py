# -*- coding: utf-8 -*-
"""方向 A（CAP 扫描）+ 方向 C（LLM 融合近似）一键出数。

A: SLC_TOPK_PATHCAP = 16 / 32 / 48 各跑一遍 v20graph，写入 topk_eval_v20graph_cap<X>.json。
C: 读纯LLM 臂 top-5 + v20graph_ppr 全排序，近似 RRF 融合，写入 topk_eval_v20rrf_llm.json。
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))
sys.path.insert(0, str(ROOT / "tools"))

from _topk_head import (fetch_path_rows, cap_per_gene, metrics_of, ppr_ranking,
                        fused_ranked, load_model as load_pairwise, prob_of,
                        CAP as ORIG_CAP, PACK_PATH, TRACES)
from knowledge_agents.graph_features import gene_graph_feat

TRACES_DIR = TRACES


def run_cap(cap: int, label: str):
    """单次 CAP 跑 v20graph 臂（纯图 logistic，不融合 PPR）"""
    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    m = load_pairwise()
    rows = []
    for i, pack in enumerate(packs):
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        cancer = (pack.get("constraints") or {}).get("cancer", "")
        ppr_named = len({g.upper() for g, _s in (pack.get("ppr_candidates") or [])
                         if g.upper() in set(pool)})

        raw = fetch_path_rows(_be, pool, cancer)
        capped = cap_per_gene(raw, cap)
        if not capped:
            rows.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                             pred=None, error="no_gene_disease_paths(不可测)",
                             ndcg5=0.0, kendall_tau=0.0, precision5=0.0, hit5=0.0, mrr=0.0,
                             tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0},
                             ppr=None, seeds=[], n_seed_ok=1, per_seed=[], rationale=""))
            continue
        feats = {g: gene_graph_feat(g, capped, cancer) for g in pool}
        probs = {g: prob_of(feats[g], m) for g in pool}
        ranked = sorted(pool, key=lambda g: (-probs[g], g))
        pred = ranked[:5]
        gm = metrics_of(pred, gt)
        rows.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                         pred=pred, ppr=None, seeds=[], n_seed_ok=1, per_seed=[],
                         rationale="", tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0},
                         error="", **gm))
    ok = [r for r in rows if not r["error"]]
    out = TRACES_DIR / f"topk_eval_v20graph_{label}.json"
    json.dump(dict(system=f"v20graph_{label}", n=len(rows), n_ok=len(ok),
                   n_err=len(rows)-len(ok), cap=cap, results=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ndcg = sum(r["ndcg5"] for r in ok) / len(ok)
    hit = sum(r["hit5"] for r in ok) / len(ok)
    print(f"  CAP={cap:>2d}  NDCG@5={ndcg:.4f}  hit@5={hit:.3f}  -> {out.name}")
    return ndcg, hit


def rrf_score(rank_vg: int, rank_llm: int, k: int = 60) -> float:
    return 1.0 / (k + rank_vg) + 1.0 / (k + rank_llm)


def run_rrf_llm():
    """近似 RRF: v20graph_ppr 全排序 + 纯 LLM top-5（未知基因 rank=pool_size）"""
    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    m = load_pairwise()
    llm_data = json.loads((TRACES_DIR / "topk_eval_pure_llm.json").read_text(encoding="utf-8"))
    llm_map = {r["qid"]: r for r in llm_data["results"]}

    rows = []
    for i, pack in enumerate(packs):
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        cancer = (pack.get("constraints") or {}).get("cancer", "")
        ppr_named = len({g.upper() for g, _s in (pack.get("ppr_candidates") or [])
                         if g.upper() in set(pool)})
        ppr_map = {g.upper(): (s or 0.0) for g, s in (pack.get("ppr_candidates") or [])}

        raw = fetch_path_rows(_be, pool, cancer)
        capped = cap_per_gene(raw, ORIG_CAP)
        if not capped:
            rows.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                             pred=None, error="no_gene_disease_paths(不可测)",
                             ndcg5=0.0, kendall_tau=0.0, precision5=0.0, hit5=0.0, mrr=0.0,
                             tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0},
                             ppr=None, seeds=[], n_seed_ok=1, per_seed=[], rationale=""))
            continue
        feats = {g: gene_graph_feat(g, capped, cancer) for g in pool}
        probs = {g: prob_of(feats[g], m) for g in pool}

        # v20graph_ppr 全排序（graph + PPR RRF）
        vg_ranked = fused_ranked(pool, probs, ppr_map)
        vg_r = {g: i for i, g in enumerate(vg_ranked)}

        # LLM 近似排序：top-5 有秩，其余 = pool_size
        llm_pred = (llm_map.get(qid) or {}).get("pred") or []
        llm_r = {}
        for g in pool:
            try:
                idx = llm_pred.index(g)
                llm_r[g] = idx
            except ValueError:
                llm_r[g] = len(pool)  # 未知 = 最差

        # RRF(k=60)
        fused = sorted(pool, key=lambda g: (-rrf_score(vg_r[g], llm_r[g]), g))
        pred = fused[:5]
        gm = metrics_of(pred, gt)
        rows.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                         pred=pred, ppr=None, seeds=[], n_seed_ok=1, per_seed=[],
                         rationale="", tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0},
                         error="", **gm))
        overlap = len(set(llm_pred[:5]) & set(vg_ranked[:5]))
        print(f"  [{i+1:2d}] {qid[-24:]:<24} vg_top={vg_ranked[:3]} llm_top={llm_pred[:3]} "
              f"overlap@5={overlap}  fused_NDCG={gm['ndcg5']:.3f}")

    ok = [r for r in rows if not r["error"]]
    out = TRACES_DIR / "topk_eval_v20rrf_llm.json"
    json.dump(dict(system="v20rrf_llm", n=len(rows), n_ok=len(ok),
                   n_err=len(rows)-len(ok), cap=ORIG_CAP, results=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ndcg = sum(r["ndcg5"] for r in ok) / len(ok)
    hit = sum(r["hit5"] for r in ok) / len(ok)
    tau = sum(r["kendall_tau"] for r in ok) / len(ok)
    print(f"  v20rrf_llm  NDCG@5={ndcg:.4f}  hit@5={hit:.3f}  tau={tau:+.4f} -> {out.name}")
    return ndcg, hit


def main():
    # 读基准
    v20full = json.loads((TRACES_DIR / "topk_eval_v20full.json").read_text(encoding="utf-8"))
    v20graph = json.loads((TRACES_DIR / "topk_eval_v20graph.json").read_text(encoding="utf-8"))
    v20graph_ppr = json.loads((TRACES_DIR / "topk_eval_v20graph_ppr.json").read_text(encoding="utf-8"))
    pure = json.loads((TRACES_DIR / "topk_eval_pure_llm.json").read_text(encoding="utf-8"))
    pipeline = json.loads((TRACES_DIR / "topk_eval_pipeline.json").read_text(encoding="utf-8"))

    def _m(d, key):
        ok = [r for r in d["results"] if not r["error"]]
        return sum(r[key] for r in ok) / len(ok) if ok else 0.0

    print("=" * 70)
    print(f"baseline  v20graph    NDCG={_m(v20graph,'ndcg5'):.4f}  hit={_m(v20graph,'hit5'):.3f}  "
          f"tau={_m(v20graph,'kendall_tau'):+.4f}")
    print(f"baseline  v20graph_ppr NDCG={_m(v20graph_ppr,'ndcg5'):.4f}  hit={_m(v20graph_ppr,'hit5'):.3f}  "
          f"tau={_m(v20graph_ppr,'kendall_tau'):+.4f}")
    print(f"baseline  pure_llm   NDCG={_m(pure,'ndcg5'):.4f}  hit={_m(pure,'hit5'):.3f}  "
          f"tau={_m(pure,'kendall_tau'):+.4f}")
    print(f"baseline  pipeline   NDCG={_m(pipeline,'ndcg5'):.4f}  hit={_m(pipeline,'hit5'):.3f}  "
          f"tau={_m(pipeline,'kendall_tau'):+.4f}")
    print()

    print("=== A: CAP 扫描 ===")
    for cap in (32, 48):
        run_cap(cap, f"cap{cap}")

    print("\n=== C: v20graph_ppr + pureLLM RRF 融合 ===")
    run_rrf_llm()

    print("\n=== 汇总 ===")
    rows = []
    for f in sorted(TRACES_DIR.glob("topk_eval_v20graph_cap*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        ndcg = _m(d, "ndcg5")
        hit = _m(d, "hit5")
        rows.append((f.stem.replace("topk_eval_", ""), ndcg, hit))
    for name in ("v20graph", "v20graph_ppr", "v20full", "v20rrf_llm", "pure_llm", "pipeline"):
        p = TRACES_DIR / f"topk_eval_{name}.json"
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            rows.append((name, _m(d, "ndcg5"), _m(d, "hit5")))
    rows.sort(key=lambda x: -x[1])
    print(f"{'Arm':<24s} {'NDCG@5':>8s} {'hit@5':>8s}")
    print("-" * 42)
    for name, ndcg, hit in rows:
        print(f"{name:<24s} {ndcg:>8.4f} {hit:>8.3f}")


_be = None

if __name__ == "__main__":
    from knowledge_agents.tools.slc_backend import get_default_backend
    _be = get_default_backend()
    if type(_be).__name__ == "UnavailableBackend":
        raise SystemExit("MCP 后端不可用")
    main()