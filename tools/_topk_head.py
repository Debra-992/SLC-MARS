# -*- coding: utf-8 -*-
"""v20 topK 排序头（ranking head）——独立 harness，绝不影响 v20 二分类判定/总表/消融。

做法（"把 v20 的证据内核真正用到 topk"，而非此前只喂 LLM 的 PPR 提示重排）：
  1. 候选池 pool = PPR top-15 ∪ GT（与既有 topk_eval_pack 完全同池同题，可直接并表）；
  2. 经真实 MCP 后端 kg_cypher 逐题取 pool 基因的 gene_disease_paths（与 project_knowledge 同式）；
  3. 每个基因的 path 行【按癌种优先截断到 CAP=16 行】——这是 pairwise logistic 训练时所见的
     特征量级上限（project_knowledge 的 LIMIT 16），不做截断会 out-of-distribution 使 z 分数爆表、
     排序退化成图谱度（train/serve skew）；
  4. 用已序列化的 pairwise 图特征 logistic（与 decision_action._score_pairwise_graph 同一系数、
     同一 mu/sd/数学）逐基因打 P(必需/winner) 分 → 按分降序 → 取前 5 为预测排序；
  5. 指标 NDCG@5 / Kendall τ / PR@5 / hit@5 / MRR 全部复用 _topk_eval 的实现，口径与其他臂一致。

两臂：
  v20graph        —— 纯图 logistic 排序（新模块）
  v20graph_ppr    —— 图 logistic 分 + degree-corrected PPR 邻近度 融合排序（消融：加计算邻近通道有没有用）

用法：
  . tools\\slc_mars_env.ps1
  .venv\\Scripts\\python.exe tools\\_topk_head.py            # 跑全部 11 题
  SLC_TOPK_PATHCAP 环境变量可调截断上限（默认 16，= 训练特征量级上界）
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))

from knowledge_agents.graph_features import PAIR_COLS, gene_graph_feat, feat_vec
from knowledge_agents.tools.slc_backend import get_default_backend
# 复用既有 topk 指标与图谱基线构造，保证与 pipeline / pure_llm 两臂逐字同口径
from _topk_eval import (ndcg_at_k, kendall_tau, precision_at_k, hit_at_k, mrr,
                       ppr_ranking, PACK_PATH)

PACK_PATH = Path(os.environ.get("SLC_TOPK_PACK", str(ROOT / "outputs" / "traces" / "topk_eval_pack.json")))
CAP = int(os.environ.get("SLC_TOPK_PATHCAP", "16"))
MODEL_PATH = ROOT / "outputs" / "traces" / "pairwise_graph_model.json"
TRACES = ROOT / "outputs" / "traces"


def load_model() -> dict:
    m = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    assert m.get("task") == "pairwise_graph_logistic" and m.get("coef"), "pairwise 工件异常"
    return m


def prob_of(feat: dict, m: dict) -> float:
    """与 decision_action._score_pairwise_graph 完全一致的推理：z=(x-mu)/sd → coef·z+b → sigmoid。"""
    cols = m["features"]
    mu, sd, coef = m["mu"], m["sd"], m["coef"]
    b = float(m.get("intercept", 0.0))
    x = feat_vec(feat, cols)
    lg = b + sum(coef[i] * ((x[i] - mu[i]) / sd[i]) for i in range(len(cols)))
    return 1.0 / (1.0 + math.exp(-lg))


def fetch_path_rows(be, pool: list[str], cancer: str) -> list[dict]:
    """经 sanctioned 只读 kg_cypher 取 pool 基因的 gene_disease_paths（与 project_knowledge 同式）。
    癌种匹配行优先返回（ORDER BY 主键=cancer 命中），供逐基因截断时优先保留 matched 行。"""
    glist = ",".join("'" + g.replace("'", "") + "'" for g in pool)
    safe_cancer = cancer.replace("'", "")
    cy = (
        "MATCH (g:SLCGene)-[:IS_SOURCE]->(e:RelaEvent)<-[:IS_TARGET]-(d:Disease) "
        f"WHERE g.name IN [{glist}] "
        "OPTIONAL MATCH (p:Pathway)-[:IS_MEDIATOR]->(e) "
        "RETURN g.name AS gene, p.name AS pathway, e.rel_AB AS rel_AB, e.rel_BC AS rel_BC, "
        "CASE WHEN e.rel_AB = e.rel_BC THEN 'PROMOTION' ELSE 'SUPPRESSION' END AS gene_disease_effect, "
        "d.name AS disease, d.diseaseSort AS disease_sort "
        f"ORDER BY CASE WHEN toLower(d.name) CONTAINS toLower('{safe_cancer}') THEN 0 ELSE 1 END, d.name, g.name"
    )
    res = be.call("kg_cypher", {"query": cy, "limit": 1000})
    if res.get("status") != "ok":
        return []
    return res.get("rows") or []


def cap_per_gene(rows: list[dict], cap: int = CAP) -> list[dict]:
    """每基因至多保留 cap 条 path 行（rows 已癌种优先排序 → 保留的是 matched 优先的 top-cap）。
    复刻 pairwise 训练里 project_knowledge LIMIT 16 的特征量级，杜绝 topk 池 out-of-distribution。"""
    kept: list[dict] = []
    seen: dict[str, int] = {}
    for r in rows:
        g = str(r.get("gene", "")).upper()
        if seen.get(g, 0) >= cap:
            continue
        seen[g] = seen.get(g, 0) + 1
        kept.append(r)
    return kept


def metrics_of(pred: list[str], gt: list[str]) -> dict:
    return dict(ndcg5=ndcg_at_k(pred, gt, k=5), kendall_tau=kendall_tau(pred, gt),
                precision5=precision_at_k(pred, gt, k=5), hit5=hit_at_k(pred, gt, k=5),
                mrr=mrr(pred, gt))


def fused_ranked(pool: list[str], probs: dict[str, float], ppr_map: dict[str, float]) -> list[str]:
    """图 logistic 排序 与 PPR 邻近度排序 的 Reciprocal Rank Fusion（RRF, k=60）。
    用秩而非原始分：两通道量纲天差地别（logistic 概率 vs PPR 分值），原始分相加会被
    大尺度通道淹没，那测不出"加邻近通道有没有用"，只测出"谁数值大"。RRF 尺度无关。"""
    graph_rank = sorted(pool, key=lambda g: (-probs.get(g, 0.0), g))
    gr = {g: i for i, g in enumerate(graph_rank)}
    # 无 PPR 分的基因排到图谱提名段之后、按名稳定
    ppr_rank = sorted(pool, key=lambda g: (-ppr_map.get(g, 0.0), g))
    pr = {g: i for i, g in enumerate(ppr_rank)}
    k = 60
    def rrf(g: str) -> float:
        return 1.0 / (k + gr[g]) + 1.0 / (k + pr[g])
    return sorted(pool, key=lambda g: (-rrf(g), g))


def main():
    pack_data = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    packs = pack_data["packs"]
    m = load_model()
    print(f"topk ranking head | CAP/gene={CAP} | model C={m.get('C')} "
          f"| features={len(m['features'])} | 题数={len(packs)}")
    print("=" * 92)

    be = get_default_backend()
    if type(be).__name__ == "UnavailableBackend":
        raise SystemExit("MCP 后端不可用（先 . tools\\slc_mars_env.ps1 且确保 Neo4j 可达）")

    rows_graph, rows_fuse = [], []
    for i, pack in enumerate(packs):
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        cancer = (pack.get("constraints") or {}).get("cancer", "")

        # 图谱基线 ppr 块（与 pipeline 臂同式，供并表基线列一致）
        ppr_pred = ppr_ranking(pack)
        ppr_block = None if ppr_pred is None else metrics_of(ppr_pred, gt)
        ppr_named = len({g.upper() for g, _s in (pack.get("ppr_candidates") or [])
                         if g.upper() in set(pool)})

        raw = fetch_path_rows(be, pool, cancer)
        capped = cap_per_gene(raw, CAP)
        if not capped:
            err = "no_gene_disease_paths(不可测)"
            rows_graph.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                                   pred=None, ppr=ppr_block, ndcg5=0.0, kendall_tau=0.0,
                                   precision5=0.0, hit5=0.0, mrr=0.0, error=err,
                                   tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0}))
            rows_fuse.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                                  pred=None, ppr=ppr_block, ndcg5=0.0, kendall_tau=0.0,
                                  precision5=0.0, hit5=0.0, mrr=0.0, error=err,
                                  tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0}))
            print(f"[{i+1:2d}/11] {qid[-24:]:<24} ERR {err}")
            continue

        feats = {g: gene_graph_feat(g, capped, cancer) for g in pool}
        probs = {g: prob_of(feats[g], m) for g in pool}
        ranked = sorted(pool, key=lambda g: (-probs[g], g))
        pred = ranked[:5]
        gm = metrics_of(pred, gt)

        ppr_map = {g.upper(): (s or 0.0) for g, s in (pack.get("ppr_candidates") or [])}
        fused = fused_ranked(pool, probs, ppr_map)
        fm = metrics_of(fused[:5], gt)

        base = dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                    pred=pred, ppr=ppr_block, seeds=[], n_seed_ok=1, per_seed=[],
                    rationale="", tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0},
                    error="")
        rows_graph.append({**base, **gm})
        rows_fuse.append({**base, **fm, "pred": fused[:5]})

        print(f"[{i+1:2d}/11] {qid[-24:]:<24} pool={len(pool):>2d} named={ppr_named:>2d} "
              f"raw={len(raw):>3d}->capped={len(capped):>3d} | "
              f"graph NDCG={gm['ndcg5']:.3f} hit@5={gm['hit5']:.0f} | "
              f"fuse NDCG={fm['ndcg5']:.3f} hit@5={fm['hit5']:.0f}")

    # 写两臂（topk_compare 会自动发现）
    for name, data in (("v20graph", rows_graph), ("v20graph_ppr", rows_fuse)):
        ok = [r for r in data if not r["error"]]
        json.dump(dict(system=name, hint="none", seeds=[], thinking="OFF", dry_run=False,
                       n=len(data), n_ok=len(ok), n_err=len(data) - len(ok),
                       cap=CAP, results=data),
                  open(TRACES / f"topk_eval_{name}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"\n写入 topk_eval_{name}.json  (ok={len(ok)}/{len(data)})")


if __name__ == "__main__":
    main()
