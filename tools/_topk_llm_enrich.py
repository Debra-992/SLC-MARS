# -*- coding: utf-8 -*-
"""方向 2：LLM 作为特征提取器——逐基因癌种相关性打分 + v20graph_ppr RRF 融合。

核心假设：per-gene scoring（逐基因独立评分）比 holistic ranking（全池一次排序）更精准，
因为 LLM 不用在 context window 中同时比较 18 个基因，每个基因获得专注评估。

流程：
  1. 对每题，调 LLM 给 pool 中每个基因打 1-5 分（批量 prompt，单题 1 次调用）
  2. 按分降序 → per_gene_score 排序
  3. RRF(k=60) 融合 per_gene_score 排名 + v20graph_ppr 排名
  4. 对比 pure_llm / v20rrf_llm / v20rrf_pergene

用法:
  . tools\slc_mars_env.ps1
  $env:NO_PROXY='*'; $env:PYTHONIOENCODING='utf-8'
  .venv\Scripts\python.exe tools\_topk_llm_enrich.py
"""

from __future__ import annotations
import asyncio, json, math, os, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))
sys.path.insert(0, str(ROOT / "tools"))

from _topk_eval import llm_settings, usage_triplet, ndcg_at_k, hit_at_k, PACK_PATH
from _topk_head import (fetch_path_rows, cap_per_gene, metrics_of, fused_ranked,
                        load_model, prob_of, CAP as ORIG_CAP, TRACES)

TRACES_DIR = TRACES
K_RRF = 60

# ---- Per-gene 评分 prompt ----
SCORE_PROMPT = """你是癌症生物学专家。请对以下每个 SLC 基因在 {cancer} 中的功能相关性打分（1-5 分整数）。

评分标准:
  1 = 无已知关联或无关
  2 = 弱相关（仅泛癌代谢通路中的一般成员）
  3 = 中等相关（有文献报道但非核心角色）
  4 = 较强相关（在该癌种有明确功能研究，涉及关键通路）
  5 = 强相关（该癌种的已知驱动依赖/药物靶点/CRISPR 必需基因）

仅输出 JSON，不要额外解释:
{{
  "scores": {{"GENE1": 4, "GENE2": 3, ...}},
  "rationale_brief": "一句话总结评分逻辑"
}}"""


def build_score_prompt(cancer: str, genes: list[str]) -> str:
    gene_list = "\n".join(f"  - {g}" for g in sorted(genes))
    return SCORE_PROMPT.format(cancer=cancer, genes=gene_list) + f"\n基因列表:\n{gene_list}"


def parse_scores(text: str, valid_genes: set[str]) -> dict[str, int]:
    """从 LLM 输出提取逐基因评分。"""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        scores_raw = obj.get("scores", {})
        if not isinstance(scores_raw, dict):
            return {}
        scores = {}
        for g, s in scores_raw.items():
            g_upper = str(g).upper()
            if g_upper in valid_genes:
                try:
                    v = int(s)
                    scores[g_upper] = max(1, min(5, v))
                except (ValueError, TypeError):
                    pass
        return scores
    except json.JSONDecodeError:
        return {}


async def run():
    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        raise SystemExit("config2.yaml 未找到 llm 配置")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=300, max_retries=0)
    model = cfg["model"]
    print(f"LLM: {cfg['base_url'][:40]}...  model={model}")

    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    m_graph = load_model()

    # ---- Phase 1: Per-gene LLM 评分 ----
    print("\n=== Phase 1: LLM per-gene scoring ===")
    from knowledge_agents.tools.slc_backend import get_default_backend
    be = get_default_backend()
    if type(be).__name__ == "UnavailableBackend":
        raise SystemExit("MCP 后端不可用")

    per_gene_scores: dict[str, dict[str, int]] = {}  # qid -> {gene: score}
    t0 = time.time()
    total_tok = 0

    for i, pack in enumerate(packs):
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        cancer = (pack.get("constraints") or {}).get("cancer", "")
        valid = set(pool)

        prompt = build_score_prompt(cancer, pool)
        try:
            extra = {"enable_thinking": False}
            rsp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    extra_body=extra,
                ), timeout=120)
            text = rsp.choices[0].message.content or ""
            p, c, t = usage_triplet(rsp)
            total_tok += t
            scores = parse_scores(text, valid)
            per_gene_scores[qid] = scores
            missing = valid - set(scores.keys())
            status = f"ok={len(scores)}/{len(valid)}" + (f" miss={len(missing)}" if missing else "")
            print(f"  [{i+1:2d}] {qid[-24:]:<24s} {status:<20s} tok={t:>5d}")
            if missing:
                print(f"        missing: {sorted(missing)[:5]}...")
        except Exception as e:
            print(f"  [{i+1:2d}] {qid[-24:]:<24s} ERR {type(e).__name__}: {str(e)[:60]}")
            per_gene_scores[qid] = {}

    print(f"  total_tok={total_tok}  elapsed={time.time()-t0:.0f}s")

    # ---- Phase 2: Per-gene 排序臂 ----
    print("\n=== Phase 2: Ranking arms ===")
    results_pergene, results_rrf = [], []
    for i, pack in enumerate(packs):
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        cancer = (pack.get("constraints") or {}).get("cancer", "")
        ppr_map = {g.upper(): (s or 0.0) for g, s in (pack.get("ppr_candidates") or [])}

        # v20graph_ppr ranking (recompute via async MCP)
        safe_cancer = cancer.replace("'", "")
        glist = ",".join("'" + g.replace("'", "") + "'" for g in pool)
        res = await be.acall("kg_cypher", {
            "query": (
                "MATCH (g:SLCGene)-[:IS_SOURCE]->(e:RelaEvent)<-[:IS_TARGET]-(d:Disease) "
                f"WHERE g.name IN [{glist}] "
                "OPTIONAL MATCH (p:Pathway)-[:IS_MEDIATOR]->(e) "
                "RETURN g.name AS gene, p.name AS pathway, e.rel_AB AS rel_AB, e.rel_BC AS rel_BC, "
                "CASE WHEN e.rel_AB = e.rel_BC THEN 'PROMOTION' ELSE 'SUPPRESSION' END AS gene_disease_effect, "
                "d.name AS disease, d.diseaseSort AS disease_sort "
                f"ORDER BY CASE WHEN toLower(d.name) CONTAINS toLower('{safe_cancer}') THEN 0 ELSE 1 END, d.name, g.name"
            ), "limit": 1000
        })
        rows_raw = res.get("rows") or [] if res.get("status") == "ok" else []
        capped = cap_per_gene(rows_raw, ORIG_CAP)
        if not capped:
            results_pergene.append(dict(qid=qid, pred=None, ndcg5=0, hit5=0, error="no_paths"))
            results_rrf.append(dict(qid=qid, pred=None, ndcg5=0, hit5=0, error="no_paths"))
            continue

        from knowledge_agents.graph_features import gene_graph_feat
        feats = {g: gene_graph_feat(g, capped, cancer) for g in pool}
        probs = {g: prob_of(feats[g], m_graph) for g in pool}
        vg_ranked = fused_ranked(pool, probs, ppr_map)
        vg_r = {g: idx for idx, g in enumerate(vg_ranked)}

        # Per-gene score ranking
        scores = per_gene_scores.get(qid, {})
        pergene_ranked = sorted(pool, key=lambda g: (-scores.get(g, 0), g))
        pg_r = {g: idx for idx, g in enumerate(pergene_ranked)}

        # Per-gene only (baseline)
        pg_pred = pergene_ranked[:5]
        gm_pg = dict(ndcg5=ndcg_at_k(pg_pred, gt), hit5=hit_at_k(pg_pred, gt))

        # RRF: v20graph_ppr + per_gene_score
        def rrf(g):
            return 1.0 / (K_RRF + vg_r.get(g, len(pool))) + 1.0 / (K_RRF + pg_r.get(g, len(pool)))

        fused = sorted(pool, key=lambda g: (-rrf(g), g))
        gm_fuse = dict(ndcg5=ndcg_at_k(fused[:5], gt), hit5=hit_at_k(fused[:5], gt))

        results_pergene.append(dict(qid=qid, pred=pg_pred, error="", **gm_pg))
        results_rrf.append(dict(qid=qid, pred=fused[:5], error="", **gm_fuse))

        print(f"  [{i+1:2d}] {qid[-24:]:<24s} "
              f"pergene_NDCG={gm_pg['ndcg5']:.3f} hit={gm_pg['hit5']:.0f} | "
              f"rrf_NDCG={gm_fuse['ndcg5']:.3f} hit={gm_fuse['hit5']:.0f}")

    # ---- Phase 3: Write & Compare ----
    out_pg = TRACES_DIR / "topk_eval_v20pergene.json"
    ok_pg = [r for r in results_pergene if not r["error"]]
    json.dump(dict(system="v20pergene", n=len(results_pergene), n_ok=len(ok_pg),
                   n_err=len(results_pergene) - len(ok_pg), results=results_pergene),
              open(out_pg, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    out_rrf = TRACES_DIR / "topk_eval_v20rrf_pergene.json"
    ok_rrf = [r for r in results_rrf if not r["error"]]
    json.dump(dict(system="v20rrf_pergene", n=len(results_rrf), n_ok=len(ok_rrf),
                   n_err=len(results_rrf) - len(ok_rrf), results=results_rrf),
              open(out_rrf, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    ndcg_pg = sum(r["ndcg5"] for r in ok_pg) / len(ok_pg)
    hit_pg = sum(r["hit5"] for r in ok_pg) / len(ok_pg)
    ndcg_rrf = sum(r["ndcg5"] for r in ok_rrf) / len(ok_rrf)
    hit_rrf = sum(r["hit5"] for r in ok_rrf) / len(ok_rrf)

    # ---- Comparison ----
    def load_ndcg(name):
        p = TRACES_DIR / f"topk_eval_{name}.json"
        if not p.exists(): return None
        d = json.loads(p.read_text(encoding="utf-8"))
        ok_r = [r for r in d["results"] if not r["error"]]
        return sum(r["ndcg5"] for r in ok_r) / len(ok_r), sum(r["hit5"] for r in ok_r) / len(ok_r)

    print(f"\n{'='*70}")
    print(f"{'Arm':<30s} {'NDCG@5':>8s} {'hit@5':>8s}")
    print(f"{'-'*48}")
    arms = [
        ("v20pergene", ndcg_pg, hit_pg),
        ("v20rrf_pergene", ndcg_rrf, hit_rrf),
    ]
    for name in ("pure_llm", "v20graph", "v20graph_ppr", "v20rrf_llm", "v20rrf_llm_full"):
        r = load_ndcg(name)
        if r:
            arms.append((name, r[0], r[1]))
    arms.sort(key=lambda x: -x[1])
    for name, ndcg, hit in arms:
        new_flag = " ← NEW" if name in ("v20pergene", "v20rrf_pergene") else ""
        print(f"{name:<30s} {ndcg:>8.4f} {hit:>8.3f}{new_flag}")

    # Write per-gene scores for reference
    json.dump(per_gene_scores, open(TRACES_DIR / "pergene_llm_scores.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(run())