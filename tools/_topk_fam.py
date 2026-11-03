# -*- coding: utf-8 -*-
"""杠杆3 落地：v20graph_fam —— 给"图沉默基因"用【同亚家族旁系】的疾病路径证据补分。

诊断实锤：候选池里 n_paths==0 的图沉默基因，29/29 全是 GT 基因（占全部 GT 的 53%）。
它们因缺 gene->pathway->disease 直连证据，pairwise 图 logistic 只能给"全 0 特征 → 基线概率"，
排序退化成按名字典序、几乎必然沉底——这正是 PPR/图头系统性漏掉的那批必需基因。

做法（纯特征输入侧增强，复用【同一个 pairwise 模型】、不重训、不碰冠军二分类路径）：
  1. 非沉默基因：概率与 v20graph 逐字相同（直接用自己的 path 特征过模型）。
  2. 沉默基因 g：取其 SLC 亚家族(fam = SLC<编号>)在【全图谱】里有疾病路径证据的旁系成员，
     用同一模型给每个成员打真实概率（成员特征天然在分布内，不做外推特征算术），
     取这些概率的均值作为 g 的"家族先验" → g 的分 = alpha × 家族先验。
     这样沉默基因获得的是家族级必需性证据，而非臆造的直接证据。
  3. 家族无可用旁系（如 SLCO4A1 无数字亚家族、或全族皆沉默）→ 退回基线（与 v20graph 同）。
  4. 按分降序取前 5，指标复用 _topk_eval，与 v20graph/pipeline/pure_llm 同池同题、口径一致。

alpha 由 SLC_TOPK_FAMALPHA 控制（默认 1.0，即直接采信家族均值；<1 表示对家族外推打折）。

用法：
  . tools\\slc_mars_env.ps1
  .venv\\Scripts\\python.exe tools\\_topk_fam.py
"""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))
sys.path.insert(0, str(ROOT / "tools"))

from knowledge_agents.graph_features import gene_graph_feat
from knowledge_agents.tools.slc_backend import get_default_backend
from _topk_head import (fetch_path_rows, cap_per_gene, metrics_of, ppr_ranking,
                       load_model, prob_of, CAP, PACK_PATH, TRACES)

FAM_RE = re.compile(r"^SLC(\d+)", re.I)
ALPHA = float(os.environ.get("SLC_TOPK_FAMALPHA", "1.0"))


def subfamily(gene: str) -> str | None:
    m = FAM_RE.match((gene or "").strip())
    return ("SLC" + m.group(1)) if m else None


def fetch_family_rows(be, fams: set[str], cancer: str) -> list[dict]:
    """经只读 kg_cypher 取【全图谱】里属于这些亚家族的基因的 disease 路径行（癌种优先截断由调用方负责）。"""
    if not fams:
        return []
    pred = " OR ".join("g.name STARTS WITH 'SLC%s'" % f[3:].replace("'", "") for f in sorted(fams))
    safe_cancer = cancer.replace("'", "")
    cy = (
        "MATCH (g:SLCGene)-[:IS_SOURCE]->(e:RelaEvent)<-[:IS_TARGET]-(d:Disease) "
        f"WHERE {pred} "
        "OPTIONAL MATCH (p:Pathway)-[:IS_MEDIATOR]->(e) "
        "RETURN g.name AS gene, p.name AS pathway, e.rel_AB AS rel_AB, e.rel_BC AS rel_BC, "
        "CASE WHEN e.rel_AB = e.rel_BC THEN 'PROMOTION' ELSE 'SUPPRESSION' END AS gene_disease_effect, "
        "d.name AS disease, d.diseaseSort AS disease_sort "
        f"ORDER BY CASE WHEN toLower(d.name) CONTAINS toLower('{safe_cancer}') THEN 0 ELSE 1 END, d.name, g.name"
    )
    # 注意：后端 _kg 硬约束 1<=limit<=1000，超限直接 ValueError->rejected（不是返回部分结果）
    res = be.call("kg_cypher", {"query": cy, "limit": 1000})
    if res.get("status") != "ok":
        print(f"    [fam query FAILED] status={res.get('status')} err={str(res.get('error'))[:160]} "
              f"| pred={pred[:80]}")
        return []
    return res.get("rows") or []


def family_prior_map(be, fams: set[str], cancer: str, m: dict) -> dict[str, float]:
    """每个亚家族：取其成员（有疾病路径证据者）在本模型下的概率均值作为家族先验。"""
    rows = cap_per_gene(fetch_family_rows(be, fams, cancer), CAP)
    if not rows:
        return {}
    member_genes = sorted({str(r.get("gene", "")).upper() for r in rows})
    acc: dict[str, list[float]] = {}
    for g in member_genes:
        f = gene_graph_feat(g, rows, cancer)
        if f["n_paths"] == 0:
            continue
        fam = subfamily(g)
        if fam not in fams:
            continue
        acc.setdefault(fam, []).append(prob_of(f, m))
    return {fam: sum(v) / len(v) for fam, v in acc.items() if v}


def main():
    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    m = load_model()
    print(f"杠杆3 v20graph_fam | CAP={CAP} | alpha={ALPHA} | 题数={len(packs)}")
    print("=" * 92)

    be = get_default_backend()
    if type(be).__name__ == "UnavailableBackend":
        raise SystemExit("MCP 后端不可用（先 . tools\\slc_mars_env.ps1）")

    rows = []
    for i, pack in enumerate(packs):
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        cancer = (pack.get("constraints") or {}).get("cancer", "")

        ppr_pred = ppr_ranking(pack)
        ppr_block = None if ppr_pred is None else metrics_of(ppr_pred, gt)
        ppr_named = len({g.upper() for g, _s in (pack.get("ppr_candidates") or [])
                         if g.upper() in set(pool)})

        capped = cap_per_gene(fetch_path_rows(be, pool, cancer), CAP)
        if not capped:
            err = "no_gene_disease_paths(不可测)"
            rows.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                             pred=None, ppr=ppr_block, ndcg5=0.0, kendall_tau=0.0,
                             precision5=0.0, hit5=0.0, mrr=0.0, error=err,
                             tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0}))
            print(f"[{i+1:2d}/11] {qid[-24:]:<24} ERR {err}")
            continue

        feats = {g: gene_graph_feat(g, capped, cancer) for g in pool}
        direct = {g: prob_of(feats[g], m) for g in pool}          # 非沉默基因 = v20graph 逐字分
        silent = [g for g in pool if feats[g]["n_paths"] == 0]
        fams = {subfamily(g) for g in silent if subfamily(g)}
        prior = family_prior_map(be, fams, cancer, m) if fams else {}

        probs = dict(direct)
        n_rescued = 0
        for g in silent:
            fam = subfamily(g)
            if fam in prior:
                probs[g] = ALPHA * prior[fam]
                n_rescued += 1

        ranked = sorted(pool, key=lambda g: (-probs[g], g))
        gm = metrics_of(ranked[:5], gt)
        rows.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                         pred=ranked[:5], ppr=ppr_block, seeds=[], n_seed_ok=1, per_seed=[],
                         rationale="", tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0},
                         error="", **gm))
        print(f"[{i+1:2d}/11] {qid[-24:]:<24} pool={len(pool):>2d} silent={len(silent):>2d} "
              f"rescued={n_rescued:>2d}/{len(silent)} fams={sorted(fams)} | "
              f"fam NDCG={gm['ndcg5']:.3f} (v20graph 参考) hit@5={gm['hit5']:.0f}")

    ok = [r for r in rows if not r["error"]]
    json.dump(dict(system="v20graph_fam", hint="none", seeds=[], thinking="OFF", dry_run=False,
                   n=len(rows), n_ok=len(ok), n_err=len(rows) - len(ok),
                   cap=CAP, alpha=ALPHA, results=rows),
              open(TRACES / "topk_eval_v20graph_fam.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n写入 topk_eval_v20graph_fam.json  (ok={len(ok)}/{len(rows)})")


if __name__ == "__main__":
    main()
