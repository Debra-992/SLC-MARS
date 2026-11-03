# -*- coding: utf-8 -*-
"""lever 1 落地：v20full —— 把 v20 二分类一直在用、但 topk 图头从未接入的【组学通道】接进排序头。

与 v20graph（纯 pairwise 图 logistic）的唯一区别是特征空间：
  v20graph 只喂 SLC->Pathway->Disease 的 12 个图链列；
  v20full  额外喂 PPR 邻近 / TIMER / KM / DESeq2 log2FC / 旁系冗余，用全特征 pointwise 必需性模型
           （tools/_topk_fit_full.py 在 calibstore membership 上训练、序列化为 fullfeature_model.json）。

做法（独立 harness，绝不触碰 v20 二分类判定/总表/消融）：
  1. 候选池 pool = PPR top-15 ∪ GT（与 _topk_head / pipeline 完全同池同题，可直接并表）；
  2. 图链特征：经 sanctioned 只读 kg_cypher 取 pool 基因 path 行、按癌种优先逐基因截断 CAP=16
              （与 _topk_head 同一取数与截断，杜绝 out-of-distribution）；
  3. 组学特征：逐题跑真实 ComputeRole(genes=pool, bucket=topk, cancer) 得到 ComputedKnowledge
              —— TIMER/KM 逐基因、DESeq2 癌种级(磁盘缓存)、旁系冗余据 DE 面板派生；
  4. 每基因用 feature_vector.features_for_gene（与 calib_dump._features 同键口径）拼全特征，
              fullfeature 模型打 P(必需) → 降序取前 5；
  5. 指标全部复用 _topk_eval（与 pipeline / pure_llm / v20graph 逐字同口径）。

用法：
  . tools\\slc_mars_env.ps1
  .venv\\Scripts\\python.exe tools\\_topk_full.py            # 跑全部 11 题
"""
from __future__ import annotations
import asyncio, json, math, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))
sys.path.insert(0, str(ROOT / "tools"))

from knowledge_agents.feature_vector import FULL_COLS, vec, features_for_gene
from knowledge_agents.tools.slc_backend import get_default_backend
from knowledge_agents.roles.compute_role import ComputeRole
from knowledge_agents.schema import parse_knowledge, ComputedKnowledge
# 复用 _topk_head 的截断/指标与图谱基线构造，保证与其他臂逐字同口径
from _topk_head import cap_per_gene, metrics_of, ppr_ranking, CAP, PACK_PATH, TRACES

MODEL_PATH = TRACES / "fullfeature_model.json"


async def afetch_path_rows(be, pool: list[str], cancer: str) -> list[dict]:
    """与 _topk_head.fetch_path_rows 同一 cypher，但在 async 环境用 await be.acall
    （sync 的 be.call 内部会再开一个 asyncio.run，在已运行的事件循环里会崩）。"""
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
    res = await be.acall("kg_cypher", {"query": cy, "limit": 1000})
    if res.get("status") != "ok":
        return []
    return res.get("rows") or []


def load_model() -> dict:
    m = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    assert m.get("task") == "fullfeature_membership_pointwise" and m.get("coef"), "fullfeature 工件异常"
    assert m.get("features") == list(FULL_COLS), "fullfeature 特征列序与 feature_vector 不一致"
    return m


def prob_of(x: list[float], m: dict) -> float:
    """与 _topk_fit_full 完全一致的推理：z=(x-mu)/sd → coef·z+b → sigmoid。"""
    mu, sd, coef = m["mu"], m["sd"], m["coef"]
    b = float(m.get("intercept", 0.0))
    lg = b + sum(coef[i] * ((x[i] - mu[i]) / sd[i]) for i in range(len(FULL_COLS)))
    return 1.0 / (1.0 + math.exp(-lg))


async def fetch_ck(role: ComputeRole, qid: str, pool: list[str], cancer: str, question: str) -> ComputedKnowledge | None:
    """逐题跑 ComputeRole（bucket=topk, genes=全池）取组学知识；异常/不可用诚实返回 None。"""
    role.set_task(qid=qid, gene=(pool[0] if pool else None), genes=pool,
                  cancer=cancer, task_bucket="topk")
    try:
        await role.run(f"qid={qid} gene={pool[0] if pool else ''} cancer={cancer} bucket=topk {question}")
        ck = parse_knowledge(role.rc.memory.get()[-1].content)
        return ck if isinstance(ck, ComputedKnowledge) else None
    except Exception as e:  # noqa BLE001
        print(f"    ! ComputeRole 失败 {type(e).__name__}: {str(e)[:60]}")
        return None


async def main():
    pack_data = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    packs = pack_data["packs"]
    m = load_model()
    trained = set(m.get("trained_qids") or [])
    topk_qids = {p["qid"] for p in packs}
    overlap = sorted(trained & topk_qids)
    print(f"v20full ranking head | CAP/gene={CAP} | C={m.get('C')} | 列={len(FULL_COLS)} "
          f"| 题数={len(packs)} | 训练∩topk 题(须为空)={overlap}")
    assert not overlap, "标签泄漏：topk 题出现在全特征训练集"
    print("=" * 92)

    be = get_default_backend()
    if type(be).__name__ == "UnavailableBackend":
        raise SystemExit("MCP 后端不可用（先 . tools\\slc_mars_env.ps1 且确保 Neo4j 可达）")

    role = ComputeRole()
    rows = []
    for i, pack in enumerate(packs):
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        cancer = (pack.get("constraints") or {}).get("cancer", "")
        question = pack.get("question", "")

        ppr_pred = ppr_ranking(pack)
        ppr_block = None if ppr_pred is None else metrics_of(ppr_pred, gt)
        ppr_named = len({g.upper() for g, _s in (pack.get("ppr_candidates") or [])
                         if g.upper() in set(pool)})

        raw = await afetch_path_rows(be, pool, cancer)
        capped = cap_per_gene(raw, CAP)
        ck = await fetch_ck(role, qid, pool, cancer, question)

        if not capped and ck is None:
            err = "no_evidence(不可测)"
            rows.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                             pred=None, ppr=ppr_block, ndcg5=0.0, kendall_tau=0.0,
                             precision5=0.0, hit5=0.0, mrr=0.0, error=err,
                             tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0}))
            print(f"[{i+1:2d}/11] {qid[-24:]:<24} ERR {err}")
            continue

        feats = {g: features_for_gene(g, capped, cancer, ck) for g in pool}
        probs = {g: prob_of(vec(feats[g], FULL_COLS), m) for g in pool}
        ranked = sorted(pool, key=lambda g: (-probs[g], g))
        pred = ranked[:5]
        gm = metrics_of(pred, gt)

        rows.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                         pred=pred, ppr=ppr_block, seeds=[], n_seed_ok=1, per_seed=[],
                         rationale="", tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0},
                         error="", **gm))
        cktag = "ck=Y" if ck is not None else "ck=N"
        deN = (ck.differential_expression and len(ck.differential_expression.top_genes or [])) if ck else 0
        print(f"[{i+1:2d}/11] {qid[-24:]:<24} pool={len(pool):>2d} named={ppr_named:>2d} "
              f"capped={len(capped):>3d} {cktag} de_panel={deN or '-'} | "
              f"v20full NDCG={gm['ndcg5']:.3f} hit@5={gm['hit5']:.0f}")

    ok = [r for r in rows if not r["error"]]
    json.dump(dict(system="v20full", hint="none", seeds=[], thinking="OFF", dry_run=False,
                   n=len(rows), n_ok=len(ok), n_err=len(rows) - len(ok),
                   cap=CAP, model="fullfeature_membership_pointwise", results=rows),
              open(TRACES / "topk_eval_v20full.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n写入 topk_eval_v20full.json  (ok={len(ok)}/{len(rows)})")


if __name__ == "__main__":
    asyncio.run(main())
