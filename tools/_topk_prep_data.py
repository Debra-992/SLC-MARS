# -*- coding: utf-8 -*-
"""topk LLM 重排评测数据准备：对 11 道 topk 题跑 PPR 计算 → PPR top-15 + GT 全并集 → 输出评测包。

输出: outputs/traces/topk_eval_pack.json
每道题包含:
  - qid, question, constraints
  - gt_ranking: GT 有序 5 基因
  - ppr_candidates: PPR 全列表 (gene, score) 按 PPR 降序
  - eval_pool: 供 LLM 重排的候选基因列表 (PPR top-15 ∪ GT, 当前为 PPR 序、评测时会 shuffle)
  - gt_in_pool: GT 基因在 eval_pool 中的位置索引
  - ndcg_ideal_dcg: 用于 NDCG 归一化的理想 DCG@5
  - ppr_ndcg@5: PPR 本身的 NDCG@5 vs GT（基线）

用法: . tools\slc_mars_env.ps1 ; .venv\Scripts\python.exe tools\_topk_prep_data.py
"""
from __future__ import annotations
import asyncio, json, math, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.clean_questions import load_jsonl
from knowledge_agents.roles.compute_role import ComputeRole
from knowledge_agents.schema import parse_knowledge, ComputedKnowledge

OUT = ROOT / "outputs" / "traces" / "topk_eval_pack.json"


# ---- 指标: NDCG@k ----
def compute_ndcg(pred_order: list[str], gt_ideal: list[str], k: int = 5) -> float:
    """pred_order: 预测排序（仅基因名）; gt_ideal: GT 理想排序.
    relevance(g) = 0 if g not in gt_ideal, else 1/log2(gt_rank+2)."""
    gt_rank = {g: i for i, g in enumerate(gt_ideal)}
    rel = {g: (1.0 / math.log2(gt_rank[g] + 2)) for g in gt_ideal}
    dcg = sum(rel.get(g, 0) / math.log2(i + 2) for i, g in enumerate(pred_order[:k]))
    ideal_scores = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal_scores))
    return dcg / idcg if idcg > 0 else 0.0


async def get_ppr(q: dict) -> tuple[list[tuple[str, float]], str | None]:
    """返回 (gene, ppr_score) 降序列表，及错误信息（若失败）。"""
    from knowledge_agents.roles.compute_role import ComputeRole
    from knowledge_agents.schema import parse_knowledge, ComputedKnowledge
    role = ComputeRole()
    c = q.get("constraints", {})
    cancer = c.get("cancer", "")
    role.set_task(qid=q["qid"], gene=None, genes=[], cancer=cancer, task_bucket="topk")
    try:
        await role.run(f"gene=None cancer={cancer} bucket=topk {q['question']}")
    except Exception as e:
        return [], str(e)
    ck = parse_knowledge(role.rc.memory.get()[-1].content)
    gc = (ck.graph_computation if isinstance(ck, ComputedKnowledge) else None)
    if not gc or not gc.candidates:
        return [], "no PPR output"
    return [(c.gene, c.degree_corrected_score) for c in gc.candidates], None


async def main():
    qs = [r for r in load_jsonl(ROOT / "data" / "questions_clean.jsonl")
          if r.get("task_bucket") == "topk"]
    gts = {r["qid"]: r.get("ground_truth", [])
           for r in load_jsonl(ROOT / "evaluation" / "ground_truth_clean.jsonl")}

    packs = []
    for i, q in enumerate(qs):
        qid = q["qid"]
        gt = gts.get(qid, [])
        print(f"[{i+1:2d}/{len(qs)}] {qid} ...", end=" ", flush=True)
        cands, err = await get_ppr(q)
        if err:
            print(f"ERR: {err}")
            # 即使 PPR 失败也产出 fallback：eval_pool = GT 基因本身（5 基因排序，NDCG 仍有效）
            pool_genes = gt
            gt_in_pool = [(g, i) for i, g in enumerate(gt)]
            ideal_dcg = sum((1.0 / math.log2(i + 2)) / math.log2(i + 2) for i in range(min(5, len(gt))))
            packs.append(dict(qid=qid, question=q["question"],
                              constraints=q.get("constraints", {}),
                              gt_ranking=gt, ppr_candidates=[],
                              eval_pool=pool_genes,
                              gt_in_pool=[list(t) for t in gt_in_pool],
                              ndcg_ideal_dcg=ideal_dcg,
                              ppr_baseline_ndcg5=None,
                              error=err, fallback="PPR unavailable, GT-only pool"))
            continue

        # PPR top-15 + GT union
        ppr_all = [c[0] for c in cands]
        ppr_top15 = ppr_all[:15]
        pool_genes = list(dict.fromkeys(ppr_top15 + [g for g in gt if g not in ppr_top15]))
        # 评测时 shuffle；这里保持 PPR 序含 GT 补位，方便人类审阅
        gt_in_pool = [(g, pool_genes.index(g)) for g in gt if g in pool_genes]

        # PPR baseline NDCG
        ppr_ndcg = compute_ndcg(ppr_all, gt, k=5)
        # ideal DCG for GT (用于后续评测归一化)
        ideal_dcg = sum((1.0 / math.log2(i + 2)) / math.log2(i + 2)
                        for i in range(min(5, len(gt))))

        print(f"PPR={len(cands)} genes  pool={len(pool_genes)}  GT-in-pool={len(gt_in_pool)}/5  "
              f"PPR_NDCG@5={ppr_ndcg:.3f}")

        packs.append(dict(
            qid=qid,
            question=q["question"],
            constraints=q.get("constraints", {}),
            task_type=q.get("task_type", ""),
            gt_ranking=gt,
            ppr_candidates=[list(c) for c in cands[:30]],    # 只存 top-30 紧凑
            ppr_candidate_count=len(cands),
            eval_pool=pool_genes,                              # PPR top-15 ∪ GT
            gt_in_pool=[list(t) for t in gt_in_pool],
            ndcg_ideal_dcg=ideal_dcg,
            ppr_baseline_ndcg5=ppr_ndcg,
        ))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"packs": packs, "meta": {"n": len(packs), "description":
               "topk LLM re-rank eval data: PPR top-15 ∪ GT genes as eval pool. "
               "PPR baseline NDCG@5 shown for reference."}},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n写入 {OUT}")
    print(f"PPR NDCG@5 基线均值: {sum(p.get('ppr_baseline_ndcg5',0) for p in packs)/len(packs):.3f}")


if __name__ == "__main__":
    asyncio.run(main())