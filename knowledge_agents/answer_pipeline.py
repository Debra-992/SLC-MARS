# -*- coding: utf-8 -*-
"""端到端答题管线：用 MetaGPT 三智能体协作把 pairwise / membership 问题跑出答案。

协作链（复用 MetaGPT Role/Action/Message 基座）：
    题目 -> Kira(知识检索) -> Cael(计算) -> Vega(决策) -> Answer
三者各产一种结构化知识（BackgroundKnowledge / ComputedKnowledge / Answer），
Vega 只综合前两路已产出的证据、不重新取数；证据不足则弃答（unknown != negative）。

用法：
  python -m knowledge_agents.answer_pipeline --qid <qid>        # 单题全过程
  python -m knowledge_agents.answer_pipeline --seed 7           # 随机单题
  python -m knowledge_agents.answer_pipeline --eval 20 --buckets pairwise,membership  # 批量评测
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re

from knowledge_agents.run_single_question import (
    GTRUTH,
    QUESTIONS,
    extract_genes,
    format_ground_truth,
    load_jsonl,
)
from knowledge_agents.roles.compute_role import ComputeRole
from knowledge_agents.roles.decision_role import DecisionRole
from knowledge_agents.roles.knowledge_query_role import KnowledgeQueryRole
from knowledge_agents.schema import Answer, BackgroundKnowledge, ComputedKnowledge, parse_knowledge

SUPPORTED_BUCKETS = {"pairwise", "membership"}


class AnswerTeam:
    """Kira ∥ Cael 并行供给 -> Vega 决策 的 MetaGPT 团队编排。

    Kira(知识检索) 与 Cael(计算) 相互独立，用 asyncio.gather 并行；两者都回来后
    再由 Vega 综合出 Answer。
    """

    def __init__(self):
        self.kira = KnowledgeQueryRole()
        self.cael = ComputeRole()
        self.vega = DecisionRole()

    @staticmethod
    def _effective_genes(record: dict) -> tuple[list[str], str | None]:
        """pairwise=题面两基因；membership=只取 'Is/Does <GENE>' 子句真正在问的基因。

        membership 题面 treatment 里的 'vs SLCxxx.KO'（=qid 基因）是筛选敲除的
        另一个基因、非判定对象，必须排除，否则会把被敲除基因误当成问询对象。
        """
        question = record["question"]
        bucket = record.get("task_bucket", "")
        genes = extract_genes(question, record.get("constraints", {}))
        gene = genes[0] if genes else None
        if bucket == "membership":
            m = re.search(r"\b(?:Is|Does)\s+(SLC[A-Z0-9]+)\b", question, re.I)
            target = m.group(1).upper() if m else gene
            genes = [target] if target else []
            gene = target
        return genes, gene

    async def supply(self, record: dict) -> dict:
        """只跑 Kira∥Cael（并行），返回两路证据 + 判定入参，供 decide() 复用。"""
        qid = record["qid"]
        question = record["question"]
        constraints = record.get("constraints", {})
        bucket = record.get("task_bucket", "")
        task_type = record.get("task_type", "")
        genes, gene = self._effective_genes(record)
        cancer = constraints.get("cancer", "")

        self.cael.set_task(qid=qid, gene=gene, genes=genes, cancer=cancer, task_bucket=bucket)
        # v11：Prior 智能体与 Kira∥Cael 并行（env SLC_PRIOR_AGENT 门控，关闭时行为=旧版）
        # v16：SLC_TRIAD_VOTE 开时 L2（角色异构 persona）也在 supply 段并行抽出（prior2），
        #      消掉决策层串行等待；SLC_PRIOR_RACE>1 时 L1/L2 走竞速（首个有效票）。
        # v17：SLC_PRIOR_RETRIEVE 开时 L2 额外挂项目文献库紧凑摘录（仅 L2，L1/证据通道不变）。
        from knowledge_agents import prior_agent
        prior2 = None
        if prior_agent.prior_enabled():
            triad = os.environ.get("SLC_TRIAD_VOTE", "0").lower() in {"1", "true", "on", "yes"}
            p1_coro = (prior_agent.ask_prior_ensemble(question, bucket, genes)
                       if prior_agent.prior_ensemble_n() > 1
                       else prior_agent.ask_prior_raced(question, bucket, genes))
            coros = [self.kira.run(f"qid={qid} {question}"),
                     self.cael.run(f"gene={gene or ''} cancer={cancer} bucket={bucket} {question}"),
                     p1_coro]
            if triad:
                # v17：SLC_PRIOR_RETRIEVE 开时 L2 票挂轻检索文学包（票源去同源化）；
                # 构建失败/门控关 -> pack="" 逐字节同 v16。L1 保持裸记忆，三票信息源各自独立。
                pack = ""
                if prior_agent.prior_retrieve_enabled():
                    pack = await asyncio.to_thread(
                        prior_agent.build_retrieval_pack,
                        question, bucket, genes, constraints)
                coros.append(prior_agent.ask_prior_raced(
                    question, bucket, genes,
                    persona=prior_agent.ROLE_PERSONAS["L2"], temperature=0.7,
                    retrieval=pack))
            got = await asyncio.gather(*coros)
            kg, ck, prior = got[0], got[1], got[2]
            prior2 = got[3] if triad else None
        else:
            prior = None
            kg, ck = await asyncio.gather(
                self.kira.run(f"qid={qid} {question}"),
                self.cael.run(f"gene={gene or ''} cancer={cancer} bucket={bucket} {question}"),
            )
        kg = parse_knowledge(self.kira.rc.memory.get()[-1].content)
        ck = parse_knowledge(self.cael.rc.memory.get()[-1].content)
        assert isinstance(kg, BackgroundKnowledge) and isinstance(ck, ComputedKnowledge)
        # v18 ②：membership 负证据通道——文献共现富集/旁系对比做成合成 KGEvidence 挂进
        #   kg.evidence（Vega/法官/事实摘要零签名改动全可见）；失败/无数据 => 逐字节 v16。
        from knowledge_agents import negative_evidence
        if negative_evidence.neg_evid_enabled() and bucket == "membership" and gene and cancer:
            partners = ((ck.paralog_partners or {}).get(str(gene).upper())
                        or (ck.paralog_partners or {}).get(str(gene)) or [])
            rows = await asyncio.to_thread(negative_evidence.cooc_contrast_rows,
                                           str(gene), str(cancer), list(partners)[:2])
            if rows:
                from knowledge_agents.schema import KGEvidence
                kg.evidence.append(KGEvidence(operation="lit_cooc_contrast", rows=rows))
        return {"kg": kg, "ck": ck, "genes": genes, "gene": gene, "bucket": bucket,
                "task_type": task_type, "question": question, "constraints": constraints,
                "prior": prior, "prior2": prior2}

    async def decide(self, supplied: dict, mode: str | None = None) -> Answer:
        """用已产出的证据跑 Vega 决策。mode 覆盖 SLC_DECISION_MODE（heuristic|llm|auto）。"""
        prev = os.environ.get("SLC_DECISION_MODE")
        if mode is not None:
            os.environ["SLC_DECISION_MODE"] = mode
        try:
            self.vega.set_inputs(
                kg=supplied["kg"], ck=supplied["ck"], bucket=supplied["bucket"],
                candidates=supplied["genes"], question=supplied["question"],
                task_type=supplied["task_type"], context=supplied["constraints"],
                prior=supplied.get("prior"), prior2=supplied.get("prior2"),
            )
            await self.vega.run(
                f"bucket={supplied['bucket']} candidates={supplied['genes']} {supplied['question']}")
        finally:
            if prev is None:
                os.environ.pop("SLC_DECISION_MODE", None)
            else:
                os.environ["SLC_DECISION_MODE"] = prev
        ans = parse_knowledge(self.vega.rc.memory.get()[-1].content)
        assert isinstance(ans, Answer)
        # ① env 门控置信度校准层：SLC_CONF_CALIB=1 时按 gene_margin+路径重标定 conf；
        #   只改 confidence，不动 winner/answer/abstained。默认关闭 => 逐字节 v16。
        from knowledge_agents import conf_calib
        return conf_calib.apply_calibration(ans)

    async def ask(self, record: dict, *, verbose: bool = True) -> dict:
        bucket = record.get("task_bucket", "")
        if bucket not in SUPPORTED_BUCKETS:
            return {"qid": record["qid"], "bucket": bucket, "supported": False,
                    "answer": None, "trace": {"kg": None, "ck": None, "ans": None}}

        supplied = await self.supply(record)
        ans = await self.decide(supplied)  # 用当前 SLC_DECISION_MODE（默认 auto）

        if verbose:
            _print_trace(record, supplied["kg"], supplied["ck"], ans, supplied["genes"])
        return {
            "qid": record["qid"], "bucket": bucket, "supported": True,
            "answer": ans, "ground_truth": record.get("_gt", {}),
            "trace": {"kg": supplied["kg"], "ck": supplied["ck"], "ans": ans},
        }


def _print_trace(record: dict, kg, ck, ans: Answer, genes: list[str]) -> None:
    print("\n" + "=" * 78)
    print(f"[题目] {record['qid']}  bucket={record.get('task_bucket')}  genes={genes}")
    print(f"  question: {record['question']}")
    print(f"  ground_truth -> {format_ground_truth(record.get('_gt', {}), record.get('task_bucket', ''))}")
    print("-" * 78)
    print(f"[Kira 知识检索] status={kg.status.value} 证据条数={len(kg.evidence)} "
          f"cypher={'Y' if kg.cypher else '-'}")
    print(f"[Cael 计算]     status={ck.status.value} "
          f"ppr={'Y' if ck.graph_computation else '-'} timer_genes={len(ck.expression_by_gene)} "
          f"km_genes={len(ck.survival_by_gene)} de={'Y' if ck.differential_expression else '-'}")
    print(f"[Vega 决策]      mode={ans.decision_mode} used_llm={ans.used_llm}")
    print(f"    基因打分 = {json.dumps(ans.gene_scores, ensure_ascii=False)}")
    print(f"    证据通道 = {ans.basis}")
    if ans.analogy_cancers:
        print(f"    临近癌种 = {ans.analogy_cancers}")
    print(f"    ==> 答案 : winner={ans.winner} answer={ans.answer} "
          f"abstained={ans.abstained} confidence={ans.confidence}")
    print(f"    依据     : {ans.rationale}")


# ------------------------------------------------------------------ 评测
def _score(pred: Answer, gt: dict, bucket: str) -> str:
    """返回 'correct' | 'wrong' | 'abstain'。"""
    if pred.abstained or (bucket == "pairwise" and not pred.winner) or (bucket == "membership" and pred.answer is None):
        return "abstain"
    if bucket == "pairwise":
        return "correct" if str(pred.winner).upper() == str(gt.get("winner", "")).upper() else "wrong"
    # membership
    return "correct" if bool(pred.answer) == bool(gt.get("answer")) else "wrong"


def _gt_index() -> dict:
    return {r["qid"]: r.get("ground_truth", {}) for r in load_jsonl(GTRUTH)}


def sample_questions(n: int, buckets: list[str], seed: int | None) -> list[dict]:
    """确定性抽样：同 (n, buckets, seed) 在管线与 baseline 间得到完全相同的题目。"""
    questions = [q for q in load_jsonl(QUESTIONS) if q.get("task_bucket") in buckets]
    gt_index = _gt_index()
    rng = random.Random(seed)
    picked = rng.sample(questions, min(n, len(questions)))
    out = []
    for rec in picked:
        rec = dict(rec)
        rec["_gt"] = gt_index.get(rec["qid"], {})
        out.append(rec)
    return out


async def run_eval(n: int, buckets: list[str], seed: int | None) -> None:
    sample = sample_questions(n, buckets, seed)
    team = AnswerTeam()

    stats = {b: {"correct": 0, "wrong": 0, "abstain": 0} for b in buckets}
    print(f"[评测] 抽样 {len(sample)} 题，桶={buckets}（mode={__import__('os').environ.get('SLC_DECISION_MODE','auto')}）")
    for rec in sample:
        try:
            res = await team.ask(rec, verbose=False)
        except Exception as e:  # noqa BLE001
            print(f"  ! {rec['qid']} 管线异常: {type(e).__name__}: {str(e)[:80]}")
            continue
        if not res["supported"]:
            continue
        ans = res["answer"]
        outcome = _score(ans, rec["_gt"], res["bucket"])
        stats[res["bucket"]][outcome] += 1
        print(f"  {outcome:<7} {res['bucket']:<9} {rec['qid'][:44]:<44} "
              f"[{ans.decision_mode}] pred(win={ans.winner},ans={ans.answer},c={ans.confidence:.2f}) "
              f"gt=({format_ground_truth(rec['_gt'], res['bucket'])})")

    print("\n" + "=" * 78)
    print("[评测汇总]")
    total_c = total_w = total_a = 0
    for b, s in stats.items():
        answered = s["correct"] + s["wrong"]
        tot = answered + s["abstain"]
        acc_ans = s["correct"] / answered if answered else 0.0
        acc_all = s["correct"] / tot if tot else 0.0
        cov = answered / tot if tot else 0.0
        print(f"  {b:<10} 答对={s['correct']} 答错={s['wrong']} 弃答={s['abstain']} "
              f"| 准确率(已答)={acc_ans:.1%} 覆盖率={cov:.1%} 准确率(全体)={acc_all:.1%}")
        total_c += s["correct"]; total_w += s["wrong"]; total_a += s["abstain"]
    ta = total_c + total_w
    print(f"  总体       答对={total_c} 答错={total_w} 弃答={total_a} "
          f"| 准确率(已答)={total_c / ta if ta else 0:.1%} "
          f"覆盖率={ta / (ta + total_a) if (ta + total_a) else 0:.1%}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval", type=int, default=0, help="批量评测 N 题（0=单题）")
    ap.add_argument("--buckets", default="pairwise,membership")
    ap.add_argument("--n", type=int, default=1, help="单题模式下无 --qid 时抽几题")
    args = ap.parse_args()

    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]

    if args.eval > 0:
        await run_eval(args.eval, buckets, args.seed)
        return

    questions = load_jsonl(QUESTIONS)
    gt_index = _gt_index()
    team = AnswerTeam()

    if args.qid:
        picked = [next(r for r in questions if r["qid"] == args.qid)]
    else:
        pool = [q for q in questions if q.get("task_bucket") in buckets] or questions
        rng = random.Random(args.seed)
        picked = rng.sample(pool, min(args.n, len(pool)))

    for rec in picked:
        rec = dict(rec)
        rec["_gt"] = gt_index.get(rec["qid"], {})
        await team.ask(rec, verbose=True)


if __name__ == "__main__":
    asyncio.run(main())
