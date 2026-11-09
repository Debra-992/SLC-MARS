# -*- coding: utf-8 -*-
"""End-to-end answering pipeline: three MetaGPT agents collaborate to answer pairwise / membership
questions.

Collaboration chain (built on the MetaGPT Role/Action/Message substrate):
    question -> Kira (knowledge retrieval) -> Cael (computation) -> Vega (decision) -> Answer
Each of the three produces one kind of structured knowledge (BackgroundKnowledge /
ComputedKnowledge / Answer). Vega only synthesises the evidence the other two already produced and
never re-fetches; if the evidence is insufficient it abstains (unknown != negative).

Usage:
  python -m knowledge_agents.answer_pipeline --qid <qid>        # one question, full trace
  python -m knowledge_agents.answer_pipeline --seed 7           # one random question
  python -m knowledge_agents.answer_pipeline --eval 20 --buckets pairwise,membership  # batch evaluation
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
    """MetaGPT team orchestration of Kira || Cael supplying in parallel and Vega deciding.

    Kira (knowledge retrieval) and Cael (computation) are independent and run under
    asyncio.gather; once both return, Vega synthesises the Answer.
    """

    def __init__(self):
        self.kira = KnowledgeQueryRole()
        self.cael = ComputeRole()
        self.vega = DecisionRole()

    @staticmethod
    def _effective_genes(record: dict) -> tuple[list[str], str | None]:
        """pairwise takes the two genes in the question; membership takes only the gene the
        'Is/Does <GENE>' clause is actually asking about.

        In a membership question, the 'vs SLCxxx.KO' inside the treatment (the qid gene) is the
        knockout used to screen the other gene, not the subject of the verdict, so it must be
        excluded -- otherwise the knocked-out gene would be mistaken for the one being asked about.
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
        """Run only Kira||Cael (in parallel) and return both evidence channels plus the decision inputs,
        so decide() can reuse them."""
        qid = record["qid"]
        question = record["question"]
        constraints = record.get("constraints", {})
        bucket = record.get("task_bucket", "")
        task_type = record.get("task_type", "")
        genes, gene = self._effective_genes(record)
        cancer = constraints.get("cancer", "")

        self.cael.set_task(qid=qid, gene=gene, genes=genes, cancer=cancer, task_bucket=bucket)
        # The prior agent runs in parallel with Kira||Cael, gated by SLC_PRIOR_AGENT (behaviour with
        # the gate off is the pipeline without a prior). When SLC_TRIAD_VOTE is on, L2 (the
        # role-heterogeneous persona) is also drawn in parallel inside the supply stage (prior2),
        # which removes the serial wait at decision time; with SLC_PRIOR_RACE>1, L1/L2 race for the
        # first valid vote. With SLC_PRIOR_RETRIEVE on, L2 additionally carries a compact excerpt pack
        # from the project literature corpus (L2 only; L1 and the evidence channels are unchanged).
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
                # With SLC_PRIOR_RETRIEVE on, the L2 vote carries a light retrieval pack so its
                # source is decorrelated; if the pack cannot be built, or the gate is off, pack=""
                # and the behaviour is exactly the released one. L1 keeps its bare memory, so the
                # three votes rest on independent information sources.
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
        # Membership negative-evidence channel: literature co-occurrence enrichment and the paralogue
        #   contrast become a synthetic KGEvidence appended to kg.evidence, so Vega, the judge and the
        #   fact digest all see it with zero signature change; failure or no data falls back exactly to
        #   the released behaviour.
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
        """Run the Vega decision on the evidence already produced. mode overrides SLC_DECISION_MODE
        (heuristic|llm|auto)."""
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
        # Environment-gated confidence calibration layer: with SLC_CONF_CALIB=1, conf is recalibrated
        #   from gene_margin plus the decision path. It changes confidence only, never
        #   winner/answer/abstained. Off by default, i.e. exactly the released behaviour.
        from knowledge_agents import conf_calib
        return conf_calib.apply_calibration(ans)

    async def ask(self, record: dict, *, verbose: bool = True) -> dict:
        bucket = record.get("task_bucket", "")
        if bucket not in SUPPORTED_BUCKETS:
            return {"qid": record["qid"], "bucket": bucket, "supported": False,
                    "answer": None, "trace": {"kg": None, "ck": None, "ans": None}}

        supplied = await self.supply(record)
        ans = await self.decide(supplied)  # with the current SLC_DECISION_MODE (auto by default)

        if verbose:
            _print_trace(record, supplied["kg"], supplied["ck"], ans, supplied["genes"])
        return {
            "qid": record["qid"], "bucket": bucket, "supported": True,
            "answer": ans, "ground_truth": record.get("_gt", {}),
            "trace": {"kg": supplied["kg"], "ck": supplied["ck"], "ans": ans},
        }


def _print_trace(record: dict, kg, ck, ans: Answer, genes: list[str]) -> None:
    print("\n" + "=" * 78)
    print(f"[question] {record['qid']}  bucket={record.get('task_bucket')}  genes={genes}")
    print(f"  question: {record['question']}")
    print(f"  ground_truth -> {format_ground_truth(record.get('_gt', {}), record.get('task_bucket', ''))}")
    print("-" * 78)
    print(f"[Kira retrieval]  status={kg.status.value} evidence_rows={len(kg.evidence)} "
          f"cypher={'Y' if kg.cypher else '-'}")
    print(f"[Cael compute]    status={ck.status.value} "
          f"ppr={'Y' if ck.graph_computation else '-'} timer_genes={len(ck.expression_by_gene)} "
          f"km_genes={len(ck.survival_by_gene)} de={'Y' if ck.differential_expression else '-'}")
    print(f"[Vega decision]   mode={ans.decision_mode} used_llm={ans.used_llm}")
    print(f"    gene scores = {json.dumps(ans.gene_scores, ensure_ascii=False)}")
    print(f"    evidence channels = {ans.basis}")
    if ans.analogy_cancers:
        print(f"    neighbouring cancers = {ans.analogy_cancers}")
    print(f"    ==> answer: winner={ans.winner} answer={ans.answer} "
          f"abstained={ans.abstained} confidence={ans.confidence}")
    print(f"    rationale: {ans.rationale}")


# ------------------------------------------------------------------ evaluation
def _score(pred: Answer, gt: dict, bucket: str) -> str:
    """Return 'correct' | 'wrong' | 'abstain'."""
    if pred.abstained or (bucket == "pairwise" and not pred.winner) or (bucket == "membership" and pred.answer is None):
        return "abstain"
    if bucket == "pairwise":
        return "correct" if str(pred.winner).upper() == str(gt.get("winner", "")).upper() else "wrong"
    # membership
    return "correct" if bool(pred.answer) == bool(gt.get("answer")) else "wrong"


def _gt_index() -> dict:
    return {r["qid"]: r.get("ground_truth", {}) for r in load_jsonl(GTRUTH)}


def sample_questions(n: int, buckets: list[str], seed: int | None) -> list[dict]:
    """Deterministic sampling: the same (n, buckets, seed) yields exactly the same questions for the
    pipeline and for the baselines."""
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
    print(f"[evaluation] sampled {len(sample)} questions, buckets={buckets} (mode={__import__('os').environ.get('SLC_DECISION_MODE','auto')})")
    for rec in sample:
        try:
            res = await team.ask(rec, verbose=False)
        except Exception as e:  # noqa BLE001
            print(f"  ! {rec['qid']} pipeline exception: {type(e).__name__}: {str(e)[:80]}")
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
    print("[evaluation summary]")
    total_c = total_w = total_a = 0
    for b, s in stats.items():
        answered = s["correct"] + s["wrong"]
        tot = answered + s["abstain"]
        acc_ans = s["correct"] / answered if answered else 0.0
        acc_all = s["correct"] / tot if tot else 0.0
        cov = answered / tot if tot else 0.0
        print(f" {b:<10} correct={s['correct']} wrong={s['wrong']} abstain={s['abstain']} "
              f"| accuracy(answered)={acc_ans:.1%} coverage={cov:.1%} accuracy(all)={acc_all:.1%}")
        total_c += s["correct"]; total_w += s["wrong"]; total_a += s["abstain"]
    ta = total_c + total_w
    print(f"  OVERALL      correct={total_c} wrong={total_w} abstain={total_a} "
          f"| accuracy(answered)={total_c / ta if ta else 0:.1%} "
          f"coverage={ta / (ta + total_a) if (ta + total_a) else 0:.1%}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval", type=int, default=0, help="batch-evaluate N questions (0 = single question)")
    ap.add_argument("--buckets", default="pairwise,membership")
    ap.add_argument("--n", type=int, default=1, help="in single-question mode with no --qid, how many questions to draw")
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
