# -*- coding: utf-8 -*-
"""Stock MetaGPT baseline: unmodified MetaGPT Role/Action (pure LLM reasoning) on the same questions.

Measured against this project's three-agent pipeline (Kira + Cael + Vega, eight tool evidence channels). Per constraint #3:
  - no CRISPR reference readings are wired in
  - no retrieval and no tool calls at all (the baseline answers from parametric LLM knowledge only)
The output has the same shape as the main pipeline's Answer, so one _score can compare accuracy and coverage directly.

Note: stock MetaGPT needs a reachable LLM. If the endpoint is unavailable (placeholder key here, public OpenAI blocked),
preflight reports the LLM as unusable, every question is recorded as an honest abstain, and a real endpoint is requested.

Usage:
  python -m knowledge_agents.baseline_metagpt --eval 12 --seed 5
  python -m knowledge_agents.baseline_metagpt --qid <qid>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re

from metagpt.actions import Action
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message

from knowledge_agents.answer_pipeline import _score, sample_questions
from knowledge_agents.run_single_question import extract_genes, format_ground_truth
from knowledge_agents.schema import Answer, ToolStatus

SUPPORTED_BUCKETS = {"pairwise", "membership"}

_BASELINE_PROMPT = """You are a biomedical question-answering assistant. Answer from your own knowledge only, using no tools, no retrieval and no external data.
You must commit to one answer (this is a native LLM baseline, abstention is not allowed): pairwise picks exactly one candidate, membership answers true or false.

Question (bucket={bucket}): {question}
{candidates_line}
Output JSON only (abstained is always false):
- pairwise: {{"winner": "<one of the candidate genes>", "confidence": 0~1, "abstained": false, "rationale": "..."}}
- membership: {{"answer": true/false, "confidence": 0~1, "abstained": false, "rationale": "..."}}
"""


def _extract_json(rsp: str) -> dict | None:
    m = re.search(r"\{.*\}", rsp, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class BaselineAnswer(Action):
    """Stock MetaGPT action: a single _aask answers the question."""

    name: str = "BaselineAnswer"

    async def run(self, question: str, *, bucket: str, candidates: list[str], task_type: str = "") -> Answer:
        cline = f"candidate genes: {candidates}" if bucket == "pairwise" else f"target gene: {candidates[:1]}"
        prompt = _BASELINE_PROMPT.format(bucket=bucket, question=question, candidates_line=cline)
        base = Answer(
            question=f"bucket={bucket} candidates={candidates}",
            status=ToolStatus.UNAVAILABLE, summary="",
            bucket=bucket, task_type=task_type, candidates=list(candidates),
            abstained=True, confidence=0.0, decision_mode="llm", used_llm=False,
        )
        timeout = float(os.environ.get("SLC_BASELINE_LLM_TIMEOUT", "30"))
        try:
            rsp = await asyncio.wait_for(self._aask(prompt), timeout=timeout)
        except Exception as e:  # noqa BLE001 -- an unreachable endpoint is recorded honestly as unknown
            base.rationale = f"[the stock MetaGPT baseline needs a reachable LLM; this call failed: {type(e).__name__}]"
            base.error = type(e).__name__
            return base

        data = _extract_json(rsp)
        if not data:
            base.rationale = "[LLM reply did not parse as JSON]"
            return base
        abstained = bool(data.get("abstained")) or (data.get("winner") is None and data.get("answer") is None)
        winner = data.get("winner") if bucket == "pairwise" else None
        answer = data.get("answer") if bucket == "membership" else None
        if winner and winner.upper() not in {c.upper() for c in candidates}:
            winner = None
        return Answer(
            question=base.question,
            status=ToolStatus.OK if not abstained else ToolStatus.UNAVAILABLE,
            summary=str(data.get("rationale", "")),
            bucket=bucket, task_type=task_type, candidates=list(candidates),
            winner=winner, answer=answer, abstained=abstained,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            rationale=str(data.get("rationale", "")), decision_mode="llm", used_llm=True,
        )


class BaselineRole(Role):
    """Stock MetaGPT single agent (pure LLM)."""

    name: str = "Base"
    profile: str = "NativeMetaGPTBaseline"
    goal: str = "answer pairwise/membership from parametric LLM knowledge only (no tools, no retrieval, no CRISPR)"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([BaselineAnswer])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._inputs: dict = {}

    def set_inputs(self, *, bucket, candidates, question, task_type=""):
        self._inputs = {"bucket": bucket, "candidates": candidates, "question": question, "task_type": task_type}

    async def _act(self) -> Message:
        todo: BaselineAnswer = self.rc.todo
        i = self._inputs
        ans = await todo.run(i["question"], bucket=i["bucket"], candidates=i["candidates"], task_type=i["task_type"])
        msg = Message(content=ans.to_message_content(), role=self.profile, cause_by=BaselineAnswer, sent_from=self.name)
        self.rc.memory.add(msg)
        return msg


async def preflight_llm() -> bool:
    role = BaselineRole()
    role.set_inputs(bucket="membership", candidates=["SLC2A1"], question="Is SLC2A1 a glucose transporter?", task_type="membership")
    try:
        await asyncio.wait_for(role.run("preflight"), timeout=float(os.environ.get("SLC_BASELINE_LLM_TIMEOUT", "30")))
    except Exception:  # noqa BLE001
        return False
    from knowledge_agents.schema import parse_knowledge
    ans = parse_knowledge(role.rc.memory.get()[-1].content)
    return bool(getattr(ans, "used_llm", False))


async def run_baseline_eval(n: int, buckets: list[str], seed: int | None, verbose: bool = True) -> None:
    sample = sample_questions(n, buckets, seed)
    ok = await preflight_llm()
    print(f"[baseline] stock MetaGPT, sampled {len(sample)} questions, buckets={buckets}")
    if not ok:
        print(" [!] LLM endpoint unusable (placeholder key / public OpenAI unreachable). Every question will be recorded as abstain, nothing fabricated.")
        print(" To produce a real baseline, configure a usable LLM: Config(llm=... base_url+api_key) or a local gateway.")
    stats = {b: {"correct": 0, "wrong": 0, "abstain": 0} for b in buckets}
    if not ok:
        # LLM unusable: do not burn a 30s timeout per question; record the abstain directly and honestly
        for rec in sample:
            if rec.get("task_bucket", "") in SUPPORTED_BUCKETS:
                stats[rec["task_bucket"]]["abstain"] += 1
        _print_summary(stats, buckets)
        return
    for rec in sample:
        bucket = rec.get("task_bucket", "")
        if bucket not in SUPPORTED_BUCKETS:
            continue
        genes = extract_genes(rec["question"], rec.get("constraints", {}))
        role = BaselineRole()
        role.set_inputs(bucket=bucket, candidates=genes, question=rec["question"], task_type=rec.get("task_type", ""))
        try:
            await role.run("answer")
            from knowledge_agents.schema import parse_knowledge
            ans = parse_knowledge(role.rc.memory.get()[-1].content)
        except Exception as e:  # noqa BLE001
            print(f" ! {rec['qid'][:48]} baseline error: {type(e).__name__}")
            continue
        outcome = _score(ans, rec["_gt"], bucket)
        stats[bucket][outcome] += 1
        if verbose:
            print(f"  {outcome:<7} {bucket:<9} {rec['qid'][:44]:<44} "
                  f"pred(win={ans.winner},ans={ans.answer},c={ans.confidence:.2f}) "
                  f"gt=({format_ground_truth(rec['_gt'], bucket)})")
    _print_summary(stats, buckets)


def _print_summary(stats, buckets):
    print("\n" + "=" * 78)
    print("[baseline summary] stock MetaGPT (pure LLM, no tools/retrieval/CRISPR)")
    tc = tw = ta = 0
    for b in buckets:
        s = stats[b]
        answered = s["correct"] + s["wrong"]
        tot = answered + s["abstain"]
        print(f" {b:<10} correct={s['correct']} wrong={s['wrong']} abstain={s['abstain']} "
              f"| acc (answered)={s['correct']/answered if answered else 0:.1%} "
              f"coverage={answered/tot if tot else 0:.1%} acc (all)={s['correct']/tot if tot else 0:.1%}")
        tc += s["correct"]; tw += s["wrong"]; ta += s["abstain"]
    a = tc + tw
    print(f" overall correct={tc} wrong={tw} abstain={ta} "
          f"| acc (answered)={tc/a if a else 0:.1%} coverage={a/(a+ta) if (a+ta) else 0:.1%}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval", type=int, default=0)
    ap.add_argument("--buckets", default="pairwise,membership")
    args = ap.parse_args()
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]

    if args.eval > 0:
        await run_baseline_eval(args.eval, buckets, args.seed)
        return

    # single question
    from knowledge_agents.answer_pipeline import _gt_index
    from knowledge_agents.run_single_question import QUESTIONS, load_jsonl
    qs = load_jsonl(QUESTIONS)
    rec = next(r for r in qs if r["qid"] == args.qid) if args.qid else sample_questions(1, buckets, args.seed)[0]
    rec = dict(rec); rec["_gt"] = _gt_index().get(rec["qid"], {})
    bucket = rec.get("task_bucket", "")
    genes = extract_genes(rec["question"], rec.get("constraints", {}))
    role = BaselineRole()
    role.set_inputs(bucket=bucket, candidates=genes, question=rec["question"], task_type=rec.get("task_type", ""))
    await role.run("answer")
    from knowledge_agents.schema import parse_knowledge
    ans = parse_knowledge(role.rc.memory.get()[-1].content)
    print("\n" + "=" * 78)
    print(f"[baseline single] {rec['qid']} bucket={bucket} candidates={genes}")
    print(f"  question: {rec['question']}")
    print(f"  ground_truth -> {format_ground_truth(rec['_gt'], bucket)}")
    print(f"  ==> pred winner={ans.winner} answer={ans.answer} abstained={ans.abstained} "
          f"used_llm={ans.used_llm} conf={ans.confidence}")
    print(f"  rationale: {ans.rationale}")


if __name__ == "__main__":
    asyncio.run(main())
