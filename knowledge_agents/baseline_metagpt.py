# -*- coding: utf-8 -*-
"""原生 MetaGPT baseline：用 stock MetaGPT Role/Action（纯 LLM 推理）回答同样的题目。

对照本项目的三智能体管线（Kira+Cael+Vega 有 8 工具证据）。按约束 #3：
  - 不接入 CRISPR 基准读数
  - 不进行任何检索 / 工具调用（baseline 只凭 LLM 参数化知识答题）
输出与主管线同构的 Answer，便于同一 _score 直接对比准确率/覆盖率。

注意：原生 MetaGPT 依赖可用 LLM。若端点不可达（本环境 key 为占位、公网 OpenAI 被墙），
preflight 会判定 LLM 不可用，baseline 每题记 abstain（不伪造），并提示需要真实端点。

用法：
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

_BASELINE_PROMPT = """你是生物医学问答助手。仅凭你自己的知识回答，不使用任何工具、检索或外部数据。
必须给出一个明确答案（这是原生 LLM 基线，不允许 abstain）：pairwise 必须二选一，membership 必须 true 或 false。

问题(桶={bucket}): {question}
{candidates_line}
请只输出 JSON（abstained 恒为 false）：
- pairwise: {{"winner": "<候选基因之一>", "confidence": 0~1, "abstained": false, "rationale": "..."}}
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
    """原生 MetaGPT 动作：一次 _aask 完成答题。"""

    name: str = "BaselineAnswer"

    async def run(self, question: str, *, bucket: str, candidates: list[str], task_type: str = "") -> Answer:
        cline = f"候选基因: {candidates}" if bucket == "pairwise" else f"目标基因: {candidates[:1]}"
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
        except Exception as e:  # noqa BLE001 —— LLM 不可达即如实记未知
            base.rationale = f"[原生 MetaGPT baseline 需要可用 LLM；本次调用失败: {type(e).__name__}]"
            base.error = type(e).__name__
            return base

        data = _extract_json(rsp)
        if not data:
            base.rationale = "[LLM 返回无法解析为 JSON]"
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
    """原生 MetaGPT 单智能体（纯 LLM）。"""

    name: str = "Base"
    profile: str = "NativeMetaGPTBaseline"
    goal: str = "仅用 LLM 参数化知识回答 pairwise/membership（不用工具/检索/CRISPR）"

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
    print(f"[baseline] 原生 MetaGPT，抽样 {len(sample)} 题，桶={buckets}")
    if not ok:
        print("  ⚠ LLM 端点不可用（key 为占位/公网 OpenAI 不可达）。每题将记 abstain，不伪造。")
        print("     要跑出真实 baseline，请配置可用 LLM：Config(llm=... base_url+api_key) 或本地网关。")
    stats = {b: {"correct": 0, "wrong": 0, "abstain": 0} for b in buckets}
    if not ok:
        # LLM 不可用：不逐题空跑 30s 超时，直接如实记 abstain（不伪造）
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
            print(f"  ! {rec['qid'][:48]} baseline 异常: {type(e).__name__}")
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
    print("[baseline 汇总] 原生 MetaGPT（纯 LLM，无工具/检索/CRISPR）")
    tc = tw = ta = 0
    for b in buckets:
        s = stats[b]
        answered = s["correct"] + s["wrong"]
        tot = answered + s["abstain"]
        print(f"  {b:<10} 答对={s['correct']} 答错={s['wrong']} 弃答={s['abstain']} "
              f"| 准确率(已答)={s['correct']/answered if answered else 0:.1%} "
              f"覆盖率={answered/tot if tot else 0:.1%} 准确率(全体)={s['correct']/tot if tot else 0:.1%}")
        tc += s["correct"]; tw += s["wrong"]; ta += s["abstain"]
    a = tc + tw
    print(f"  总体       答对={tc} 答错={tw} 弃答={ta} "
          f"| 准确率(已答)={tc/a if a else 0:.1%} 覆盖率={a/(a+ta) if (a+ta) else 0:.1%}")


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

    # 单题
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
    print(f"[baseline 单题] {rec['qid']} bucket={bucket} candidates={genes}")
    print(f"  question: {rec['question']}")
    print(f"  ground_truth -> {format_ground_truth(rec['_gt'], bucket)}")
    print(f"  ==> pred winner={ans.winner} answer={ans.answer} abstained={ans.abstained} "
          f"used_llm={ans.used_llm} conf={ans.confidence}")
    print(f"  rationale: {ans.rationale}")


if __name__ == "__main__":
    asyncio.run(main())
