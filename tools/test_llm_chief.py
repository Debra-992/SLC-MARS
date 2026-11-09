# -*- coding: utf-8 -*-
"""Offline unit test for the LLM chief-judge fusion layer (mock LLM, no endpoint needed). It verifies:
  1) the agree branch is a free win with zero extra calls;
  2) the abstention branch calls the evidence-augmented LLM (with the factual review material),
     decision_mode=+prior and the provenance names the augmented re-answer;
  3) the augmented LLM abstaining -> fall back to the bare prior;
  4) the conflict branch goes to the LLM judge, decision_mode=+judge and conformal is skipped;
  5) judge absent -> the prior takes priority (no longer the evidence channel);
  6) SLC_LLM_CHIEF off -> the plain debate behaviour is unchanged (regression).
"""
import asyncio
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from knowledge_agents import prior_agent
from knowledge_agents.actions.decision_action import AnswerDecision
from knowledge_agents.schema import (Answer, BackgroundKnowledge, ComputedKnowledge,
                                     ToolStatus)

os.environ["SLC_PRIOR_AGENT"] = "1"
os.environ["SLC_LLM_CHIEF"] = "1"
# Pin the semantics under test: the vote layer must be off (the released environment defaults TRIAD=1,
# so the test has to disable it explicitly).
os.environ["SLC_TRIAD_VOTE"] = "0"
os.environ["SLC_TRIAD_CRITIQUE"] = "0"
# Pin the semantics under test: the debate layer must be off (the released environment defaults DEBATE=1,
# and scenario 6 asserts the no-debate fallback path).
os.environ["SLC_DEBATE"] = "0"
ad = AnswerDecision()
ok = True


def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


calls = []


def mk_aug(answer=None, winner=None, abstained=False):
    async def _f(question, bucket, candidates, digest, conflict_hint=""):
        calls.append(("aug", digest[:20]))
        if abstained:
            return {"winner": None, "answer": None, "abstained": True, "confidence": 0}
        return {"winner": winner, "answer": answer, "abstained": False,
                "confidence": 0.7, "rationale": "facts+memory"}
    return _f


def mk_judge(answer=None, winner=None, none=False):
    async def _f(question, bucket, candidates, evidence, prior_view):
        calls.append(("judge", evidence[:20]))
        if none:
            return None
        return {"winner": winner, "answer": answer, "confidence": 0.8, "rationale": "r"}
    return _f


def ab(bucket, *, abstained=False, winner=None, answer=None, conf=0.3):
    return Answer(question="q", bucket=bucket, abstained=abstained, winner=winner,
                  answer=answer, confidence=conf, status=ToolStatus.OK,
                  candidates=["SLC25A1", "SLC1A6"], rationale="evid-reason")


kg, ck = BackgroundKnowledge(question="t"), ComputedKnowledge(question="t")
CANDS = ["SLC25A1", "SLC1A6"]


def fuse(ans, prior, bucket):
    """_fuse(ans, prior, question, kg, ck, bucket, candidates, task_type, context)"""
    return asyncio.run(ad._fuse(ans, prior, "q", kg, ck, bucket, CANDS, "", None))


# 1) agree is a free win
prior_agent.ask_prior_augmented = mk_aug(winner="SLC25A1")
ans, skip = fuse(ab("pairwise", winner="SLC25A1"),
                 {"winner": "SLC25A1", "confidence": 0.6, "answer": None}, "pairwise")
check("agree: zero extra LLM call, no skip", (not calls) and skip is False and "prior channel: agreement" in ans.rationale)

# 2) abstain -> the augmented LLM fills in the answer
calls.clear()
ans, skip = fuse(ab("pairwise", abstained=True, winner=None),
                 {"winner": "SLC1A6", "confidence": 0.5, "answer": None}, "pairwise")
check("abstain->augmented-LLM chief (+prior, skip conformal, augmented re-answer source)",
      ans.decision_mode.endswith("+prior") and ans.winner == "SLC25A1"
      and skip is True and "evidence-augmented LLM re-answer" in ans.rationale and calls and calls[0][0] == "aug")

# 3) the augmented LLM abstains -> the bare prior covers the gap
calls.clear()
prior_agent.ask_prior_augmented = mk_aug(abstained=True)
ans, skip = fuse(ab("membership", abstained=True),
                 {"answer": True, "winner": None, "confidence": 0.55}, "membership")
check("aug abstains -> raw prior fallback (labelled as a prior fallback)",
      ans.answer is True and "+prior" in ans.decision_mode and "fallback" in ans.rationale)

# 4) conflict -> the judge
calls.clear()
prior_agent.adjudicate = mk_judge(winner="SLC1A6")
ans, skip = fuse(ab("pairwise", winner="SLC25A1"),
                 {"winner": "SLC1A6", "confidence": 0.9, "answer": None}, "pairwise")
check("conflict->LLM judge (+judge, skip conformal, facts passed)",
      ans.decision_mode.endswith("+judge") and ans.winner == "SLC1A6"
      and skip is True and calls and calls[0][0] == "judge")

# 5) judge absent -> the prior wins (the evidence channel is no longer prioritised)
calls.clear()
prior_agent.adjudicate = mk_judge(none=True)
ans, skip = fuse(ab("pairwise", winner="SLC25A1", conf=0.4),
                 {"winner": "SLC1A6", "confidence": 0.9, "answer": None}, "pairwise")
check("judge absent -> prior beats evidence channel",
      ans.winner == "SLC1A6" and "+prior" in ans.decision_mode)

# 6) gate off -> the plain debate behaviour (the evidence channel keeps priority on a conflict, no judge call)
calls.clear()
os.environ["SLC_LLM_CHIEF"] = "0"
prior_agent.adjudicate = mk_judge(winner="SLC1A6")
ans, skip = fuse(ab("pairwise", winner="SLC25A1", conf=0.4),
                 {"winner": "SLC1A6", "confidence": 0.9, "answer": None}, "pairwise")
check("chief off -> plain debate behaviour (evidence wins on conflict, no judge call)",
      ans.winner == "SLC25A1" and "conflict not arbitrated" in ans.rationale and not calls
      and skip is False)

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
