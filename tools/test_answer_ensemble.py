# -*- coding: utf-8 -*-
"""Offline unit test for the answer-layer ensemble (mock LLM, no endpoint needed). It verifies:
  1) with N<=1 _decide_fuse delegates to a single _fuse byte-for-byte (the plain fusion path, no
     ensemble drawn);
  2) with N=3 several independent fusions are combined by plurality at the [answer layer] (2:1 -> the
     majority is adopted);
  3) low consensus (share < threshold) -> an honest abstention, winner/answer cleared, +ansens marked;
  4) all prior draws fail -> every pass abstains on the evidence -> the answer-layer plurality is an
     abstention -> an honest abstention;
  5) decision_mode contains +ansens and basis contains answer_ensemble.
"""
import asyncio
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from knowledge_agents import prior_agent as pa
from knowledge_agents.actions.decision_action import AnswerDecision
from knowledge_agents.schema import Answer, BackgroundKnowledge, ComputedKnowledge, ToolStatus

os.environ["SLC_PRIOR_AGENT"] = "1"
os.environ["SLC_LLM_CHIEF"] = "1"
# Pin the semantics under test: the vote layer must be off (the released environment defaults TRIAD=1)
os.environ["SLC_TRIAD_VOTE"] = "0"
os.environ["SLC_TRIAD_CRITIQUE"] = "0"
ad = AnswerDecision()
ok = True
kg, ck = BackgroundKnowledge(question="t"), ComputedKnowledge(question="t")
CANDS = ["SLC25A1", "SLC1A6"]
SENTINEL = {"winner": "S", "answer": None, "confidence": 0.5}   # only a signal that the channel is enabled


def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def ab(bucket, *, abstained=False, winner=None, answer=None, conf=0.4):
    return Answer(question="q", bucket=bucket, abstained=abstained, winner=winner,
                  answer=answer, confidence=conf, status=ToolStatus.OK,
                  candidates=CANDS, rationale="evid")


def decide(ans, prior=SENTINEL):
    return asyncio.run(ad._decide_fuse(ans, prior, "q", kg, ck, ans.bucket, CANDS, "", None))


# 1) N<=1 delegates byte-for-byte to a single _fuse (agree branch: evidence winner == prior winner, the ensemble is never called)
os.environ["SLC_ANSWER_ENSEMBLE"] = "1"
n_calls = {"aug": 0}
async def _aug_none(*a, **k):
    n_calls["aug"] += 1
    return {"winner": "SLC25A1", "answer": None, "abstained": False, "confidence": 0.6, "rationale": "r"}
pa.ask_prior_augmented = _aug_none
ans, skip = decide(ab("pairwise", winner="SLC25A1"), prior={"winner": "SLC25A1", "confidence": 0.6})
check("N<=1 -> single plain _fuse agree path (no +ansens, keeps evidence answer)",
      ans.winner == "SLC25A1" and "+ansens" not in ans.decision_mode and "prior channel: agreement" in ans.rationale)

# 2) N=3 answer-layer plurality 2:1: t<0.7 the prior says SLC25A1 (agree), t>=0.7 it says SLC1A6 (conflict -> judge None -> prior SLC1A6)
async def _aug_by_temp(question, bucket, candidates, digest, conflict_hint="", temperature=0.2):
    win = "SLC25A1" if temperature < 0.7 else "SLC1A6"
    return {"winner": win, "answer": None, "abstained": False, "confidence": 0.6, "rationale": "r"}
async def _judge_none(*a, **k):
    return None
pa.ask_prior_augmented = _aug_by_temp
pa.adjudicate = _judge_none
os.environ["SLC_ANSWER_ENSEMBLE"] = "3"
os.environ["SLC_ENS_MIN_CONSENSUS"] = "0.5"
ans, skip = decide(ab("pairwise", winner="SLC25A1"))
check("N=3 answer-layer plurality picks 2:1 majority (SLC25A1) + +ansens tag",
      ans.winner == "SLC25A1" and ans.decision_mode.endswith("+ansens")
      and "answer_ensemble" in ans.basis and "[answer-layer ensemble]" in ans.rationale)

# 3) low consensus -> honest abstention: the threshold is raised to 0.8, so 2:1 = 0.67 < 0.8 triggers it
os.environ["SLC_ENS_MIN_CONSENSUS"] = "0.8"
ans, skip = decide(ab("pairwise", winner="SLC25A1"))
check("low consensus (<0.8) -> honest abstain, winner cleared, skip conformal",
      ans.abstained is True and ans.winner is None and skip is True
      and "low-consensus abstention" in ans.rationale and "+ansens" in ans.decision_mode)

# 4) all prior draws fail (None) -> every _fuse abstains on the evidence -> the answer-layer plurality is an abstention -> honest abstention
async def _aug_all_none(*a, **k):
    return None
pa.ask_prior_augmented = _aug_all_none
os.environ["SLC_ENS_MIN_CONSENSUS"] = "0.5"
ans, skip = decide(ab("membership", abstained=True, answer=None))
check("all priors fail -> all draws abstain -> answer-layer ABSTAIN (no fabrication)",
      ans.abstained is True and ans.answer is None and "__ABSTAIN__" in ans.rationale)

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
