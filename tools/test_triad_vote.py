# -*- coding: utf-8 -*-
"""Offline unit test for the three-channel vote (mock LLM, no endpoint needed). It verifies:
  1) gate off -> byte-identical to the plain prior-channel fusion (agree passes straight through, a
     conflict goes to the single-judge adjudicate rather than to adjudicate_multi);
  2) agree (evidence == prior) -> straight through, zero extra calls (no second vote drawn, no final
     agent consulted);
  3) conflict 2:1 with the judge present -> adjudicate_multi rules, +triad / triad_vote / skip all true;
  4) conflict 2:1 but the judge absent -> deterministic majority-vote fallback (the prior vote is
     adopted, with the pure-LLM vote source labelled honestly);
  5) evidence abstains + nobody holds a valid vote / judge absent -> the honest abstention stands
     (nothing is invented);
  6) membership: evidence(True) + vote 2(True) outvotes the prior(False) -> the evidence answer is kept.
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
# Pin the semantics under test: the released environment defaults CRITIQUE=1 and RACE=2, so this test
# has to declare the plain-vote configuration explicitly.
os.environ["SLC_TRIAD_CRITIQUE"] = "0"
os.environ["SLC_PRIOR_RACE"] = "1"
ad = AnswerDecision()
ok = True
kg, ck = BackgroundKnowledge(question="t"), ComputedKnowledge(question="t")
CANDS = ["SLC25A1", "SLC1A6"]

calls = {"aug": 0, "judge12": 0, "multi": 0, "multi_views": None, "multi_tally": None}


async def _aug(question, bucket, candidates, digest, conflict_hint="", temperature=0.2, persona=""):
    calls["aug"] += 1
    return {"winner": "SLC1A6", "answer": None, "abstained": False,
            "confidence": 0.6, "rationale": "llm2"}


async def _j12(*a, **k):
    calls["judge12"] += 1
    return None


async def _multi_none(*a, **k):
    calls["multi"] += 1
    return None


async def _multi_win(question, bucket, candidates, evidence, views, tally=""):
    calls["multi"] += 1
    calls["multi_views"] = views
    calls["multi_tally"] = tally
    return {"winner": "SLC1A6", "answer": None, "confidence": 0.7,
            "rationale": "debated verdict"}


def reset(**kw):
    for k in calls:
        calls[k] = 0 if isinstance(calls[k], int) else None
    os.environ.update({k: v for k, v in kw.items() if k.startswith("SLC_")})


def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def ab(bucket, *, abstained=False, winner=None, answer=None, conf=0.4):
    return Answer(question="q", bucket=bucket, abstained=abstained, winner=winner,
                  answer=answer, confidence=conf, status=ToolStatus.OK,
                  candidates=CANDS, rationale="evid")


def decide(ans, prior):
    return asyncio.run(ad._decide_fuse(ans, prior, "q", kg, ck, ans.bucket, CANDS, "", None))


P1 = {"winner": "SLC1A6", "answer": None, "confidence": 0.6, "rationale": "llm1"}
PE = {"winner": "SLC25A1", "answer": None, "confidence": 0.6, "rationale": "llm1"}

pa.ask_prior_augmented = _aug
pa.adjudicate = _j12
pa.adjudicate_multi = _multi_none

# 1a) gate off + agree -> straight through on the plain prior channel
reset(SLC_TRIAD_VOTE="0")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), PE)
check("gate off: plain prior-channel agree path byte-identical (prior channel agreement tag)",
      ans.winner == "SLC25A1" and "+triad" not in ans.decision_mode
      and "prior channel: agreement" in ans.rationale and calls["aug"] == 0)

# 1b) gate off + conflict -> the single-judge adjudicate (judge None -> prior fallback), multi never touched
reset(SLC_TRIAD_VOTE="0")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), P1)
check("gate off: conflict uses the single-judge adjudicate (not multi), prior fallback wins",
      calls["judge12"] == 1 and calls["multi"] == 0 and ans.winner == "SLC1A6")

# 2) gate on + agree -> straight through, zero calls for vote 2 / the final agent
reset(SLC_TRIAD_VOTE="1")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), PE)
check("triad on: agree -> zero extra calls (no L2 draw, no final agent)",
      ans.winner == "SLC25A1" and calls["aug"] == 0 and calls["multi"] == 0
      and "[triad vote: agreement]" in ans.rationale)

# 3) gate on + conflict 2:1 + judge present -> adjudicate_multi over all three views, +triad marking, conformal skipped
async def _multi_on(question, bucket, candidates, evidence, views, tally=""):
    calls["multi"] += 1
    calls["multi_views"] = views
    calls["multi_tally"] = tally
    return {"winner": "SLC1A6", "answer": None, "confidence": 0.7, "rationale": "debated"}
pa.adjudicate_multi = _multi_on
reset(SLC_TRIAD_VOTE="1")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), P1)
vnames = [n for n, _ in calls["multi_views"]] if calls["multi_views"] else []
check("triad conflict 2:1 -> final agent debates 3 views + tally, verdict tagged +triad",
      ans.winner == "SLC1A6" and skip is True and "+triad" in ans.decision_mode
      and "triad_vote" in ans.basis and "[triad vote]" in ans.rationale
      and len(vnames) == 3 and "W:SLC1A6" in (calls["multi_tally"] or "")
      and calls["aug"] == 1 and calls["judge12"] == 0)

# 4) judge absent -> deterministic majority-vote fallback (L1+L2=SLC1A6 beats the evidence SLC25A1), with the pure-LLM vote labelled honestly
pa.adjudicate_multi = _multi_none
reset(SLC_TRIAD_VOTE="1")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), P1)
check("judge absent -> deterministic majority adopts pure-LLM vote (honest source tag)",
      ans.winner == "SLC1A6" and skip is True and "pure-LLM independent vote" in ans.rationale
      and "triad_vote" in ans.basis)

# 5) evidence abstains + both votes abstain + judge absent -> the honest abstention stands (nothing invented)
async def _aug_abst(question, bucket, candidates, digest, conflict_hint="", temperature=0.2, persona=""):
    return {"winner": None, "answer": None, "abstained": True, "confidence": 0.0}
async def _j_none(*a, **k):
    return None
pa.ask_prior_augmented = _aug_abst
pa.adjudicate = _j_none
pa.adjudicate_multi = _j_none
reset(SLC_TRIAD_VOTE="1")
ans, skip = decide(ab("membership", abstained=True), {"abstained": True, "confidence": 0.0})
check("all abstain + judge absent -> honest abstain preserved (no fabrication)",
      ans.abstained is True and ans.answer is None and skip is False)

# 6) membership: evidence True + vote 2 True outvote the prior False; judge absent -> the evidence answer is kept
async def _aug_true(question, bucket, candidates, digest, conflict_hint="", temperature=0.2, persona=""):
    return {"winner": None, "answer": True, "abstained": False,
            "confidence": 0.6, "rationale": "llm2"}
pa.ask_prior_augmented = _aug_true
reset(SLC_TRIAD_VOTE="1")
ans, skip = decide(ab("membership", answer=True),
                   {"winner": None, "answer": False, "confidence": 0.6})
check("membership majority E+L2(True) beats L1(False) -> evidence answer kept",
      ans.answer is True and "agrees with the evidence channel" in ans.rationale and skip is False)

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
