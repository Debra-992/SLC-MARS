# -*- coding: utf-8 -*-
"""Offline unit test for the cross-critique layer (mock LLM, no endpoint needed). It verifies the
three upgrades on top of the plain three-channel vote:
  1) gate off (TRIAD=0) -> byte-identical to the plain prior-channel fusion (agree passes straight
     through, a conflict goes to the older single-judge adjudicate, multi is never touched);
  2) supply already handed over prior2 -> a conflict [no longer draws L2 serially] (aug=0) and uses
     the supplied vote directly;
  3) CRITIQUE=1 on a conflict -> two parallel cross-critiques first, then all three votes go to the
     final agent (views marked [critiqued], decision_mode +triad+crit, rationale opened with the
     triad-critique lead); a revised vote returned by a critique really propagates into the majority;
  4) CRITIQUE=1 but agree -> still zero extra calls (neither critique nor vote);
  5) a critique returning None -> the original vote is kept (never force-overwritten), the majority
     still decides on the original votes;
  6) CRITIQUE=0 -> a conflict is not critiqued (critique=0), equivalent to the plain three-channel vote;
  7) everyone abstains + judge absent -> the honest abstention stands (nothing is invented).
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
ad = AnswerDecision()
ok = True
kg, ck = BackgroundKnowledge(question="t"), ComputedKnowledge(question="t")
CANDS = ["SLC25A1", "SLC1A6"]

calls = {"aug": 0, "crit": 0, "crit_args": None, "judge12": 0, "multi": 0,
         "multi_views": None, "multi_tally": None}


def _mk_aug(win):
    async def _aug(question, bucket, candidates, digest, conflict_hint="",
                   temperature=0.2, persona=""):
        calls["aug"] += 1
        return {"winner": win, "answer": None, "abstained": False,
                "confidence": 0.6, "rationale": "serial"}
    return _aug


async def _crit_revise(question, bucket, candidates, persona, own_view, others_view):
    calls["crit"] += 1
    calls["crit_args"] = (persona, own_view, others_view)
    # the L1 memory channel is persuaded by L2 and switches to SLC1A6 (aligning with L2 -> a 2:1 majority)
    return {"winner": "SLC1A6", "answer": None, "abstained": False,
            "confidence": 0.8, "rationale": "revised"}


async def _crit_none(question, bucket, candidates, persona, own_view, others_view):
    calls["crit"] += 1
    return None


async def _j12(*a, **k):
    calls["judge12"] += 1
    return None


async def _multi_none(*a, **k):
    calls["multi"] += 1
    return None


async def _multi_on(question, bucket, candidates, evidence, views, tally=""):
    calls["multi"] += 1
    calls["multi_views"] = views
    calls["multi_tally"] = tally
    return {"winner": "SLC1A6", "answer": None, "confidence": 0.7, "rationale": "debated"}


def reset(**kw):
    for k in list(calls):
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


def decide(ans, prior, prior2=None):
    return asyncio.run(ad._decide_fuse(ans, prior, "q", kg, ck, ans.bucket, CANDS, "",
                                       None, prior2))


P1 = {"winner": "SLC25A1", "answer": None, "confidence": 0.6, "rationale": "llm1"}   # same side as the evidence
PE = {"winner": "SLC25A1", "answer": None, "confidence": 0.6, "rationale": "llm1"}
P2 = {"winner": "SLC1A6", "answer": None, "confidence": 0.6, "rationale": "llm2"}
CONFLICT_PRIOR = {"winner": "SLC1A6", "answer": None, "confidence": 0.6, "rationale": "llm1"}

# default wiring
pa.ask_prior_augmented = _mk_aug("SLC1A6")
pa.critique_prior = _crit_revise
pa.adjudicate = _j12
pa.adjudicate_multi = _multi_on

# 1) gate off + agree -> straight through on the plain prior channel; conflict -> the older adjudicate (multi/critique never touched)
reset(SLC_TRIAD_VOTE="0", SLC_TRIAD_CRITIQUE="0")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), PE)
check("gate off agree: plain prior channel byte-identical, no triad",
      ans.winner == "SLC25A1" and "+triad" not in ans.decision_mode and calls["multi"] == 0)
reset(SLC_TRIAD_VOTE="0", SLC_TRIAD_CRITIQUE="0")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), CONFLICT_PRIOR)
check("gate off conflict: single-judge adjudicate only (multi=0 crit=0)",
      calls["judge12"] == 1 and calls["multi"] == 0 and calls["crit"] == 0)

# 2) supply handed over prior2 -> a conflict does not draw L2 serially (aug=0), all three views include the supplied L2
pa.adjudicate_multi = _multi_on
reset(SLC_TRIAD_VOTE="1", SLC_TRIAD_CRITIQUE="0")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), CONFLICT_PRIOR, prior2=P2)
vnames = [n for n, _ in calls["multi_views"]] if calls["multi_views"] else []
check("supplied prior2 -> NO serial L2 draw (aug=0), 3 views to final agent",
      calls["aug"] == 0 and calls["multi"] == 1 and len(vnames) == 3
      and "+triad" in ans.decision_mode and calls["crit"] == 0)

# 3) CRITIQUE=1 on a conflict -> 2 parallel critiques + [critiqued] views + the +crit marking + the critique lead in the rationale
reset(SLC_TRIAD_VOTE="1", SLC_TRIAD_CRITIQUE="1")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), CONFLICT_PRIOR, prior2=P2)
crit_tagged = any("[critiqued]" in n for n in ([x for x, _ in calls["multi_views"]]))
check("critique on conflict: 2 parallel critiques, revised views, tagged +triad+crit",
      calls["crit"] == 2 and crit_tagged and "+triad+crit" in ans.decision_mode
      and "[triad vote + cross-critique]" in ans.rationale and calls["aug"] == 0)

# 4) CRITIQUE=1 but agree -> zero extra calls
reset(SLC_TRIAD_VOTE="1", SLC_TRIAD_CRITIQUE="1")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), PE)
check("critique on but agree -> zero extra (no critique, no vote)",
      calls["crit"] == 0 and calls["multi"] == 0 and "[triad vote: agreement]" in ans.rationale)

# 5) critique returns None -> the original votes are kept and the majority still decides on them
# (here the originals already conflict, so there is no majority -> the plain prior-fallback rule applies)
pa.critique_prior = _crit_none
pa.adjudicate_multi = _multi_none
reset(SLC_TRIAD_VOTE="1", SLC_TRIAD_CRITIQUE="1")
# E=SLC25A1, L1=SLC1A6(prior), L2=SLC1A6(prior2) -> L2 matches L1, so the majority is SLC1A6; judge absent, majority adopted
ans, skip = decide(ab("pairwise", winner="SLC25A1"), CONFLICT_PRIOR, prior2=P2)
check("critique returns None -> original votes kept, majority still decides",
      calls["crit"] == 2 and ans.winner == "SLC1A6" and "triad_vote" in ans.basis)

# 6) CRITIQUE=0 -> a conflict is not critiqued (equivalent to the plain three-channel vote)
pa.critique_prior = _crit_revise
pa.adjudicate_multi = _multi_on
reset(SLC_TRIAD_VOTE="1", SLC_TRIAD_CRITIQUE="0")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), CONFLICT_PRIOR, prior2=P2)
check("critique off -> no critique call (plain three-channel vote equivalent)",
      calls["crit"] == 0 and calls["multi"] == 1 and "+crit" not in ans.decision_mode)

# 7) everyone abstains + judge absent + critiques abstain -> the honest abstention stands
async def _crit_abst(question, bucket, candidates, persona, own_view, others_view):
    return None
async def _aug_abst(question, bucket, candidates, digest, conflict_hint="",
                    temperature=0.2, persona=""):
    return {"winner": None, "answer": None, "abstained": True, "confidence": 0.0}
pa.ask_prior_augmented = _aug_abst
pa.critique_prior = _crit_abst
pa.adjudicate = _j12
pa.adjudicate_multi = _multi_none
reset(SLC_TRIAD_VOTE="1", SLC_TRIAD_CRITIQUE="1")
ans, skip = decide(ab("membership", abstained=True), {"abstained": True, "confidence": 0.0},
                   prior2={"abstained": True, "confidence": 0.0})
check("all abstain + judge absent -> honest abstain preserved",
      ans.abstained is True and ans.answer is None and skip is False)

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
