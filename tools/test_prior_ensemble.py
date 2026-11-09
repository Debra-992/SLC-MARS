# -*- coding: utf-8 -*-
"""Offline unit test for the ensemble prior (mock LLM, no endpoint needed). It verifies:
  1) _aggregate on pairwise: the mode plus the agreement rate, ties broken by the summed member
     confidence;
  2) _aggregate with everything abstaining / failing => abstained=True (nothing is invented);
  3) ask_prior_ensemble with n<=1 delegates to ask_prior byte-for-byte (the plain one-shot behaviour);
  4) ask_prior_ensemble with n=3 runs three parallel draws over the temperature band and aggregates
     them into a prior carrying agreement;
  5) the _fuse agree branch carries the ensemble-prior annotation;
  6) _prior_view appends the consensus only when agreement is present (otherwise it is the plain view).
"""
import asyncio
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from knowledge_agents import prior_agent as pa
from knowledge_agents.actions.decision_action import AnswerDecision
from knowledge_agents.schema import Answer, BackgroundKnowledge, ComputedKnowledge, ToolStatus

ok = True


def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


# 1) pairwise mode + tie-breaking
agg = pa._aggregate("pairwise", [
    {"winner": "SLC25A1", "confidence": 0.6},
    {"winner": "SLC1A6", "confidence": 0.9},
    {"winner": "SLC25A1", "confidence": 0.7},
], n_draws=3)
check("aggregate plurality winner + agreement 0.67",
      agg["winner"] == "SLC25A1" and abs(agg["agreement"] - 0.667) < 1e-3
      and agg["votes"] == {"SLC25A1": 2, "SLC1A6": 1} and agg["n_valid"] == 3)

# 1b) a 1-1 tie -> the side with the higher summed member confidence wins
agg_t = pa._aggregate("pairwise", [
    {"winner": "A", "confidence": 0.4}, {"winner": "B", "confidence": 0.85},
], n_draws=2)
check("aggregate tie broken by summed confidence", agg_t["winner"] == "B" and agg_t["agreement"] == 0.5)

# 2) everything abstains / fails -> abstained
agg_a = pa._aggregate("membership", [
    {"abstained": True}, None, {"answer": None, "abstained": False},
], n_draws=3)
check("aggregate all-abstain/err -> abstained True, no fabrication",
      agg_a["abstained"] is True and agg_a["n_valid"] == 0
      and agg_a["answer"] is None and agg_a["agreement"] == 0.0)

# 3) n<=1 delegates to ask_prior
seen = {}
async def _fake_ask_prior(q, b, c):
    seen["called"] = True
    return {"winner": "X", "answer": None, "confidence": 0.5, "abstained": False}
_real_ap = pa.ask_prior
pa.ask_prior = _fake_ask_prior
os.environ["SLC_PRIOR_ENSEMBLE"] = "1"
r1 = asyncio.run(pa.ask_prior_ensemble("q", "pairwise", ["X", "Y"]))
check("ensemble n<=1 delegates to ask_prior (one-shot regression)",
      seen.get("called") and r1.get("winner") == "X" and "agreement" not in r1)
pa.ask_prior = _real_ap

# 4) n=3 parallel temperature-spread draws, aggregated
temps_used = []
async def _fake_aug(q, b, c, digest, conflict_hint="", temperature=0.2):
    temps_used.append(temperature)
    win = "SLC25A1" if temperature < 0.9 else "SLC1A6"
    return {"winner": win, "answer": None, "confidence": 0.7, "abstained": False,
            "rationale": "r"}
_real_aug = pa.ask_prior_augmented
pa.ask_prior_augmented = _fake_aug
os.environ["SLC_PRIOR_ENSEMBLE"] = "3"
r3 = asyncio.run(pa.ask_prior_ensemble("q", "pairwise", ["SLC25A1", "SLC1A6"]))
check("ensemble n=3 spreads temps 0.5/0.75/1.0 + plurality + agreement field",
      sorted(temps_used) == [0.5, 0.75, 1.0] and r3["winner"] == "SLC25A1"
      and r3["n_draws"] == 3 and r3["n_valid"] == 3 and r3["agreement"] == round(2/3, 3))
pa.ask_prior_augmented = _real_aug
pa._aggregate.__module__  # keep linter quiet

# 5) the _fuse agree branch carries the ensemble-prior annotation
os.environ["SLC_PRIOR_AGENT"] = "1"
os.environ["SLC_LLM_CHIEF"] = "1"
# Pin the semantics under test: the vote layer must be off (the released environment defaults TRIAD=1)
os.environ["SLC_TRIAD_VOTE"] = "0"
os.environ["SLC_TRIAD_CRITIQUE"] = "0"
ad = AnswerDecision()
kg, ck = BackgroundKnowledge(question="t"), ComputedKnowledge(question="t")
ens_prior = {"winner": "SLC25A1", "answer": None, "confidence": 0.7,
             "agreement": 0.667, "n_valid": 3, "n_draws": 3, "votes": {"SLC25A1": 2, "SLC1A6": 1}}
ans = Answer(question="q", bucket="pairwise", abstained=False, winner="SLC25A1",
             answer=None, confidence=0.4, status=ToolStatus.OK,
             candidates=["SLC25A1", "SLC1A6"], rationale="evid")
out, skip = asyncio.run(ad._fuse(ans, ens_prior, "q", kg, ck, "pairwise",
                                 ["SLC25A1", "SLC1A6"], "", None))
check("agree branch annotates ensemble agreement, no skip",
      skip is False and "ensemble-prior agreement rate" in out.rationale and out.winner == "SLC25A1")

# 6) _prior_view with and without agreement
v_ens = AnswerDecision._prior_view(ens_prior)
v_plain = AnswerDecision._prior_view({"winner": "A", "answer": None, "confidence": 0.5})
check("_prior_view appends consensus only when present",
      "agreement=" in v_ens and "vote pattern=" in v_ens and "agreement=" not in v_plain)

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
