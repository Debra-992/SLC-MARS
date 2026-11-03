# -*- coding: utf-8 -*-
"""v15 三通道表决离线单测（mock LLM，免端点）。验证：
  1) 门控关闭 -> 逐字节 v12（agree 直通 + 冲突走旧 adjudicate 而非 adjudicate_multi）；
  2) agree（证据==先验）-> 直通，零额外调用（不抽票2、不请最终智能体）；
  3) 冲突 2:1 且法官在场 -> adjudicate_multi 裁决，+triad/triad_vote/skip=真；
  4) 冲突 2:1 但法官缺席 -> 确定性多数票退回（采纳先验票，如实标注纯LLM票来源）；
  5) 证据弃答 + 全员无有效票/法官缺席 -> 维持诚实弃答（绝不臆造）；
  6) membership：证据(True)+票2(True) 多数压先验(False) -> 保留证据答案。
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
# 钉住 v15 语义（冠军 v16 定版后 slc_mars_env 默认 CRITIQUE=1/RACE=2，测试需显式声明）
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

# 1a) 门控关 + agree -> v12 直通
reset(SLC_TRIAD_VOTE="0")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), PE)
check("gate off: v12 agree path byte-identical (prior·v11 一致 tag)",
      ans.winner == "SLC25A1" and "+triad" not in ans.decision_mode
      and "prior·v11 一致" in ans.rationale and calls["aug"] == 0)

# 1b) 门控关 + 冲突 -> 走旧 adjudicate（judge None -> prior fallback），不碰 multi
reset(SLC_TRIAD_VOTE="0")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), P1)
check("gate off: conflict uses v12 adjudicate (not multi), prior fallback wins",
      calls["judge12"] == 1 and calls["multi"] == 0 and ans.winner == "SLC1A6")

# 2) 门控开 + agree -> 直通，票2/最终智能体零调用
reset(SLC_TRIAD_VOTE="1")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), PE)
check("triad on: agree -> zero extra calls (no L2 draw, no final agent)",
      ans.winner == "SLC25A1" and calls["aug"] == 0 and calls["multi"] == 0
      and "三通道·v15 一致" in ans.rationale)

# 3) 门控开 + 冲突 2:1 + 法官在场 -> adjudicate_multi 三票呈词 + +triad 标注 + skip
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
      and "triad_vote" in ans.basis and "三通道表决·v15" in ans.rationale
      and len(vnames) == 3 and "W:SLC1A6" in (calls["multi_tally"] or "")
      and calls["aug"] == 1 and calls["judge12"] == 0)

# 4) 法官缺席 -> 确定性多数票退回（L1+L2=SLC1A6 压证据 SLC25A1），如实标注纯LLM票
pa.adjudicate_multi = _multi_none
reset(SLC_TRIAD_VOTE="1")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), P1)
check("judge absent -> deterministic majority adopts pure-LLM vote (honest source tag)",
      ans.winner == "SLC1A6" and skip is True and "纯LLM独立票·v15" in ans.rationale
      and "triad_vote" in ans.basis)

# 5) 证据弃答 + 票1票2全弃权 + 法官缺席 -> 维持诚实弃答（不臆造）
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

# 6) membership：证据 True + 票2 True 多数压先验 False，法官缺席 -> 保留证据答案
async def _aug_true(question, bucket, candidates, digest, conflict_hint="", temperature=0.2, persona=""):
    return {"winner": None, "answer": True, "abstained": False,
            "confidence": 0.6, "rationale": "llm2"}
pa.ask_prior_augmented = _aug_true
reset(SLC_TRIAD_VOTE="1")
ans, skip = decide(ab("membership", answer=True),
                   {"winner": None, "answer": False, "confidence": 0.6})
check("membership majority E+L2(True) beats L1(False) -> evidence answer kept",
      ans.answer is True and "与证据通道一致" in ans.rationale and skip is False)

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
