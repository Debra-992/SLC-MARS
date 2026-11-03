# -*- coding: utf-8 -*-
"""v16 离线单测（mock LLM，免端点）。验证 v15 三通道之上的三点升级：
  1) 门控关（TRIAD=0）-> 逐字节 v12（agree 直通 / 冲突走旧 adjudicate，不碰 multi）；
  2) supply 已给 prior2 -> 冲突态【不再串行抽 L2】（aug=0），直接用供票；
  3) CRITIQUE=1 冲突态 -> 先并行交叉批评 2 次，再三票呈最终智能体（视图带 [critiqued]、
     decision_mode +triad+crit16、rationale v16 头）；批评返回修订票真 propagate 到多数；
  4) CRITIQUE=1 但 agree -> 仍零额外调用（不批评、不表决）；
  5) 批评返回 None -> 保留原票（不强行改票），多数仍据原票；
  6) CRITIQUE=0 -> 冲突不批评（critique=0），等价 v15 行为；
  7) 全弃权 + 法官缺席 -> 维持诚实弃答（绝不臆造）。
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
    # L1 记忆通道被 L2 说服，改投 SLC1A6（与 L2 一致 -> 2:1 多数）
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


P1 = {"winner": "SLC25A1", "answer": None, "confidence": 0.6, "rationale": "llm1"}   # 与证据同侧
PE = {"winner": "SLC25A1", "answer": None, "confidence": 0.6, "rationale": "llm1"}
P2 = {"winner": "SLC1A6", "answer": None, "confidence": 0.6, "rationale": "llm2"}
CONFLICT_PRIOR = {"winner": "SLC1A6", "answer": None, "confidence": 0.6, "rationale": "llm1"}

# 默认接线
pa.ask_prior_augmented = _mk_aug("SLC1A6")
pa.critique_prior = _crit_revise
pa.adjudicate = _j12
pa.adjudicate_multi = _multi_on

# 1) 门控关 + agree -> v12 直通；冲突 -> 旧 adjudicate（不碰 multi/critique）
reset(SLC_TRIAD_VOTE="0", SLC_TRIAD_CRITIQUE="0")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), PE)
check("gate off agree: v12 byte-identical, no triad",
      ans.winner == "SLC25A1" and "+triad" not in ans.decision_mode and calls["multi"] == 0)
reset(SLC_TRIAD_VOTE="0", SLC_TRIAD_CRITIQUE="0")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), CONFLICT_PRIOR)
check("gate off conflict: v12 adjudicate only (multi=0 crit=0)",
      calls["judge12"] == 1 and calls["multi"] == 0 and calls["crit"] == 0)

# 2) supply 供票 prior2 -> 冲突不串行抽 L2（aug=0），三票含供来的 L2
pa.adjudicate_multi = _multi_on
reset(SLC_TRIAD_VOTE="1", SLC_TRIAD_CRITIQUE="0")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), CONFLICT_PRIOR, prior2=P2)
vnames = [n for n, _ in calls["multi_views"]] if calls["multi_views"] else []
check("supplied prior2 -> NO serial L2 draw (aug=0), 3 views to final agent",
      calls["aug"] == 0 and calls["multi"] == 1 and len(vnames) == 3
      and "+triad" in ans.decision_mode and calls["crit"] == 0)

# 3) CRITIQUE=1 冲突 -> 并行批评 2 次 + [critiqued] 视图 + +crit16 + v16 rationale 头
reset(SLC_TRIAD_VOTE="1", SLC_TRIAD_CRITIQUE="1")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), CONFLICT_PRIOR, prior2=P2)
crit_tagged = any("[critiqued]" in n for n in ([x for x, _ in calls["multi_views"]]))
check("critique on conflict: 2 parallel critiques, revised views, tagged +crit16/v16",
      calls["crit"] == 2 and crit_tagged and "+crit16" in ans.decision_mode
      and "交叉批评·v16" in ans.rationale and calls["aug"] == 0)

# 4) CRITIQUE=1 但 agree -> 零额外调用
reset(SLC_TRIAD_VOTE="1", SLC_TRIAD_CRITIQUE="1")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), PE)
check("critique on but agree -> zero extra (no critique, no vote)",
      calls["crit"] == 0 and calls["multi"] == 0 and "三通道·v15 一致" in ans.rationale)

# 5) 批评返回 None -> 保留原票，多数仍据原票（此处原票已冲突，无多数 -> 退 v12 prior 胜证据）
pa.critique_prior = _crit_none
pa.adjudicate_multi = _multi_none
reset(SLC_TRIAD_VOTE="1", SLC_TRIAD_CRITIQUE="1")
# E=SLC25A1, L1=SLC1A6(prior), L2=SLC1A6(prior2) -> L2 与 L1 同，多数=SLC1A6；法官缺席多数退回
ans, skip = decide(ab("pairwise", winner="SLC25A1"), CONFLICT_PRIOR, prior2=P2)
check("critique returns None -> original votes kept, majority still decides",
      calls["crit"] == 2 and ans.winner == "SLC1A6" and "triad_vote" in ans.basis)

# 6) CRITIQUE=0 -> 冲突不批评（等价 v15）
pa.critique_prior = _crit_revise
pa.adjudicate_multi = _multi_on
reset(SLC_TRIAD_VOTE="1", SLC_TRIAD_CRITIQUE="0")
ans, skip = decide(ab("pairwise", winner="SLC25A1"), CONFLICT_PRIOR, prior2=P2)
check("critique off -> no critique call (v15 equivalent)",
      calls["crit"] == 0 and calls["multi"] == 1 and "+crit16" not in ans.decision_mode)

# 7) 全弃权 + 法官缺席 + 批评弃权 -> 维持诚实弃答
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
