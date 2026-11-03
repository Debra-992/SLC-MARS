# -*- coding: utf-8 -*-
"""v12 LLM 主审融合层离线单测（mock LLM，免端点）。验证：
  1) agree 分支零调用白赚；
  2) 弃答分支调用证据增强LLM（带事实审料），decision_mode=+prior、溯源含 v12；
  3) 增强LLM 弃答 -> 退回裸 prior；
  4) 冲突分支走 LLM 法官，decision_mode=+judge、跳过保形；
  5) 法官缺席 -> prior 优先（不再证据优先）；
  6) SLC_LLM_CHIEF 关 -> v11 辩论行为原样（回归）。
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
# 钉住 v12 语义：表决层必须关（v16 定版后 slc_mars_env 默认 TRIAD=1，测试需显式关闭）
os.environ["SLC_TRIAD_VOTE"] = "0"
os.environ["SLC_TRIAD_CRITIQUE"] = "0"
# 钉住 v12 语义：辩论层必须关（slc_mars_env 默认 DEBATE=1，场景6 断言的是无辩论的 v11 兜底路径）
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


# 1) agree 白赚
prior_agent.ask_prior_augmented = mk_aug(winner="SLC25A1")
ans, skip = fuse(ab("pairwise", winner="SLC25A1"),
                 {"winner": "SLC25A1", "confidence": 0.6, "answer": None}, "pairwise")
check("agree: zero extra LLM call, no skip", (not calls) and skip is False and "一致" in ans.rationale)

# 2) 弃答 -> 增强LLM 补答
calls.clear()
ans, skip = fuse(ab("pairwise", abstained=True, winner=None),
                 {"winner": "SLC1A6", "confidence": 0.5, "answer": None}, "pairwise")
check("abstain->augmented-LLM chief (+prior, skip conformal, v12 src)",
      ans.decision_mode.endswith("+prior") and ans.winner == "SLC25A1"
      and skip is True and "v12" in ans.rationale and calls and calls[0][0] == "aug")

# 3) 增强LLM 弃答 -> 裸 prior 兜底
calls.clear()
prior_agent.ask_prior_augmented = mk_aug(abstained=True)
ans, skip = fuse(ab("membership", abstained=True),
                 {"answer": True, "winner": None, "confidence": 0.55}, "membership")
check("aug abstains -> raw prior fallback (labeled v11 fallback)",
      ans.answer is True and "+prior" in ans.decision_mode and "fallback" in ans.rationale)

# 4) 冲突 -> 法官
calls.clear()
prior_agent.adjudicate = mk_judge(winner="SLC1A6")
ans, skip = fuse(ab("pairwise", winner="SLC25A1"),
                 {"winner": "SLC1A6", "confidence": 0.9, "answer": None}, "pairwise")
check("conflict->LLM judge (+judge, skip conformal, facts passed)",
      ans.decision_mode.endswith("+judge") and ans.winner == "SLC1A6"
      and skip is True and calls and calls[0][0] == "judge")

# 5) 法官缺席 -> prior 赢（证据不再优先）
calls.clear()
prior_agent.adjudicate = mk_judge(none=True)
ans, skip = fuse(ab("pairwise", winner="SLC25A1", conf=0.4),
                 {"winner": "SLC1A6", "confidence": 0.9, "answer": None}, "pairwise")
check("judge absent -> prior beats evidence channel",
      ans.winner == "SLC1A6" and "+prior" in ans.decision_mode)

# 6) 门关 -> v11 回归（冲突保持证据优先，不调法官）
calls.clear()
os.environ["SLC_LLM_CHIEF"] = "0"
prior_agent.adjudicate = mk_judge(winner="SLC1A6")
ans, skip = fuse(ab("pairwise", winner="SLC25A1", conf=0.4),
                 {"winner": "SLC1A6", "confidence": 0.9, "answer": None}, "pairwise")
check("chief off -> v11 behavior (evidence wins on conflict, no judge call)",
      ans.winner == "SLC25A1" and "冲突未仲裁" in ans.rationale and not calls
      and skip is False)

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
