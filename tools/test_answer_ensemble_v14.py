# -*- coding: utf-8 -*-
"""v14 答案层集成离线单测（mock LLM，免端点）。验证：
  1) N<=1 时 _decide_fuse 逐字节委托单次 _fuse（v12 回归，不抽集成）；
  2) N=3 多次独立融合在【答案层】取众数（2:1 -> 采纳多数）；
  3) 低共识（比例<阈值）-> 诚实弃答、winner/answer 清空、+ansens 标注；
  4) 先验全失败 -> 各次走证据弃答 -> 答案层众数为弃权 -> 诚实弃答；
  5) decision_mode 含 +ansens、basis 含 answer_ensemble。
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
# 钉住 v14 语义：表决层必须关（v16 定版后 slc_mars_env 默认 TRIAD=1）
os.environ["SLC_TRIAD_VOTE"] = "0"
os.environ["SLC_TRIAD_CRITIQUE"] = "0"
ad = AnswerDecision()
ok = True
kg, ck = BackgroundKnowledge(question="t"), ComputedKnowledge(question="t")
CANDS = ["SLC25A1", "SLC1A6"]
SENTINEL = {"winner": "S", "answer": None, "confidence": 0.5}   # 仅作"通道已启用"信号


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


# 1) N<=1 逐字节委托单次 _fuse（agree 分支：evidence winner==prior winner，不调集成）
os.environ["SLC_ANSWER_ENSEMBLE"] = "1"
n_calls = {"aug": 0}
async def _aug_none(*a, **k):
    n_calls["aug"] += 1
    return {"winner": "SLC25A1", "answer": None, "abstained": False, "confidence": 0.6, "rationale": "r"}
pa.ask_prior_augmented = _aug_none
ans, skip = decide(ab("pairwise", winner="SLC25A1"), prior={"winner": "SLC25A1", "confidence": 0.6})
check("N<=1 -> single _fuse v12 agree path (no +ansens, keeps evidence answer)",
      ans.winner == "SLC25A1" and "+ansens" not in ans.decision_mode and "一致" in ans.rationale)

# 2) N=3 答案层众数 2:1：t<0.7 先验=SLC25A1(agree)，t>=0.7 先验=SLC1A6(冲突->法官None->prior SLC1A6)
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
      and "answer_ensemble" in ans.basis and "答案层集成·v14" in ans.rationale)

# 3) 低共识 -> 诚实弃答：阈值拉到 0.8，2:1=0.67<0.8 触发弃权
os.environ["SLC_ENS_MIN_CONSENSUS"] = "0.8"
ans, skip = decide(ab("pairwise", winner="SLC25A1"))
check("low consensus (<0.8) -> honest abstain, winner cleared, skip conformal",
      ans.abstained is True and ans.winner is None and skip is True
      and "低共识弃答" in ans.rationale and "+ansens" in ans.decision_mode)

# 4) 先验全失败(None) -> 每次 _fuse 直通证据弃答 -> 答案层众数=弃权 -> 诚实弃答
async def _aug_all_none(*a, **k):
    return None
pa.ask_prior_augmented = _aug_all_none
os.environ["SLC_ENS_MIN_CONSENSUS"] = "0.5"
ans, skip = decide(ab("membership", abstained=True, answer=None))
check("all priors fail -> all draws abstain -> answer-layer ABSTAIN (no fabrication)",
      ans.abstained is True and ans.answer is None and "__ABSTAIN__" in ans.rationale)

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
