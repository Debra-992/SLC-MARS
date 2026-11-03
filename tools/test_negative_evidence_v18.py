# -*- coding: utf-8 -*-
"""v18 离线测试：② membership 负证据通道 + ③ 选择性 abstention。
全程 mock Mongo（patch 模块级缓存函数），不碰内网不碰端点。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_agents import negative_evidence as ne  # noqa: E402
from knowledge_agents.schema import (Answer, BackgroundKnowledge, ComputedKnowledge,  # noqa: E402
                                     KGEvidence, ToolStatus)
from knowledge_agents.actions.decision_action import AnswerDecision  # noqa: E402

os.environ["SLC_PRIOR_AGENT"] = "1"
os.environ["SLC_LLM_CHIEF"] = "1"
os.environ["SLC_TRIAD_VOTE"] = "0"   # 本测试不涉表决层
os.environ["SLC_TRIAD_CRITIQUE"] = "0"
ad = AnswerDecision()
ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {extra}" if extra else ""))


# ---- ② 门控
os.environ.pop("SLC_NEG_EVID", None)
check("N1 SLC_NEG_EVID default OFF (= v16 byte-identical)", ne.neg_evid_enabled() is False)
os.environ["SLC_NEG_EVID"] = "1"
check("N2 gate on", ne.neg_evid_enabled() is True)

# ---- ② mock 数据：SLC25A1 100篇/共现10，癌种背景5000/库50000 => exp=10 R=1.0
#      旁系 SLC25A6 50篇/共现8 => exp=5 R=1.6（家族内对比信号）
ne._gene_papers = lambda g: (("", "lung cancer") * 0) or tuple(
    (f"t{i}", "lung cancer" if i < (10 if g == "SLC25A1" else 8) else "other")
    for i in range(100 if g == "SLC25A1" else 50))
ne._cancer_count = lambda k: 5000
ne._total_papers = lambda: 50000
rows = ne.cooc_contrast_rows("SLC25A1", "lung cancer", ["SLC25A6", "SLC25A1"])
tgt = next((r for r in rows if r.get("gene") == "SLC25A1"), None)
par = next((r for r in rows if r.get("gene") == "SLC25A6"), None)
check("N3 target row math (exp=10 R=1.0)",
      tgt and tgt["cooccur_papers"] == 10 and tgt["expected_if_independent"] == 10.0
      and tgt["enrichment_R"] == 1.0)
check("N4 paralog contrast present & dedup self (≤2, no dup of target)",
      par and par.get("role") == "paralog_contrast" and par["enrichment_R"] == 1.6
      and sum(1 for r in rows if r.get("gene") == "SLC25A1") == 1)
check("N5 caveat row appended (no over-interpretation of absence)",
      any("caveat" in str(r) for r in rows))

# ---- ② 失败/空数据静默退化
def _boom(g):
    raise RuntimeError("mongo down")
_gp, ne._gene_papers = ne._gene_papers, _boom
check("N6 mongo exception -> [] (v16 degradation, never raises)",
      ne.cooc_contrast_rows("SLC25A1", "lung cancer", []) == [])
ne._gene_papers = lambda g: ()
check("N7 zero gene papers -> [] (no fabricated rows)",
      ne.cooc_contrast_rows("SLCX", "lung cancer", []) == [])
ne._gene_papers = _gp

# ---- ② 注入通路：合成 KGEvidence 经 _facts_digest 可见（法官/主审同一事实源）
os.environ["SLC_NEG_EVID"] = "1"
kg = BackgroundKnowledge(question="t")
kg.evidence.append(KGEvidence(operation="lit_cooc_contrast",
                               rows=ne.cooc_contrast_rows("SLC25A1", "lung cancer", ["SLC25A6"])))
ck = ComputedKnowledge(question="t")
ans = Answer(question="t", bucket="membership", candidates=["SLC25A1"])
digest = AnswerDecision._facts_digest(kg, ck, ans)
check("N8 lit_cooc_contrast flows into _facts_digest (judge/Vega visibility)",
      "lit_cooc_contrast" in digest and "enrichment_R" in digest, f"len={len(digest)}")
check("N9 digest length cap holds (1800)", len(digest) <= 1800)

# ---- ③ 选择性弃权
os.environ.pop("SLC_ABSTAIN_THETA", None)
check("S1 theta default 0 (gate off = v16)", AnswerDecision._abstain_theta() == 0.0)
os.environ["SLC_ABSTAIN_THETA"] = "abc"
check("S2 garbage theta -> 0 (safe parse)", AnswerDecision._abstain_theta() == 0.0)
os.environ["SLC_ABSTAIN_THETA"] = "0.6"
a = Answer(question="t", bucket="membership", candidates=["SLCG"], answer=True,
           confidence=0.55, abstained=False, rationale="r", basis=[])
outp = ad._maybe_selective_abstain(a)
check("S3 conf<θ -> honest abstain (fields cleared + basis tagged)",
      outp.abstained and outp.answer is None and "selective_abstain" in outp.basis
      and "选择性弃答" in outp.rationale)
a2 = Answer(question="t", bucket="membership", candidates=["SLCG"], answer=False,
            confidence=0.75, abstained=False, rationale="r", basis=[])
outp2 = ad._maybe_selective_abstain(a2)
check("S4 conf>=θ -> untouched answer preserved", not outp2.abstained and outp2.answer is False)
a3 = Answer(question="t", bucket="membership", candidates=["SLCG"], abstained=True,
            confidence=0.0, rationale="already", basis=[])
check("S5 already abstained -> idempotent",
      ad._maybe_selective_abstain(a3).abstained)
os.environ["SLC_ABSTAIN_THETA"] = "0"
a4 = Answer(question="t", bucket="membership", candidates=["SLCG"], answer=True,
            confidence=0.1, abstained=False, rationale="r", basis=[])
check("S6 θ=0 low-conf answer untouched (= v16 byte)",
      not ad._maybe_selective_abstain(a4).abstained)

print("==== ALL PASS ====" if ok else "==== SOME FAILED ====")
sys.exit(0 if ok else 1)
