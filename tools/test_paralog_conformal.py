# -*- coding: utf-8 -*-
"""Offline unit tests (no DB/LLM/tools) for:
   (1) paralog buffering computation  (2) its integration into the decision heuristic
   (3) conformal risk-control math   (4) the in-pipeline conformal abstention filter.
Run:  .venv/Scripts/python.exe tools/test_paralog_conformal.py   (repo root, self-locating)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Pin the runtime gates down: this file tests the "bare" behaviour of the heuristic decision. If the
# current shell has dot-sourced tools/slc_mars_env.ps1 first, the environment carries the released
# values of SLC_MEM_R2 / SLC_PAIRWISE_MODEL and friends, and both the membership essentiality
# re-ranking and the pairwise graph-feature logistic would rewrite the verdict -- so the same code
# would report a false failure in those shells. The test guarantees its own reproducibility.
for _gate in ("SLC_MEM_R2", "SLC_NEG_EVID", "SLC_ABSTAIN_THETA", "SLC_TRIAD_VOTE",
              "SLC_TRIAD_CRITIQUE", "SLC_PRIOR_RACE", "SLC_LLM_CHIEF", "SLC_DEBATE",
              "SLC_PRIOR_AGENT", "SLC_PRIOR_ENSEMBLE", "SLC_ANSWER_ENSEMBLE",
              "SLC_PRIOR_RETRIEVE"):
    os.environ[_gate] = "0"
for _artifact in ("SLC_PAIRWISE_MODEL", "SLC_CONFORMAL_FILE", "SLC_PARALOG_KSAT",
                  "SLC_DECISION_MODE", "SLC_DE_TOPN", "SLC_DATASET",
                  "SLC_TRACE_VERSION"):
    os.environ.pop(_artifact, None)

from knowledge_agents import conformal as cf
from knowledge_agents.tools import paralog as pl
from knowledge_agents.actions.decision_action import AnswerDecision, W_BUFFER
from knowledge_agents.schema import (
    Answer, BackgroundKnowledge, ComputedKnowledge, DifferentialExpression,
    KGEvidence, ToolStatus,
)

ad = AnswerDecision()
ok = True


def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


# ---------------------------------------------------------- (1) paralog math
check("subfamily SLC7A11->SLC7", pl.subfamily("SLC7A11") == "SLC7")
check("subfamily SLC25A1->SLC25", pl.subfamily("SLC25A1") == "SLC25")
check("subfamily SLC12A2->SLC12", pl.subfamily("SLC12A2") == "SLC12")
check("subfamily TP53->None", pl.subfamily("TP53") is None)

# The panel is now {gene: expression weight in [0,1]}; weight=1.0 is the old "count" form (a strongly
# expressed paralogue)
panel = {"SLC25A1": 1.0, "SLC25A3": 1.0, "SLC25A5": 1.0, "SLC2A1": 1.0, "SLC7A11": 1.0}
r, partners = pl.redundancy("SLC25A32", panel, k_sat=4)
check("redundancy SLC25 family=3/4=0.75", abs(r - 0.75) < 1e-9 and partners == ["SLC25A1", "SLC25A3", "SLC25A5"])
r_solo, ps = pl.redundancy("SLC2A1", panel, k_sat=4)
check("redundancy solo SLC2A1=0", r_solo == 0.0 and ps == [])
r_sat, _ = pl.redundancy("SLC25A32", {g: 1.0 for g in ["SLC25A1", "SLC25A3", "SLC25A5", "SLC25A7", "SLC25A9"]}, k_sat=4)
check("redundancy saturates to 1.0", r_sat == 1.0)
# graded by expression magnitude: a weakly expressed paralogue (weight 0.5) contributes half as much
r_mag, _ = pl.redundancy("SLC7A2", {"SLC7A11": 0.5, "SLC7A5": 1.0}, k_sat=4)
check("redundancy weights by expression magnitude (0.5+1.0)/4=0.375", abs(r_mag - 0.375) < 1e-9)
# a large family whose paralogues are all lowly expressed -> the buffering score stays low, it is no
# longer inflated by the member count alone
r_lowfam, _ = pl.redundancy("SLC25A1", {g: 0.1 for g in ["SLC25A2", "SLC25A3", "SLC25A4", "SLC25A5"]}, k_sat=4)
check("low-expression big family buffered little", r_lowfam < 0.15)
# pseudogenes, antisense and intronic transcripts must never count as buffering paralogues
check("is_functional_symbol SLC25A1 True", pl.is_functional_symbol("SLC25A1"))
check("is_functional_symbol SLC25A14P1 False", not pl.is_functional_symbol("SLC25A14P1"))
check("is_functional_symbol SLC22A18AS False", not pl.is_functional_symbol("SLC22A18AS"))
check("is_functional_symbol SLC7A6OS False", not pl.is_functional_symbol("SLC7A6OS"))
check("is_functional_symbol SLC7A2-IT1 False", not pl.is_functional_symbol("SLC7A2-IT1"))
r_pseudo, ps_pseudo = pl.redundancy("SLC25A1", {"SLC25A1P1": 1.0, "SLC25A2": 1.0, "SLC25A21-AS1": 1.0, "SLC25A24P2": 1.0}, k_sat=4)
check("redundancy ignores pseudogenes (only SLC25A2 counts)", r_pseudo == 0.25 and ps_pseudo == ["SLC25A2"])


def de_with(genes):
    return DifferentialExpression(status=ToolStatus.OK, cancer_type="Breast",
                                  top_genes=[{"gene_symbol": g, "log2fc": 5.0} for g in genes])


# panel_from_de: expression weights derived from the DE table by |log2fc| (strong dysregulation = full
# weight)
_p = pl.panel_from_de(de_with(["SLC25A1", "SLC25A3"]))
check("panel_from_de full weight for |log2fc|>=5", _p.get("SLC25A1") == 1.0 and _p.get("SLC25A3") == 1.0)
_p2 = pl.panel_from_de(DifferentialExpression(
    status=ToolStatus.OK, cancer_type="Breast",
    top_genes=[{"gene_symbol": "SLC7A11", "log2fc": 2.5}]))
check("panel_from_de half weight for |log2fc|=2.5", abs(_p2.get("SLC7A11") - 0.5) < 1e-9)


# ---------------------------------------------------------- (2) decision integration
ck_buf = ComputedKnowledge(question="t")
ck_buf.paralog_redundancy = {"SLC25A1": 0.75}
strength, pol, buf, basis = AnswerDecision._collect_scores(
    BackgroundKnowledge(question="t", status=ToolStatus.OK), ck_buf, ["SLC25A1"], verb=0)
check("buf = -redundancy", abs(buf["SLC25A1"] + 0.75) < 1e-9)
check("basis has paralog_buffered", "paralog_buffered" in basis["SLC25A1"])

# correction: with no causal direction (pol=0), even a gene buffered by its paralogues must never be
# guessed into a negative on its own -> abstain honestly
a_neg = ad._heuristic("q", BackgroundKnowledge(question="t", status=ToolStatus.OK),
                      ck_buf, "membership", ["SLC25A1"], "membership_positive", {"cancer": "Breast"})
check("buffered-only membership (no causal polarity) -> abstain, not fabricated False",
      a_neg.abstained is True and a_neg.answer is None)

# membership with neither a direction nor buffering evidence -> abstain honestly (unknown != negative)
ck_empty = ComputedKnowledge(question="t")
a_abst = ad._heuristic("q", BackgroundKnowledge(question="t", status=ToolStatus.OK),
                       ck_empty, "membership", ["SLCXX"], "membership_positive", {"cancer": "Breast"})
check("no-direction & no-buffer membership -> abstain", a_abst.abstained is True)

# pairwise with two candidates of equal strength: the buffered one ends lower on combined, so the
# winner must be the unbuffered one
kgp = BackgroundKnowledge(question="t", status=ToolStatus.OK)
kgp.evidence.append(KGEvidence(operation="gene_disease_paths", rows=[
    {"gene": "SLC7A11", "disease": "Breast", "gene_disease_effect": "PROMOTION"},
    {"gene": "SLC7A5", "disease": "Breast", "gene_disease_effect": "PROMOTION"},
]))
ck_pair = ComputedKnowledge(question="t")
ck_pair.paralog_redundancy = {"SLC7A11": 1.0, "SLC7A5": 0.0}
a_pair = ad._heuristic("q", kgp, ck_pair, "pairwise", ["SLC7A11", "SLC7A5"],
                       "pairwise_positive", {"cancer": "Breast"})
check("pairwise ranks buffered gene lower (winner=SLC7A5)", a_pair.winner == "SLC7A5")

# ---------------------------------------------------------- (3) conformal math
preds = [cf.Pred(score=0.05, loss=0), cf.Pred(score=0.10, loss=0), cf.Pred(score=0.20, loss=0),
         cf.Pred(score=0.30, loss=0), cf.Pred(score=0.50, loss=1), cf.Pred(score=0.55, loss=0),
         cf.Pred(score=0.80, loss=1), cf.Pred(score=0.90, loss=1), cf.Pred(score=0.95, loss=1)]
# alpha=0 cannot guarantee any selection at all -> None
check("calibrate alpha=0 -> abstain all (None)", cf.calibrate(preds, 0.0) is None)
tau = cf.calibrate(preds, 0.1)
res = cf.apply_threshold(preds, tau, n_total=9)
# finite-sample guarantee: E[loss * 1[s <= tau]] <= alpha  =>  (sum(loss_selected) + 1) / (N + 1) <= alpha
bound = (sum(p.loss for p in preds if p.score <= tau) + 1) / (len(preds) + 1)
check(f"conformal bound within alpha (bound={bound:.3f} <= 0.1)", tau is not None and bound <= 0.1 + 1e-9)
curve = cf.risk_coverage_curve(preds, [0.0, 0.1, 0.2, 0.3, 0.5], n_total=9)
covs = [c["coverage"] for c in curve]
check("risk-coverage: coverage non-decreasing in alpha", all(covs[i] <= covs[i + 1] for i in range(len(covs) - 1)))
check("nonconformity(0.7)=0.3", abs(cf.nonconformity(0.7) - 0.3) < 1e-9)
check("ece in [0,1]", 0.0 <= cf.ece([(1.0 - p.score, p.loss) for p in preds]) <= 1.0)

# ---------------------------------------------------------- (4) conformal filter hook
import knowledge_agents.actions.decision_action as da
da._CONFORMAL_CACHE = {"membership": 0.9}
low = Answer(question="q", bucket="membership", confidence=0.5, answer=True)
low = AnswerDecision._conformal_filter(low, "membership")
check("conf<threshold -> abstained + threshold recorded",
      low.abstained is True and low.answer is None and low.conformal_threshold == 0.9)
high = Answer(question="q", bucket="membership", confidence=0.95, answer=True)
high = AnswerDecision._conformal_filter(high, "membership")
check("conf>=threshold -> keep answer", high.abstained is False and high.answer is True)
da._CONFORMAL_CACHE = {}

print("\n==== " + ("ALL PASS" if ok else "SOME FAILED") + " ====")
raise SystemExit(0 if ok else 1)
