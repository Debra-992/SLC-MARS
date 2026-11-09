# -*- coding: utf-8 -*-
"""Offline unit test for _direction/_verb_dir/_heuristic sign logic (no DB/LLM)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_agents.actions.decision_action import AnswerDecision
from knowledge_agents.schema import (
    BackgroundKnowledge, ComputedKnowledge, KGEvidence, ToolStatus,
)

CK = ComputedKnowledge(question="t")
ad = AnswerDecision()


def kg_with(rows):
    kg = BackgroundKnowledge(question="t", status=ToolStatus.OK)
    kg.evidence.append(KGEvidence(operation="gene_disease_paths", rows=rows))
    return kg


def mk(gene, disease, ab, bc, effect):
    return {"gene": gene, "disease": disease, "rel_AB": ab, "rel_BC": bc,
            "gene_disease_effect": effect, "pathway": "proliferation"}


kg = kg_with([mk("SLC25A32", "Lymphoma", "SUPPRESSION", "PROMOTION", "SUPPRESSION")])
a = ad._heuristic("q", kg, CK, "membership", ["SLC25A32"], "membership_negative", {"cancer": "Lymphoma"})
print("T1 inhibit+SUPP ->", a.answer, "abstain=", a.abstained, "|", a.rationale)

b = ad._heuristic("q", kg, CK, "membership", ["SLC25A32"], "membership_positive", {"cancer": "Lymphoma"})
print("T2 promote+SUPP ->", b.answer, "abstain=", b.abstained)

kgp = kg_with([
    mk("SLC6A1", "Ovarian cancer", "SUPPRESSION", "PROMOTION", "SUPPRESSION"),
    mk("SLC6A1", "Ovarian cancer", "SUPPRESSION", "PROMOTION", "SUPPRESSION"),
    mk("SLC6A1", "Ovarian cancer", "PROMOTION", "SUPPRESSION", "SUPPRESSION"),
    mk("SLC16A1", "Ovarian cancer", "PROMOTION", "PROMOTION", "PROMOTION"),
])
c = ad._heuristic("q", kgp, CK, "pairwise", ["SLC16A1", "SLC6A1"], "pairwise_negative", {"cancer": "Ovarian cancer"})
print("T3 pairwise inhibit ->", c.winner, "abstain=", c.abstained, "|", c.rationale)

kgt = kg_with([
    mk("SLC9A1", "Lymphoma", "PROMOTION", "PROMOTION", "PROMOTION"),
    mk("SLC9A1", "Lymphoma", "SUPPRESSION", "SUPPRESSION", "PROMOTION"),
    mk("SLC9A1", "Lymphoma", "PROMOTION", "SUPPRESSION", "SUPPRESSION"),
    mk("SLC9A1", "Lymphoma", "SUPPRESSION", "PROMOTION", "SUPPRESSION"),
])
d = ad._heuristic("q", kgt, CK, "membership", ["SLC9A1"], "membership_negative", {"cancer": "Lymphoma"})
print("T4 tie ->", d.answer, "abstain=", d.abstained, "|", d.rationale)

# T5 direction and magnitude stay separate: under inhibit (verb=-1) a SUPPRESSION gene gets a
#    positive pol and a PROMOTION gene a negative one, while their strength is equal
kgc = kg_with([
    mk("SLCSUP", "Lymphoma", "SUPPRESSION", "PROMOTION", "SUPPRESSION"),
    mk("SLCPRO", "Lymphoma", "PROMOTION", "PROMOTION", "PROMOTION"),
])
strength, pol, buf, basis = AnswerDecision._collect_scores(kgc, CK, ["SLCPRO", "SLCSUP"], verb=-1)
print("T5 strength->", {k: round(v, 2) for k, v in strength.items()}, "pol->", pol)
print("T5 ok(pol SLCSUP>0>pol SLCPRO & strength equal) ->",
      pol["SLCSUP"] > 0 > pol["SLCPRO"] and abs(strength["SLCSUP"] - strength["SLCPRO"]) < 1e-9)

# T7 a magnitude-only membership question (no gene_disease_paths) must abstain honestly rather than
#    defaulting to positive (the True-bias is gone)
ckg = BackgroundKnowledge(question="t", status=ToolStatus.OK)  # no disease-path evidence at all
t7 = ad._heuristic("q", ckg, CK, "membership", ["SLCXX"], "membership_negative", {"cancer": "Lymphoma"})
print("T7 magnitude-only membership ->", t7.answer, "abstain=", t7.abstained, "|", t7.rationale)


# T6 (C) signed rows recovered from graph neighbours (normalised into gene_disease_paths) are
#    trusted by _direction
kgn = kg_with([mk("SLCX", "colorectal cancer(CRC)", "SUPPRESSION", "PROMOTION", "SUPPRESSION")])
dirn = ad._direction(kgn, "SLCX", "colorectal")
print("T6 _direction on normalized rows ->", dirn)

