# -*- coding: utf-8 -*-
"""Per-gene graph features of the SLC->Pathway->Disease chain -- the single source for training and online scoring.

The pairwise graph-feature logistic is fitted offline (tools/fit_pairwise_model.py), evaluated offline
(tools/train_membership_logit.py) and used by Vega's online decision (decision_action); all three read
`gene_graph_feat` and `PAIR_COLS` from here, so train/serve skew is impossible: the feature values must match bit for bit.

Rows arrive with two key-naming styles; this function accepts both:
  * the raw gene_disease_paths rows of online kg.evidence: {gene, pathway, rel_AB, rel_BC,
    gene_disease_effect, disease,...}
  * the compact kg_paths rows written by an offline calibration dump: {gene, pathway, rel_AB, rel_BC, effect, disease}
The matched (this cancer type) test keeps the heuristic's definition: substring of cancer(lower) in disease(lower).
"""
from __future__ import annotations

# per-gene graph feature columns (the order is the model coefficient order, written into the artifact and expanded the same way online)
PAIR_COLS = ["n_paths", "n_promote", "n_suppress", "n_cancer_hit",
             "n_ab_promote", "n_ab_suppress", "n_bc_promote", "n_bc_suppress",
             "n_promote_matched", "n_suppress_matched",
             "n_distinct_pathways", "n_distinct_pathways_matched"]


def _effect_of(row: dict) -> str:
    return str(row.get("gene_disease_effect") or row.get("effect") or "").upper()


def gene_graph_feat(gene: str, path_rows: list[dict], cancer: str) -> dict:
    """Extract this gene's fine-grained graph features along the SLC->Pathway->Disease chain from raw gene_disease_paths rows.

    Only evidence fields are read (gene/pathway/rel_AB/rel_BC/effect/disease); ground-truth labels are never touched.
    """
    gene_u = (gene or "").upper()
    cancer_l = (cancer or "").lower()
    f = {k: 0 for k in PAIR_COLS if k.startswith("n_") and "distinct" not in k}
    pw: set[str] = set()
    pwm: set[str] = set()
    for r in path_rows or []:
        if not isinstance(r, dict) or str(r.get("gene", "")).upper() != gene_u:
            continue
        f["n_paths"] += 1
        eff = _effect_of(r)
        ab = str(r.get("rel_AB") or "").upper()
        bc = str(r.get("rel_BC") or "").upper()
        f["n_promote"] += eff == "PROMOTION"
        f["n_suppress"] += eff == "SUPPRESSION"
        f["n_ab_promote"] += ab == "PROMOTION"
        f["n_ab_suppress"] += ab == "SUPPRESSION"
        f["n_bc_promote"] += bc == "PROMOTION"
        f["n_bc_suppress"] += bc == "SUPPRESSION"
        p = str(r.get("pathway") or "").upper()
        matched = bool(cancer_l) and cancer_l in str(r.get("disease", "")).lower()
        if matched:
            f["n_cancer_hit"] += 1
            f["n_promote_matched"] += eff == "PROMOTION"
            f["n_suppress_matched"] += eff == "SUPPRESSION"
            if p:
                pwm.add(p)
        if p:
            pw.add(p)
    f["n_distinct_pathways"] = len(pw)
    f["n_distinct_pathways_matched"] = len(pwm)
    return f


def feat_vec(feat: dict, cols: list[str] = PAIR_COLS) -> list[float]:
    """Feature dict -> numeric vector in the order of cols (a missing column counts as 0)."""
    return [float(feat.get(c, 0) or 0) for c in cols]


def collect_path_rows(kg) -> list[dict]:
    """Collect every gene_disease_paths row from BackgroundKnowledge (including rows filled in from neighbours)."""
    rows: list[dict] = []
    for ev in (getattr(kg, "evidence", []) or []):
        if getattr(ev, "operation", None) != "gene_disease_paths":
            continue
        for r in ev.rows or []:
            if isinstance(r, dict):
                rows.append(r)
    return rows
