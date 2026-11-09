# -*- coding: utf-8 -*-
"""Paralog buffering -- estimate the functional redundancy of a candidate gene from the SLC
panel already assayed in that cancer type.

Mechanism: SLC genes cluster by subfamily (SLC<number>, e.g. SLC7 / SLC25 / SLC12) and members
of a subfamily frequently overlap in function. If several paralogous subfamily members are
co-expressed in the target cancer type, knocking the gene out is more likely to be buffered by
its paralogs, so a CRISPR loss-of-function screen produces no significant essentiality phenotype -> this
argues for a negative membership call. Conversely, if the gene is nearly an orphan in this cancer
type (no co-expressed paralog), that is merely absence of buffering evidence; it does not license
a claim of essentiality.

This channel is therefore [one-sided evidence]: it may only weaken an essential/positive judgement,
and never manufactures a positive out of "no paralogs".

Data source: DESeq2 top_genes (the SLC panel that is significant and detected in that cancer type).
No panel available -> nothing written (unknown).
"""
from __future__ import annotations

import re

_SLC_FAMILY_RE = re.compile(r"^SLC(\d+)", re.I)
# Pseudogenes / antisense / intronic-transcript and other non-coding symbols cannot functionally
# "buffer" a knockout, so they must be excluded from the paralog count.
#   Typical: SLC25A14P1 / SLC25A5P5 (P+digit pseudogene suffix), SLC22A18AS / SLC7A6OS
#   (AS/OS antisense overlap), SLC7A2-IT1 / SLC25A21-AS1 (hyphenated IT/AS).
_PSEUDO_RE = re.compile(r"(P\d+$)|(-)|(?:AS|OS|SN)$")


def is_functional_symbol(gene: str) -> bool:
    """Rough test for a bona fide SLC gene that could encode a functional protein (pseudogenes,
    antisense and intronic transcripts excluded)."""
    g = (gene or "").strip().upper()
    return bool(g) and not _PSEUDO_RE.search(g)


def subfamily(gene: str) -> str | None:
    """Return the SLC subfamily key (the SLC family number is digits only). SLC7A11->SLC7,
    SLC25A1->SLC25, SLC12A2->SLC12, SLC3A1->SLC3. Non-SLC naming -> None (no paralog
    redundancy estimated for it)."""
    m = _SLC_FAMILY_RE.match((gene or "").strip())
    return ("SLC" + m.group(1)) if m else None


_LNCAP = 5.0  # |log2FC| saturation ceiling: normalise the effect size into [0,1] as the weight for "this paralog is actively expressed in this cancer type"


def panel_from_de(de) -> dict[str, float]:
    """Collapse DifferentialExpression.top_genes into {gene (upper case): expression weight}.
    weight = min(|log2FC|, LNCAP)/LNCAP in [0,1]: not only [is it significant] but also [how large
    the effect is] -- a paralog strongly dysregulated in this cancer type is more likely to actually
    carry the function and therefore to buffer the knockout. Functional symbols only; pseudogenes
    dropped."""
    panel: dict[str, float] = {}
    if not de or not getattr(de, "top_genes", None):
        return panel
    for row in de.top_genes or []:
        g = str(row.get("gene_symbol") or row.get("gene") or "").upper()
        if not g or not is_functional_symbol(g):
            continue
        l2 = row.get("log2fc")
        try:
            w = min(abs(float(l2)), _LNCAP) / _LNCAP if l2 is not None else 0.0
        except (TypeError, ValueError):
            w = 0.0
        panel[g] = max(panel.get(g, 0.0), w)   # deduplicate, keep the largest effect size
    return panel


def redundancy(gene: str, panel: dict[str, float], k_sat: float = 4.0) -> tuple[float, list[str]]:
    """Return (redundancy in [0,1], list of co-expressed paralogs).
    redundancy = min(1, sum of paralog expression weights / k_sat): the paralogs contribute
    weighted by their expression level, so large families and strongly expressed paralogs
    naturally score higher buffering (family-size grading is intrinsic to the sum).
    - subfamily undeterminable (non-SLC naming) -> (0.0, []);
    - empty panel / only the gene itself / paralog weights ~0 -> (0.0, []) (no buffering evidence,
      never an inference of essentiality)."""
    fam = subfamily(gene)
    if not fam:
        return 0.0, []
    self_g = (gene or "").strip().upper()
    partners = sorted(p for p in panel
                      if p != self_g and subfamily(p) == fam and is_functional_symbol(p))
    if not partners or k_sat <= 0:
        return 0.0, partners
    raw = sum(float(panel.get(p, 0.0)) for p in partners)
    return min(1.0, raw / float(k_sat)), partners


def redundancy_map(candidates: list[str], panel: dict[str, float], k_sat: float = 4.0) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Compute redundancy for every candidate gene in one pass. Keys are upper-cased to match the
    decision layer's gene.upper()."""
    red: dict[str, float] = {}
    part: dict[str, list[str]] = {}
    for g in candidates:
        gu = (g or "").strip().upper()
        if not gu:
            continue
        r, ps = redundancy(gu, panel, k_sat)
        red[gu] = r
        part[gu] = ps
    return red, part
