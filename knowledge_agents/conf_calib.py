# -*- coding: utf-8 -*-
"""1) Confidence calibration layer (env-gated, deterministic post-processing, changes only
confidence and never winner/answer/abstained).

Motivation: the confidence the pipeline actually emitted (in particular the heuristic path, where
conf is proportional to |net|) ranked inverted -- AUC-RC 0.374, the worst of all systems. Recalibrate
it from the two orthogonal signals already present at decision time: the evidence margin
gene_scores (top1-top2, or top1 for a single candidate) plus a one-hot of the decision path
(heuristic penalty / judge). A logistic regression was fitted on 100 runs and its weights collapsed
into the constants baked in here.

Out-of-sample (5-fold CV, 5 repeats) AUC-RC = 0.249 +- 0.007, matching mdagents 0.249 and well ahead
of autogen 0.333 and the uncalibrated 0.374.
Off by default: unless SLC_CONF_CALIB is "1" this module does not participate and the released
SLC-MARS behaviour is unchanged byte for byte.

The constants were fitted offline; the fitting driver is experiment scaffolding and is not part of
this release. Do not hand-edit them unless you refit.
"""
from __future__ import annotations

import math
import os

# ---- baked constants (fitted by logistic regression on 100 runs; the margin is clipped at MMAX to
# ---- reproduce the normalisation used at fit time) ----
A_MARGIN = 8.5993179835e-02     # evidence-margin coefficient (positive: a wider margin is more trustworthy)
A_HEUR = -1.2866237239e+00      # heuristic-path penalty (that path was only ~47% accurate)
A_JUDGE = 1.2721455614e-01      # judge path (close to the llm-chief baseline)
INTERCEPT = 8.4911955020e-01
MMAX = 10.25                     # upper bound of gene_margin in the training set (values above are clipped)


def calib_enabled() -> bool:
    return os.environ.get("SLC_CONF_CALIB", "0") == "1"


def _margin(gene_scores: dict | None) -> float:
    vals = sorted((float(v) for v in (gene_scores or {}).values()), reverse=True)
    if len(vals) >= 2:
        return vals[0] - vals[1]
    return vals[0] if vals else 0.0


def _flags(decision_mode: str, rationale: str, used_llm: bool) -> tuple[float, float]:
    judge = ("+judge" in decision_mode) or ("judge" in rationale)
    heur = (not used_llm) and not judge
    return (1.0 if heur else 0.0, 1.0 if judge else 0.0)


def calibrated_confidence(ans) -> float:
    """Recompute confidence from the fields ans already carries; does not mutate ans."""
    m = min(_margin(getattr(ans, "gene_scores", None)), MMAX)
    heur, judge = _flags(str(getattr(ans, "decision_mode", "") or ""),
                         str(getattr(ans, "rationale", "") or ""),
                         bool(getattr(ans, "used_llm", False)))
    logit = A_MARGIN * m + A_HEUR * heur + A_JUDGE * judge + INTERCEPT
    return round(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit)))), 3)


def apply_calibration(ans):
    """Rewrite ans.confidence in place (answered questions only; an abstain keeps its value).
    Returns ans so calls can be chained."""
    if not calib_enabled() or getattr(ans, "abstained", False):
        return ans
    ans.confidence = calibrated_confidence(ans)
    return ans
