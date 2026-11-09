# -*- coding: utf-8 -*-
"""Conformal abstention (conformal risk control for selective abstention) -- pure mathematics, unit-testable offline.

Setting: a selective classifier emits a prediction with confidence conf in [0,1], then decides to answer or abstain.
Non-conformity score s = 1 - conf (smaller s means more confident). Given a threshold tau: answer only when s <= tau, otherwise abstain.
A larger tau answers more (coverage up) but raises the error risk; the map is monotone, so operating points can be read off the risk-coverage curve.

Guarantee (Bates et al. 2021, Conformal Risk Control; finite sample, distribution-free):
  calibration set (s_i, loss_i) with loss_i in {0,1} (1 = that prediction was wrong). For a candidate tau the plus-one corrected empirical risk is
      L^(tau) = ( sum_i loss_i 1[s_i <= tau]  + 1 ) / ( N + 1 )
  tau_hat = sup{ tau: L_hat(tau) <= alpha }. Under exchangeability the test point satisfies E[ loss_test 1[s_test <= tau_hat] ] <= alpha,
  that is, over the whole population P(answered wrong AND selected) <= alpha, so the error rate among answered questions <= alpha / coverage.

Note: this module only handles samples that produced a prediction; questions the pipeline already abstained on are neither calibrated nor re-opened.
"""
from __future__ import annotations

from dataclasses import dataclass


def nonconformity(confidence: float) -> float:
    """s = 1 - conf, clamped to [0,1]."""
    return max(0.0, min(1.0, 1.0 - float(confidence)))


@dataclass
class Pred:
    """One prediction record: score = non-conformity score, loss = whether it was wrong (0/1)."""
    score: float
    loss: int


def _adjusted_risk(pairs: list[Pred], tau: float) -> float:
    """L^(tau) = (sum loss 1[s<=tau] + 1) / (N + 1)."""
    n = len(pairs)
    if n == 0:
        return 1.0
    wrong_selected = sum(p.loss for p in pairs if p.score <= tau)
    return (wrong_selected + 1) / (n + 1)


def calibrate(pairs: list[Pred], alpha: float) -> float | None:
    """Among the candidate thresholds (every observed score plus +inf) take the largest tau with L_hat(tau) <= alpha; None when nothing qualifies (abstain everywhere).
    alpha must be at least 1/(N+1) for even the lowest-score prediction to be selected."""
    if not pairs or alpha < 0:
        return None
    cands = sorted({p.score for p in pairs})
    best: float | None = None
    for tau in cands:
        if _adjusted_risk(pairs, tau) <= alpha:
            best = tau  # ascending scan, keep the largest tau still satisfying the constraint
    return best


def apply_threshold(pairs: list[Pred], tau: float | None, n_total: int | None = None) -> dict:
    """Apply the threshold to a given set: answer only when s <= tau; returns coverage and risk.
    n_total = all questions (including those the pipeline abstained on); by default the size of this set (only questions that produced a prediction)."""
    n_all = n_total if n_total is not None else len(pairs)
    answered = [p for p in pairs if tau is not None and p.score <= tau]
    na = len(answered)
    wrong = sum(p.loss for p in answered)
    return {
        "tau": tau,
        "n_total": n_all,
        "answered": na,
        "correct": na - wrong,
        "wrong": wrong,
        "coverage": (na / n_all) if n_all else 0.0,
        "risk": (wrong / na) if na else 0.0,  # error rate among answered
        "error_over_all": (wrong / n_all) if n_all else 0.0,  # error rate over all questions
    }


def risk_coverage_curve(pairs: list[Pred], alphas: list[float], n_total: int | None = None) -> list[dict]:
    """For each target risk alpha: calibrate and report on this same set to get curve points (for drawing the risk-coverage curve;
    for official evaluation use a disjoint calibration/test split instead)."""
    out = []
    for a in alphas:
        tau = calibrate(pairs, a)
        row = apply_threshold(pairs, tau, n_total)
        row["alpha"] = a
        out.append(row)
    return out


def ece(conf_correct: list[tuple[float, int]], bins: int = 10) -> float:
    """Expected calibration error: bin-weighted |mean confidence - empirical accuracy|, measuring whether the confidence numbers are deserved."""
    if not conf_correct or bins <= 0:
        return 0.0
    n = len(conf_correct)
    acc = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        box = [(c, y) for (c, y) in conf_correct if lo <= c < hi or (hi == 1.0 and c == 1.0)]
        if not box:
            continue
        conf_mean = sum(c for c, _ in box) / len(box)
        acc_mean = sum(y for _, y in box) / len(box)
        acc += (len(box) / n) * abs(conf_mean - acc_mean)
    return acc
