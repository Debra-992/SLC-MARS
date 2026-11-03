# -*- coding: utf-8 -*-
"""保形弃答（Conformal risk control for selective abstention）——纯数学、可离线单测。

设定：选择性分类器先给出预测 + 置信度 conf∈[0,1]，再决定"答"或"弃"。
非一致性分数  s = 1 - conf（s 越小越自信）。给定阈值 τ：**仅当 s ≤ τ 才作答**，否则弃答。
τ 越大 -> 作答越多(覆盖率↑) 但错答风险↑；单调，故可在风险-覆盖曲线上取操作点。

保证（Bates et al. 2021 Conformal Risk Control，有限样本、分布无关）：
  以校准集 (s_i, loss_i)，loss_i∈{0,1}（1=该预测答错）。对候选 τ 定义"加一"修正的经验风险
      L̂(τ) = ( Σ_i loss_i·1[s_i ≤ τ]  + 1 ) / ( N + 1 )
  取  τ̂ = sup{ τ : L̂(τ) ≤ α }。在交换性假设下，测试点满足  E[ loss_test·1[s_test ≤ τ̂] ] ≤ α，
  即『在全体上，错答且被选中的概率 ≤ α』。于是  已答错误率 ≤ α / 覆盖率。

注：本模块只处理"已产生预测"的样本；管线本身已弃答(无预测)的题不参与校准、也不被复启。
"""
from __future__ import annotations

from dataclasses import dataclass


def nonconformity(confidence: float) -> float:
    """s = 1 - conf，夹到 [0,1]。"""
    return max(0.0, min(1.0, 1.0 - float(confidence)))


@dataclass
class Pred:
    """一条预测记录：score=非一致性分数, loss=是否答错(0/1)。"""
    score: float
    loss: int


def _adjusted_risk(pairs: list[Pred], tau: float) -> float:
    """L̂(τ) = (Σ loss·1[s≤τ] + 1) / (N + 1)。"""
    n = len(pairs)
    if n == 0:
        return 1.0
    wrong_selected = sum(p.loss for p in pairs if p.score <= tau)
    return (wrong_selected + 1) / (n + 1)


def calibrate(pairs: list[Pred], alpha: float) -> float | None:
    """在候选阈值（各观测 score 及 +inf）里取满足 L̂(τ)≤α 的最大 τ；无解返回 None(全弃)。
    alpha 需 ≥ 1/(N+1) 才能至少选中一个最低分预测。"""
    if not pairs or alpha < 0:
        return None
    cands = sorted({p.score for p in pairs})
    best: float | None = None
    for tau in cands:
        if _adjusted_risk(pairs, tau) <= alpha:
            best = tau              # 升序扫描，保留仍满足约束的最大 τ
    return best


def apply_threshold(pairs: list[Pred], tau: float | None, n_total: int | None = None) -> dict:
    """在给定集上施加阈值：s≤τ 才作答。返回覆盖/风险。
    n_total=全体题数（含管线本身已弃答者）；缺省=当前集大小（仅统计已产生预测的题）。"""
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
        "risk": (wrong / na) if na else 0.0,          # 已答错误率
        "error_over_all": (wrong / n_all) if n_all else 0.0,  # 全体口径错答率
    }


def risk_coverage_curve(pairs: list[Pred], alphas: list[float], n_total: int | None = None) -> list[dict]:
    """对每个目标风险 α：用本集自身做"校准+报告"的曲线点（供画 risk-coverage 曲线用；
    正式评测请用 disjoint 校准/测试切分，见 conformal_report）。"""
    out = []
    for a in alphas:
        tau = calibrate(pairs, a)
        row = apply_threshold(pairs, tau, n_total)
        row["alpha"] = a
        out.append(row)
    return out


def ece(conf_correct: list[tuple[float, int]], bins: int = 10) -> float:
    """期望校准误差 ECE：|平均置信 - 经验准确率| 的分箱加权，衡量 confidence 是否名副其实。"""
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
