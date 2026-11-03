# -*- coding: utf-8 -*-
"""① 置信度校准层（env 门控 · 确定性后处理 · 只改 confidence，绝不动 winner/answer/abstained）。

动机：v16 实际吐出的 confidence（尤其启发式路径 conf∝|net|）排序倒挂——AUC-RC 0.374，
全场最差。用决策时【已有】的两个正交信号重标定：证据边际 gene_scores(top1-top2/单候选 top1)
+ 决策路径 one-hot(heuristic 罚项 / judge)。在全 100 v16 题上拟合 logistic，权重折叠成常数烘焙于此。

样本外(5-fold CV·重复5次) AUC-RC=0.249±0.007（追平 mdagents 0.249、碾压 autogen 0.333 与 v16 现用 0.374）。
默认关闭：SLC_CONF_CALIB 未置 "1" 时本模块不参与，定版 v20 逐字节不变。

拟合口径与自检见 tools/_calib_fit.py；勿手改常数除非重拟合。
"""
from __future__ import annotations

import math
import os

# ---- 烘焙常量（由 tools/_calib_fit.py 在 v16@n=100 上拟合，margin 按 MMAX 截断以复现 norm clip）----
A_MARGIN = 8.5993179835e-02     # 证据边际系数（正：边际越大越可信）
A_HEUR = -1.2866237239e+00      # 启发式路径罚项（该路径真实正确率仅 ~47%）
A_JUDGE = 1.2721455614e-01      # judge 路径（近 llm-chief 基线）
INTERCEPT = 8.4911955020e-01
MMAX = 10.25                     # 训练集 gene_margin 上界（超出按此截断）


def calib_enabled() -> bool:
    return os.environ.get("SLC_CONF_CALIB", "0") == "1"


def _margin(gene_scores: dict | None) -> float:
    vals = sorted((float(v) for v in (gene_scores or {}).values()), reverse=True)
    if len(vals) >= 2:
        return vals[0] - vals[1]
    return vals[0] if vals else 0.0


def _flags(decision_mode: str, rationale: str, used_llm: bool) -> tuple[float, float]:
    judge = ("+judge" in decision_mode) or ("法官" in rationale)
    heur = (not used_llm) and not judge
    return (1.0 if heur else 0.0, 1.0 if judge else 0.0)


def calibrated_confidence(ans) -> float:
    """由 ans 已有字段重算 confidence；不修改 ans。"""
    m = min(_margin(getattr(ans, "gene_scores", None)), MMAX)
    heur, judge = _flags(str(getattr(ans, "decision_mode", "") or ""),
                         str(getattr(ans, "rationale", "") or ""),
                         bool(getattr(ans, "used_llm", False)))
    logit = A_MARGIN * m + A_HEUR * heur + A_JUDGE * judge + INTERCEPT
    return round(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit)))), 3)


def apply_calibration(ans):
    """就地重写 ans.confidence（仅 answered 题；abstain 保持原值）。返回 ans 便于链式。"""
    if not calib_enabled() or getattr(ans, "abstained", False):
        return ans
    ans.confidence = calibrated_confidence(ans)
    return ans
