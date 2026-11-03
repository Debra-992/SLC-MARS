# -*- coding: utf-8 -*-
"""在全 100 v16 题上拟合 margin+tier 的 logistic，折叠成【原始系数常数】供运行时硬编码。

logit = a_margin*margin + a_tier*tier + intercept
（内部先做 margin/mmax 再 z-score，这里把 scaler 一并折进系数）
打印可直接粘进 conf_calib.py 的常量；并报 in-sample(乐观) 与 CV(诚实·0.2378) 两个 AUC-RC。
用法: .venv\\Scripts\\python.exe tools\\_calib_fit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from _v16_diag import build_signals, auc_rc, ece, norm  # noqa: E402
from _calib_holdout import feat, _fit_logistic, _predict, eval_cv  # noqa: E402


def flags(r):
    # 稳健 one-hot：以 llm-chief 为基线(全0)，heuristic/judge 各一罚项；agree 无训练支撑故并入 judge 侧逻辑
    judge = bool(r["judge"])
    heur = (not r["used_llm"]) and not judge
    return [1.0 if heur else 0.0, 1.0 if judge else 0.0]


def main():
    rows, _ = build_signals()
    answered = [r for r in rows if r["outcome"] in ("correct", "wrong")]
    mmax = max((r["margin"] for r in answered), default=1.0) or 1.0

    def feats(r):  # [norm_margin, heur_flag, judge_flag]
        return [norm(r["margin"], 0.0, mmax)] + flags(r)

    X = [feats(r) for r in answered]
    y = [1 if r["outcome"] == "correct" else 0 for r in answered]

    w, b, (mu, sd) = _fit_logistic(X, y)
    # feat0=margin/mmax ; feat1=heur(0/1) ; feat2=judge(0/1)
    a_margin = w[0] / (mmax * sd[0])
    a_heur = w[1] / sd[1]
    a_judge = w[2] / sd[2]
    intercept = b - sum(w[j] * mu[j] / sd[j] for j in range(3))

    # in-sample（乐观上界）
    pr_in = [(_predict((w, b, (mu, sd)), x), "correct" if v else "wrong") for x, v in zip(X, y)]
    print(f"# full-fit  AUC-RC in-sample={auc_rc(pr_in)[0]:.4f} ECE={ece(pr_in):.3f}  (n={len(y)}, wrong={sum(1 for v in y if v==0)})")
    eval_cv(X, y, "margin+heur+judge CV")

    print("\n# === 粘进 conf_calib.py 的烘焙常量 ===")
    print(f"A_MARGIN = {a_margin:.10e}")
    print(f"A_HEUR   = {a_heur:.10e}")
    print(f"A_JUDGE  = {a_judge:.10e}")
    print(f"INTERCEPT= {intercept:.10e}")
    print(f"# heur=非llm且非judge ; judge=('+judge'in mode 或 '法官'in rat) ; 否则 llm-chief 基线")
    print(f"# margin：pairwise= gene_scores top1-top2 ; membership= 单候选 top1")

    # 用折叠常数复算自检
    import math
    def raw(r):
        h, j = flags(r)
        return 1.0/(1.0+math.exp(-max(-30, min(30, a_margin*r["margin"] + a_heur*h + a_judge*j + intercept))))
    chk = [raw(r) for r in answered]
    print(f"# 折叠系数自检 最大偏差={max(abs(p-q) for p,q in zip(chk,[x[0] for x in pr_in])):.2e}")


if __name__ == "__main__":
    main()
