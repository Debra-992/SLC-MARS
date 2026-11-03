# -*- coding: utf-8 -*-
"""① 置信度校准层：样本外(5-fold CV)验证，防止在 101 行上过拟合。

对 v16-auto 已存的每题，用正交信号(gene_margin / decision-tier / raw conf / used_llm / bucket)
拟合 logistic 预测 P(答对)，用【留出一折】概率算 AUC-RC / ECE，重复 5 次取均值±std，
与 mdagents(0.249) / autogen / raw_conf 同口径比。只有样本外 AUC-RC 稳定 < mdagents 才算 ① 立得住。

环境注：本机 sklearn 的 _cython_blas DLL 被 Windows 应用控制策略拦截，故 logistic 纯 Python
手写（L2 梯度下降 + z-score 标准化），确定性、可复算。
用法: .venv\\Scripts\\python.exe tools\\_calib_holdout.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from _v16_diag import build_signals, auc_rc, ece, norm  # noqa: E402
from knowledge_agents.baselines.common import TRACES, GT_PATH, INPUT_PATH  # noqa: E402
from knowledge_agents.baselines.merge_baselines import load_jsonl, outcome_of  # noqa: E402


def feat(r, mmax):
    tier = 0.85 if r["agree"] else (0.62 if r["judge"] else (0.70 if not r["used_llm"] else 0.50))
    return [
        norm(r["margin"], 0.0, mmax),      # 证据 top1-top2 边际
        tier,                               # 决策路径可靠性分层
        1.0 if r["agree"] else 0.0,
        1.0 if r["judge"] else 0.0,
        1.0 if r["used_llm"] else 0.0,
        1.0 if r["bucket"] == "membership" else 0.0,
        r["raw"],                           # 原自报 conf（当一个特征，不迷信）
    ]


def _strat_folds(y, k, seed):
    idx0 = [i for i, v in enumerate(y) if v == 0]
    idx1 = [i for i, v in enumerate(y) if v == 1]
    for g in (idx0, idx1):
        random.Random(seed * 1000 + len(g)).shuffle(g)
    folds = [[] for _ in range(k)]
    for g in (idx0, idx1):
        for j, i in enumerate(g):
            folds[j % k].append(i)
    return folds


def _fit_logistic(X, y, iters=400, lr=0.5, l2=1.0):
    """纯 Python L2 logistic（batch 梯度下降）；先按训练集 z-score 标准化，返回 (w,b,scaler)。"""
    d = len(X[0])
    mu = [sum(r[j] for r in X) / len(X) for j in range(d)]
    sd = [(sum((r[j]-mu[j])**2 for r in X) / len(X))**0.5 or 1.0 for j in range(d)]
    Z = [[(r[j]-mu[j])/sd[j] for j in range(d)] for r in X]
    w = [0.0]*d
    b = 0.0
    n = len(Z)
    for _ in range(iters):
        gw = [0.0]*d
        gb = 0.0
        for zi, yi in zip(Z, y):
            z = sum(wj*xj for wj, xj in zip(w, zi)) + b
            p = 1.0/(1.0+math.exp(-max(-30, min(30, z))))
            e = p - yi
            gb += e
            for j in range(d):
                gw[j] += e*zi[j]
        for j in range(d):
            w[j] -= lr*(gw[j]/n + l2*w[j]/n)
        b -= lr*(gb/n)
    return w, b, (mu, sd)


def _predict(model, x):
    w, b, (mu, sd) = model
    z = sum(wj*((xj-m)/s) for wj, xj, m, s in zip(w, x, mu, sd)) + b
    return 1.0/(1.0+math.exp(-max(-30, min(30, z))))


def cv_oof(X, y, k=5, reps=5):
    """返回每次重复的样本外概率向量列表（cross_val_predict 等价）。"""
    n = len(y)
    preds_per_rep = []
    for seed in range(reps):
        folds = _strat_folds(y, k, seed+1)
        oof = [0.0]*n
        for f in range(k):
            te = folds[f]
            tr = [i for i in range(n) if i not in te]
            if not tr or not te:
                continue
            m = _fit_logistic([X[i] for i in tr], [y[i] for i in tr])
            for i in te:
                oof[i] = _predict(m, X[i])
        preds_per_rep.append(oof)
    return preds_per_rep


def eval_cv(X, y, tag):
    import statistics
    aucs, eces = [], []
    for oof in cv_oof(X, y):
        pr = list(zip(oof, ["correct" if v else "wrong" for v in y]))
        aucs.append(auc_rc(pr)[0])
        eces.append(ece(pr))
    m, s = statistics.fmean(aucs), (statistics.pstdev(aucs))
    me = statistics.fmean(eces)
    print(f"  {tag:22s} AUC-RC={m:.4f}±{s:.4f}  ECE={me:.3f}")
    return m, s


def baseline_pairs(sysname):
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    bucket_of = {i["qid"]: i["bucket"] for i in inputs}
    p = TRACES / f"answers_{sysname}.jsonl"
    seen = {}
    if p.exists():
        for rr in load_jsonl(p):
            seen[rr["qid"]] = rr
    out = []
    for q, rr in seen.items():
        o = outcome_of(rr, gt.get(q, {}), bucket_of.get(q, ""))
        if o in ("correct", "wrong"):
            out.append((float(rr.get("confidence") or 0.0), o))
    return out


def main():
    rows, _ = build_signals()
    answered = [r for r in rows if r["outcome"] in ("correct", "wrong")]
    n = len(answered)
    mmax = max((r["margin"] for r in answered), default=1.0) or 1.0
    X = [feat(r, mmax) for r in answered]
    y = [1 if r["outcome"] == "correct" else 0 for r in answered]

    print(f"# v16-auto answered n={n} (wrong={sum(1 for v in y if v==0)}); 5-fold CV 留出·重复5次·纯Python logistic")
    eval_cv(X, y, "composite(all sigs)")
    eval_cv([r[:2] for r in X], y, "margin+tier")
    eval_cv([[r[0]] for r in X], y, "gene_margin only")
    eval_cv([[r[6]] for r in X], y, "raw_conf only(对照)")

    for name in ("mdagents", "autogen"):
        pr = baseline_pairs(name)
        a, _ = auc_rc(pr)
        print(f"  {name:22s} AUC-RC={a:.4f}(given) ECE={ece(pr):.3f}  n={len(pr)}")
    pr_all = [(r["raw"], r["outcome"]) for r in answered]
    print(f"  {'raw_conf in-sample':22s} AUC-RC={auc_rc(pr_all)[0]:.4f} (过拟合上界参考)")


if __name__ == "__main__":
    main()
