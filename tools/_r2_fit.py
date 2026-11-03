# -*- coding: utf-8 -*-
"""R2 权重拟合 + 样本外区分度验证（离线·免端点，纯 Python）。

membership 特征（全部决策时可得）：极性矛盾 / 方向证据存在 / 旁系孤儿 / 旁系缓冲(冗余>0) / 量级。
5-fold CV×5 出样本外预测，报：acc(全体membership) 是否 > 60.4% 地板；且必须【真的预测出若干非必需】
（否则=R2 退化成"恒答必需"的先验 → 当场否）。
用法: .venv\\Scripts\\python.exe tools\\_r2_fit.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from _v16_diag import load_jsonl  # noqa: E402
from _calib_holdout import _fit_logistic, _predict, _strat_folds  # noqa: E402
from knowledge_agents.baselines.common import TRACES  # noqa: E402


def feats(basis: set, strength: float):
    conflict = 1.0 if {"kg_direction", "kg_direction_against"} <= basis else 0.0
    dir_present = 1.0 if (basis & {"kg_direction", "kg_direction_against"}) else 0.0
    solo = 1.0 if "paralog_solo" in basis else 0.0
    buffered = 1.0 if "paralog_buffered" in basis else 0.0
    mag = math.tanh(strength)
    return [conflict, dir_present, solo, buffered, mag]


def main():
    rows = []
    lat = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        if str((r.get("provenance") or {}).get("trace_version") or "") == "16":
            lat[r["qid"]] = r
    for r in lat.values():
        if r.get("bucket") != "membership":
            continue
        a = ((r.get("systems") or {}).get("pipeline_auto") or {}).get("answer") or {}
        gs = a.get("gene_scores") or {}
        strength = max(gs.values()) if gs else 0.0
        ess = bool((r.get("ground_truth") or {}).get("answer"))
        rows.append((feats(set(a.get("basis") or []), strength), 1 if ess else 0))
    X = [x for x, _ in rows]
    y = [yy for _, yy in rows]
    n = len(y); base = max(sum(y), n - sum(y)) / n
    print(f"# membership n={n} 必需={sum(y)} 非必需={n-sum(y)} 地板(恒答多数)={base:.1%}")

    # 5-fold CV 样本外
    a_acc, pred_neg, pred_neg_correct, pred_pos = [], 0, 0, 0
    for seed in range(5):
        folds = _strat_folds(y, 5, seed + 1)
        oof = [0] * n
        for f in range(5):
            te = folds[f]; tr = [i for i in range(n) if i not in te]
            m = _fit_logistic([X[i] for i in tr], [y[i] for i in tr])
            for i in te:
                oof[i] = _predict(m, X[i])
        acc = sum(1 for i in range(n) if (oof[i] > 0.5) == (y[i] == 1)) / n
        a_acc.append(acc)
        pred_neg += sum(1 for i in range(n) if oof[i] <= 0.5) / 5
        pred_neg_correct += sum(1 for i in range(n) if oof[i] <= 0.5 and y[i] == 0) / 5
        pred_pos += sum(1 for i in range(n) if oof[i] > 0.5) / 5
    mean_acc = sum(a_acc) / len(a_acc)
    print(f"CV 样本外 acc(全体membership) = {mean_acc:.1%}  (地板 {base:.1%}, {'超' if mean_acc>base else '不超'})")
    print(f"平均预测：必需={pred_pos:.1f}  非必需={pred_neg:.1f}（其中真非必需={pred_neg_correct:.1f}）")
    if pred_neg < 3:
        print("⚠ 几乎不预测非必需 → R2 近似恒答必需=退化先验，应否掉")
    else:
        print("✔ 有非必需预测且部分命中 → 具备区分度，R2 可作为方法")

    # 全量拟合权重（供烘焙参考；最终以臂上 500 留出为准）
    w, b, (mu, sd) = _fit_logistic(X, y)
    names = ["conflict", "dir_present", "solo", "buffered", "mag"]
    print("\n全量拟合系数(标准化空间；正=偏必需)：")
    for nm, wj in zip(names, w):
        print(f"  {nm:12s} {wj:+.3f}")
    print(f"  intercept    {b:+.3f}")
    # 原始空间等价系数（作用于未标准化 raw 特征，供管线逐字复现）
    raw_w = [wj / s for wj, s in zip(w, sd)]
    raw_b = b - sum(wj * m / s for wj, m, s in zip(w, mu, sd))
    print("\n>>> 烘焙 raw 系数（作用于 raw 特征 [conflict,dir_present,solo,buffered,mag]）：")
    for nm, wj in zip(names, raw_w):
        print(f"  {nm:12s} {wj:+.6f}")
    print(f"  {'intercept':12s} {raw_b:+.6f}")


if __name__ == "__main__":
    main()
