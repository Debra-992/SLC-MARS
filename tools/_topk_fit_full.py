# -*- coding: utf-8 -*-
"""训练并序列化 lever 1 的【全特征 pointwise 必需性排序器】—— v20full 的内核。

与既有 pairwise 图 logistic 的区别：本模型把 v20 二分类一直在用、但 topk 图头【从未接入】的
组学通道（PPR 邻近 / TIMER / KM / DESeq2 log2FC / 旁系冗余）一并喂进来，学 P(必需)。
训练数据 = calibstore 里 membership 记录（单基因 + 完整 `features` + essential 0/1 标签），
这些 `features` 正是 knowledge_agents.calib_dump._features 落盘的键，与服务端
feature_vector.features_for_gene 同键同序 -> 直接复用 feature_vector.vec 向量化，零 train/serve skew。

严格防泄漏：按 qid 哈希【不相交】TRAIN/TEST 切分（seed5, train_frac0.6），TEST 只报告泛化。
工件带 trained_qids，_topk_full 据此核对 topk 题与训练题无交集（topk 桶本就不在 calibstore 内，
故为纯任务族迁移，非记忆）。

用法：
  .venv\\Scripts\\python.exe tools\\_topk_fit_full.py \
      [--traces outputs/traces/calibstore.jsonl] [--out outputs/traces/fullfeature_model.json] \
      [--seed 5] [--train-frac 0.6]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from knowledge_agents.feature_vector import FULL_COLS, vec  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _split(qid: str, seed: int, train_frac: float) -> str:
    h = int(hashlib.sha1(f"{seed}|{qid}".encode()).hexdigest(), 16) % 1000
    return "train" if h < int(train_frac * 1000) else "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=os.path.join("outputs", "traces", "calibstore.jsonl"))
    ap.add_argument("--out", default=os.path.join("outputs", "traces", "fullfeature_model.json"))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.6)
    args = ap.parse_args()

    if not os.path.exists(args.traces):
        print(f"找不到校准池: {args.traces}（先跑 calib_dump）")
        return
    rows = [json.loads(l) for l in open(args.traces, encoding="utf-8") if l.strip()]

    data = []  # (qid, gene, label, vec, side)
    for r in rows:
        if r.get("error") or r.get("bucket") != "membership" or r.get("label") is None:
            continue
        f = r.get("features")
        if not f:
            continue
        side = _split(r["qid"], args.seed, args.train_frac)
        data.append((r["qid"], str(f.get("gene", "")).upper(),
                     int(r["label"]), vec(f, FULL_COLS), side))

    Xtr = np.array([d[3] for d in data if d[4] == "train"], float)
    ytr = np.array([d[2] for d in data if d[4] == "train"], int)
    Xte = np.array([d[3] for d in data if d[4] == "test"], float)
    yte = np.array([d[2] for d in data if d[4] == "test"], int)
    trained_qids = sorted({d[0] for d in data if d[4] == "train"})
    print(f"全特征 pointwise 训练 | 样本 {len(data)}（membership）"
          f"| 训练 {len(ytr)} / 测试 {len(yte)} | 列 {len(FULL_COLS)} | qid 不相交 seed={args.seed}")
    if len(ytr) < 40 or len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        print("样本太少或类别不足，不拟合。")
        return

    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": [0.01, 0.03, 0.1, 0.3, 1.0]},
                      scoring="neg_log_loss", cv=StratifiedKFold(3, shuffle=True, random_state=args.seed))
    gs.fit(Xtr, ytr)
    C = gs.best_params_["C"]

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    model = LogisticRegression(C=C, max_iter=2000).fit((Xtr - mu) / sd, ytr)
    coef, intercept = model.coef_[0].tolist(), float(model.intercept_[0])

    # —— TEST 留出：单基因 0.5 判 + 全体 AUC（泛化，不含训练标签）——
    def _prob(vec_list):
        z = [(vec_list[i] - mu[i]) / sd[i] for i in range(len(FULL_COLS))]
        lg = intercept + sum(coef[i] * z[i] for i in range(len(FULL_COLS)))
        return 1.0 / (1.0 + math.exp(-lg))

    p_te = np.array([_prob(list(v)) for v in Xte], float)
    try:
        holdout_auc = float(roc_auc_score(yte, p_te))
    except ValueError:
        holdout_auc = float("nan")
    holdout_acc = float(((p_te >= 0.5).astype(int) == yte).mean())
    dummy = max(float(yte.mean()), 1 - float(yte.mean()))

    artifact = {
        "task": "fullfeature_membership_pointwise",
        "features": list(FULL_COLS),
        "mu": mu.tolist(), "sd": sd.tolist(), "coef": coef, "intercept": intercept,
        "C": C, "seed": args.seed, "train_frac": args.train_frac,
        "train_gene_rows": int(len(ytr)), "trained_qids": trained_qids,
        "holdout": {"n": int(len(yte)), "auc": holdout_auc, "acc": holdout_acc, "dummy": dummy},
        "no_signal_behavior": "fallback_graph_only",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    print(f"留出(泛化) AUC={holdout_auc:.3f} acc={holdout_acc:.1%} | 多数类 dummy={dummy:.1%} | C={C}")
    from knowledge_agents.graph_features import PAIR_COLS
    graph_set = set(PAIR_COLS)
    print("学到的标准化权重（正=更可能必需/winner）：")
    for name, w in sorted(zip(FULL_COLS, coef), key=lambda t: -abs(t[1])):
        if abs(w) < 1e-3:
            continue
        tag = "图链" if name in graph_set else "组学"
        print(f"    {w:+.4f}  [{tag}] {name}")
    print(f"\n工件已写: {args.out}")


if __name__ == "__main__":
    main()
