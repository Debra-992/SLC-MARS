# -*- coding: utf-8 -*-
"""拟合并【序列化】pairwise 图特征 logistic —— 供 Vega 在线判定加载打分（推理不需 sklearn）。

严格防泄漏：只在 qid 哈希【不相交】TRAIN 子集(默认 seed5, train_frac0.6)上训练；TEST 子集仅用于
如实报告泛化准确率，其标签绝不进模型。工件里带 trained_qids 清单，公平对比据此把 pairwise 表现
拆成"训练内(乐观)"与"留出(真实泛化)"两栏，避免用训练标签充数。

特征与在线打分共用 knowledge_agents.graph_features.gene_graph_feat（同一函数、同一列序），
杜绝 train/serve skew。

用法：
  python tools/fit_pairwise_model.py [--traces outputs/traces/calibstore.jsonl]
        [--out outputs/traces/pairwise_graph_model.json] [--seed 5] [--train-frac 0.6]
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledge_agents.graph_features import PAIR_COLS, gene_graph_feat, feat_vec  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _split(qid: str, seed: int, train_frac: float) -> str:
    h = int(hashlib.sha1(f"{seed}|{qid}".encode()).hexdigest(), 16) % 1000
    return "train" if h < int(train_frac * 1000) else "test"


def _load_pairs(rows):
    """pointwise 归约：同题 winner 记正例、loser 记负例。"""
    data = []
    for r in rows:
        if r.get("error") or r.get("bucket") != "pairwise":
            continue
        w = str((r.get("ground_truth") or {}).get("winner", "")).upper()
        genes = [str(g).upper() for g in (r.get("effective_genes") or [])][:2]
        if len(genes) != 2 or not w or w not in genes:
            continue
        loser = genes[1] if genes[0] == w else genes[0]
        kgp, cancer = r.get("kg_paths") or [], r.get("cancer", "")
        side = _split(r["qid"], ARGS.seed, ARGS.train_frac)
        for g, lab in ((w, 1), (loser, 0)):
            v = feat_vec(gene_graph_feat(g, kgp, cancer), PAIR_COLS)
            data.append((r["qid"], g, lab, v, side))
    return data


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=os.path.join("outputs", "traces", "calibstore.jsonl"))
    ap.add_argument("--out", default=os.path.join("outputs", "traces", "pairwise_graph_model.json"))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ARGS = ap.parse_args()

    if not os.path.exists(ARGS.traces):
        print(f"找不到校准池: {ARGS.traces}（先跑 calib_dump）")
        return
    rows = [json.loads(l) for l in open(ARGS.traces, encoding="utf-8") if l.strip()]
    winner_by_q = {r["qid"]: str((r.get("ground_truth") or {}).get("winner", "")).upper()
                   for r in rows if r.get("bucket") == "pairwise" and not r.get("error")}
    data = _load_pairs(rows)
    trained_qids = sorted({q for q, _, _, _, s in data if s == "train"})
    Xtr = np.array([d[3] for d in data if d[4] == "train"], float)
    ytr = np.array([d[2] for d in data if d[4] == "train"], int)
    if len(ytr) < 40 or len(np.unique(ytr)) < 2:
        print("训练样本太少，不拟合。")
        return

    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": [0.01, 0.03, 0.1, 0.3, 1.0]},
                      scoring="neg_log_loss", cv=StratifiedKFold(3, shuffle=True, random_state=ARGS.seed))
    gs.fit(Xtr, ytr)
    C = gs.best_params_["C"]

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    model = LogisticRegression(C=C, max_iter=2000).fit((Xtr - mu) / sd, ytr)
    coef, intercept = model.coef_[0].tolist(), float(model.intercept_[0])

    # —— TEST 留出：按题 argmax 判 winner（泛化，不含任何训练标签）——
    byq: dict[str, list] = {}
    for q, g, lab, v, side in data:
        if side == "test":
            byq.setdefault(q, []).append((g, np.array(v, float)))

    def _prob(vec):
        z = (vec - mu) / sd
        lg = float(np.dot(coef, z) + intercept)
        return 1.0 / (1.0 + math.exp(-lg))

    nq = nok = 0
    for q, items in byq.items():
        if len(items) < 2:
            continue
        probs = [_prob(v) for _, v in items]
        nq += 1
        nok += items[int(np.argmax(probs))][0] == winner_by_q.get(q, "")
    holdout_acc = nok / nq if nq else float("nan")

    artifact = {
        "task": "pairwise_graph_logistic",
        "features": list(PAIR_COLS),
        "mu": mu.tolist(), "sd": sd.tolist(), "coef": coef, "intercept": intercept,
        "C": C, "seed": ARGS.seed, "train_frac": ARGS.train_frac,
        "train_gene_rows": int(len(ytr)), "trained_qids": trained_qids,
        "holdout": {"n_questions": nq, "acc": holdout_acc},
        "no_signal_behavior": "fallback_rules",
    }
    os.makedirs(os.path.dirname(ARGS.out), exist_ok=True)
    with open(ARGS.out, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    print(f"pairwise 图特征 logistic：训练基因行 {len(ytr)}（TRAIN 题 {len(trained_qids)}）| C={C}")
    print(f"留出(泛化) 按题 argmax 准确 = {holdout_acc:.1%} ({nok}/{nq})   [dummy 50%]")
    print("学到的标准化权重（正=更可能是 winner）：")
    for name, w in sorted(zip(PAIR_COLS, coef), key=lambda t: -abs(t[1])):
        print(f"    {w:+.4f}  {name}")
    print(f"\n工件已写: {ARGS.out}")


if __name__ == "__main__":
    main()
