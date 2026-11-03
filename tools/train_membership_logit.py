# -*- coding: utf-8 -*-
"""⑤ membership / pairwise 的【免泄漏 logistic 校准】：用 clean 真值学习证据→必需性的权重，
取代决策层里手工设定的常数（W_BUFFER / 0.3+0.12|net| 等）。

本版新增（把图谱 SLC->Pathway->Disease 这条链【拆细榨干】）：
  * membership 特征【消融】：base(旧:复合极性计数+组学) vs base+graph(rel_AB/rel_BC 两段方向、
    本癌种通路促/抑拆分、去重中介通路数)，用重复 CV-AUC 判断"图链拆细能否把 0.73 再抬高"。
  * pairwise【图特征 pointwise 模型】：把"二选一"归约为逐基因打分——同题内 winner 记正例、
    loser 记负例，per-gene 图特征从落盘的 kg_paths 原始行离线重建（故无需重跑池），
    评测时两基因各打分取 argmax；与启发式在同一 qid 不相交 test 子集上对比，附 dummy 下限。

严格防泄漏 & 公平评测：
  * 特征只来自 8 个工具的原始数值 / 原始路径行（calib_dump 落盘），绝不含真值。
  * 训练 / 评测按 qid 哈希【不相交】切分：同一题的标签只会出现在一侧，杜绝记忆泄漏。
  * 概率经 CalibratedClassifierCV 校准，可作为"置信度"喂给保形弃答。

用法：
  python tools/train_membership_logit.py                 # membership 消融 + 保形
  python tools/train_membership_logit.py --pairwise      # pairwise 图特征模型
  [--traces outputs/traces/calibstore.jsonl] [--seed 5] [--train-frac 0.6] [--alphas 0.10,0.20]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledge_agents.graph_features import PAIR_COLS, gene_graph_feat  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------- membership 特征组 ----------------
# base：与 v9 那次拿到 AUC=0.73 的同一组（复合极性计数 + 组学 + 旁系），向后兼容
NUMERIC_BASE = ["n_paths", "n_promote", "n_suppress", "n_cancer_hit", "ppr",
                "ppr_deg_corrected", "timer_rows", "de_log2fc", "de_sig_count",
                "de_panel_n", "paralog_redundancy"]
PRESENCE = ["in_graph", "timer_ok", "km_ok"]
MISSING_IND = ["ppr", "timer_rows", "de_log2fc"]
# graph-拆细：把 SLC->Pathway->Disease 链的两段方向/本癌种通路极性/通路多样性单独喂进去
NUMERIC_GRAPH = ["n_ab_promote", "n_ab_suppress", "n_bc_promote", "n_bc_suppress",
                 "n_promote_matched", "n_suppress_matched",
                 "n_distinct_pathways", "n_distinct_pathways_matched"]


def _vec(feat: dict, cols: list[str]) -> list[float]:
    row = []
    for k in cols:
        if k.startswith("has_"):
            row.append(0.0 if feat.get(k[4:]) is None else 1.0)
        elif k in PRESENCE:
            row.append(float(feat.get(k, 0) or 0))
        else:
            v = feat.get(k)
            row.append(0.0 if v is None else float(v))
    return row


def _membership_cols(use_graph: bool) -> list[str]:
    cols = list(NUMERIC_BASE) + PRESENCE + [f"has_{k}" for k in MISSING_IND]
    if use_graph:
        cols = cols + NUMERIC_GRAPH
    return cols


def _split(qid: str, seed: int, train_frac: float) -> str:
    h = int(hashlib.sha1(f"{seed}|{qid}".encode()).hexdigest(), 16) % 1000
    return "train" if h < int(train_frac * 1000) else "test"


def _load_rows(path: str) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _fit_eval(Xtr, ytr, Xte, yte, seed: int):
    """train 内网格选 C -> 3 折概率校准 -> 返回 (C, auc, acc, brier, cal, p_te)。"""
    base = LogisticRegression(max_iter=2000)
    gs = GridSearchCV(base, {"C": [0.01, 0.03, 0.1, 0.3, 1.0]}, scoring="neg_log_loss",
                      cv=StratifiedKFold(3, shuffle=True, random_state=seed))
    gs.fit(Xtr, ytr)
    C = gs.best_params_["C"]
    cal = CalibratedClassifierCV(LogisticRegression(C=C, max_iter=2000), method="sigmoid", cv=3)
    cal.fit(Xtr, ytr)
    p_te = cal.predict_proba(Xte)[:, 1]
    pred = (p_te >= 0.5).astype(int)
    try:
        auc = roc_auc_score(yte, p_te)
    except ValueError:
        auc = float("nan")
    return C, auc, float((pred == yte).mean()), brier_score_loss(yte, p_te), cal


def _cv_auc(X, y, C: int, seed: int, reps: int = 10, folds: int = 5) -> tuple[float, float]:
    """重复 K 折 CV 的 AUC（全体样本、含 held-out），衡量特征集的判别力稳定性。"""
    aucs = []
    for r in range(reps):
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed + r)
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            if len(np.unique(y[tr])) < 2:
                continue
            m = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        try:
            aucs.append(roc_auc_score(y, oof))
        except ValueError:
            pass
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.std(aucs))


def _conformal_tau(conf: np.ndarray, correct: np.ndarray, alpha: float):
    """有限样本风险控制：按置信降序遍历【全部】前缀，取满足
    (累计错误+1)/(覆盖+1) <= alpha 的最大覆盖前缀的最低置信。小前缀被 +1 下限卡住(1/(k+1)≥α)，
    不能一遇违反就 break——更大前缀反而可能达标。"""
    order = np.argsort(-conf)
    c, y = conf[order], correct[order]
    best_tau, cum_err = None, 0.0
    for i in range(len(c)):
        cum_err += (0 if y[i] else 1)
        if (cum_err + 1) / (i + 2) <= alpha:
            best_tau = c[i]
    return best_tau


# pairwise 的 per-gene 图特征与列序统一取自 knowledge_agents.graph_features（训练/服务单一来源）。


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=os.path.join("outputs", "traces", "calibstore.jsonl"))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--alphas", default="0.10,0.20")
    ap.add_argument("--pairwise", action="store_true", help="改跑 pairwise 图特征模型")
    ARGS = ap.parse_args()

    if not os.path.exists(ARGS.traces):
        print(f"找不到校准池: {ARGS.traces}（先跑 calib_dump）")
        return
    rows = _load_rows(ARGS.traces)

    if ARGS.pairwise:
        run_pairwise(rows, ARGS)
    else:
        run_membership(rows, ARGS)


def run_membership(rows, ARGS):
    tr, te = {"base": [[], []], "graph": [[], []]}, {"base": [[], []], "graph": [[], []]}
    heur_te = []
    n_mem = 0
    for r in rows:
        if r.get("error") or r.get("bucket") != "membership" or r.get("label") is None:
            continue
        if not r.get("features"):
            continue
        n_mem += 1
        y = int(r["label"])
        fb = _vec(r["features"], _membership_cols(False))
        fg = _vec(r["features"], _membership_cols(True))
        side = _split(r["qid"], ARGS.seed, ARGS.train_frac)
        dst = tr if side == "train" else te
        dst["base"][0].append(fb); dst["base"][1].append(y)
        dst["graph"][0].append(fg); dst["graph"][1].append(y)
        if side == "test":
            he = (r.get("systems", {}).get("pipeline_heuristic") or {})
            heur_te.append(he.get("outcome"))
    ntr = len(tr["base"][1]); nte = len(te["base"][1])
    print(f"membership 有效样本 {n_mem} | 训练 {ntr} / 测试 {nte}（qid 不相交, seed={ARGS.seed}）")
    if ntr < 20 or nte < 10:
        print("样本太少，跳过拟合。")
        return

    ytr_b = np.array(tr["base"][1]); yte_b = np.array(te["base"][1])
    dummy = max(float(yte_b.mean()), 1 - float(yte_b.mean()))

    print("\n===== membership：图链拆细【消融】(重复 10×5 折 CV-AUC, 全体样本) =====")
    results = {}
    for name, use_g in (("base", False), ("graph", True)):
        cols = _membership_cols(use_g)
        Xtr = np.array(tr[name][0], float); Xte = np.array(te[name][0], float)
        C, auc, acc, brier, cal = _fit_eval(Xtr, ytr_b, Xte, yte_b, ARGS.seed)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xm = (np.vstack([Xtr, Xte]) - mu) / sd
        ym = np.concatenate([ytr_b, yte_b])
        cv_m, cv_s = _cv_auc(Xm, ym, C, ARGS.seed)
        label = "base(复合极性+组学)" if not use_g else "base + graph(拆细)"
        results[name] = {"cols": cols, "Xtr": Xtr, "Xte": Xte, "ytr": ytr_b,
                         "cal": cal, "C": C}
        print(f"  {label:<24} C={C:<5} 留出AUC={auc:.3f} 留出acc={acc:.1%} "
              f"| 重复CV-AUC={cv_m:.3f}±{cv_s:.3f}")

    G = results["graph"]
    cols, Xtr, ytrb, C = G["cols"], G["Xtr"], G["ytr"], G["C"]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    w = LogisticRegression(C=C, max_iter=2000).fit((Xtr - mu) / sd, ytrb).coef_[0]
    print("\n  —— base+graph 学到的标准化权重（正=更可能必需，*=图链拆细新特征）——")
    for name, coef in sorted(zip(cols, w), key=lambda t: -abs(t[1]))[:14]:
        if abs(coef) < 1e-3:
            continue
        star = "*" if name in NUMERIC_GRAPH else " "
        print(f"   {coef:+.3f} {star} {name}")

    # logistic(graph) + 保形
    calg, Xtrg, ytrg = G["cal"], G["Xtr"], G["ytr"]
    Xte_g = G["Xte"]
    p_te = calg.predict_proba(Xte_g)[:, 1]
    pred = (p_te >= 0.5).astype(int)
    yte = np.array(te["graph"][1], int)
    conf_tr = np.max(calg.predict_proba(Xtrg), axis=1)
    correct_tr = (calg.predict_proba(Xtrg)[:, 1] >= 0.5).astype(int) == ytrg
    conf_te = np.maximum(p_te, 1 - p_te)
    correct_te = (pred == yte)

    hn = len(heur_te)
    hans = [o for o in heur_te if o in ("correct", "wrong")]
    hcorr = sum(1 for o in hans if o == "correct")
    print(f"\n  多数类 dummy={dummy:.1%} | logistic(graph)@0.5 acc={float((pred==yte).mean()):.1%}")
    print(f"  启发式(同 test)：覆盖 {len(hans)/hn if hn else 0:.1%}({len(hans)}/{hn}) "
          f"已答准确 {hcorr/len(hans) if hans else float('nan'):.1%} "
          f"净贡献 {hcorr}/{hn}={(hcorr/hn if hn else 0):.1%}")

    print("\n----- logistic(graph) + 保形弃答（test 验证）-----")
    print(f"{'alpha':>6} {'tau':>6} {'覆盖':>8} {'已答准确':>10} {'净贡献':>8}")
    for a in [float(x) for x in ARGS.alphas.split(",")]:
        tau = _conformal_tau(conf_tr, correct_tr, a)
        if tau is None:
            print(f"{a:>6.2f} {'—':>6} {'0%':>8} {'—':>10}  (α={a:.2f} 训练池不足以认证→全弃答)")
            continue
        sel = conf_te >= tau
        cov = float(sel.mean())
        accans = float(correct_te[sel].mean()) if sel.any() else float("nan")
        net = float((correct_te[sel]).sum()) / len(yte)
        print(f"{a:>6.2f} {tau:>6.2f} {cov:>7.1%} {accans:>9.1%} {net:>7.1%}")


def run_pairwise(rows, ARGS):
    """pointwise 归约：同题 winner 记正例、loser 记负例；per-gene 图特征从 kg_paths 离线重建。
    评测按【qid 不相交】，test 上每题两基因各打分取 argmax，比启发式与 dummy。"""
    # 收集 (qid, gene, label(是否winner), featvec)
    recs = []
    heur_te = {}
    for r in rows:
        if r.get("error") or r.get("bucket") != "pairwise":
            continue
        gt = r.get("ground_truth") or {}
        winner = str(gt.get("winner", "")).upper()
        genes = [str(g).upper() for g in (r.get("effective_genes") or [])][:2]
        cancer = r.get("cancer", "")
        kgp = r.get("kg_paths") or []
        if len(genes) != 2 or not winner or winner not in genes:
            continue
        loser = genes[1] if genes[0] == winner else genes[0]
        side = _split(r["qid"], ARGS.seed, ARGS.train_frac)
        he = (r.get("systems", {}).get("pipeline_heuristic") or {})
        heur_te[r["qid"]] = (side, he.get("outcome"), winner, genes)
        for g, lab in ((winner, 1), (loser, 0)):
            recs.append((r["qid"], g, lab, _vec(gene_graph_feat(g, kgp, cancer), PAIR_COLS), side))

    qids = sorted({q for q, *_ in recs})
    ntr = sum(1 for x in recs if x[4] == "train")
    nte_q = sum(1 for q in qids if heur_te[q][0] == "test")
    print(f"pairwise 有效题 {len(qids)}（每拆 2 基因行）| 训练基因行 {ntr} | test 题 {nte_q}（seed={ARGS.seed}）")
    if ntr < 30 or nte_q < 8:
        print("样本太少，跳过 pairwise 拟合。")
        return

    Xtr = np.array([x[3] for x in recs if x[4] == "train"], float)
    ytr = np.array([x[2] for x in recs if x[4] == "train"], int)
    Xte_all = np.array([x[3] for x in recs if x[4] == "test"], float)
    yte_all = np.array([x[2] for x in recs if x[4] == "test"], int)

    base = LogisticRegression(max_iter=2000)
    gs = GridSearchCV(base, {"C": [0.01, 0.03, 0.1, 0.3, 1.0]}, scoring="neg_log_loss",
                      cv=StratifiedKFold(3, shuffle=True, random_state=ARGS.seed))
    gs.fit(Xtr, ytr)
    C = gs.best_params_["C"]
    cal = CalibratedClassifierCV(LogisticRegression(C=C, max_iter=2000), method="sigmoid", cv=3)
    cal.fit(Xtr, ytr)

    # test：按题聚合两基因，一次批量打分取 argmax
    test_recs = [x for x in recs if x[4] == "test"]
    byq: dict[str, list] = {}
    for qid, g, lab, fv, _ in test_recs:
        byq.setdefault(qid, []).append((g, fv))
    nq = nok = 0
    for qid, items in byq.items():
        if len(items) < 2:
            continue
        Xq = np.array([fv for _, fv in items], float)
        scores = cal.predict_proba(Xq)[:, 1]
        pred_gene = items[int(np.argmax(scores))][0]
        truth_winner = heur_te[qid][2]
        nq += 1
        nok += pred_gene.upper() == truth_winner.upper()
    # dummy：二选一瞎猜 50%
    print(f"\n===== pairwise：图特征 logistic（pointwise 归约, test 上 argmax）=====")
    print(f"训练内 CV 选 C={C}")
    print(f"  dummy(二选一) = 50.0%")
    if nq:
        print(f"  logistic winner-argmax 准确 = {nok/nq:.1%} ({nok}/{nq} 题)")
    hans = [heur_te[q][1] for q in byq if heur_te[q][1] in ("correct", "wrong")]
    hok = sum(1 for o in hans if o == "correct")
    if byq:
        print(f"  启发式(同 test)：答题 {len(hans)}/{nq} 已答准确 {hok/len(hans) if hans else float('nan'):.1%} "
              f"净贡献 {hok}/{nq}={hok/nq if nq else 0:.1%}")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    w = LogisticRegression(C=C, max_iter=2000).fit((Xtr - mu) / sd, ytr).coef_[0]
    print("\n  —— pairwise 图特征学到的权重（正=更可能是 winner/必需）——")
    for name, coef in sorted(zip(PAIR_COLS, w), key=lambda t: -abs(t[1])):
        if abs(coef) < 1e-3:
            continue
        print(f"    {coef:+.3f}  {name}")


if __name__ == "__main__":
    main()
