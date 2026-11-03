# -*- coding: utf-8 -*-
"""图谱 SLC->Pathway->Disease 链的【逐基因图特征】——训练与在线打分的唯一来源。

pairwise 图特征 logistic 在离线拟合（tools/fit_pairwise_model.py）、离线评测
（tools/train_membership_logit.py）与 Vega 在线判定（decision_action）三处都用到这里的
`gene_graph_feat` 与 `PAIR_COLS`，以杜绝 train/serve skew：三处看到的特征值必须逐位一致。

行来源有两种键名风格，本函数都能吃：
  * 在线 kg.evidence 的 gene_disease_paths 原始行：{gene, pathway, rel_AB, rel_BC,
    gene_disease_effect, disease, ...}
  * calib_dump 落盘的精简 kg_paths 行：{gene, pathway, rel_AB, rel_BC, effect, disease}
matched（命中本癌种）判定与启发式保持同一口径：`cancer(小写) in disease(小写)` 子串。
"""
from __future__ import annotations

# 逐基因图特征列（顺序即模型系数顺序，写入工件、在线打分按此展开）
PAIR_COLS = ["n_paths", "n_promote", "n_suppress", "n_cancer_hit",
             "n_ab_promote", "n_ab_suppress", "n_bc_promote", "n_bc_suppress",
             "n_promote_matched", "n_suppress_matched",
             "n_distinct_pathways", "n_distinct_pathways_matched"]


def _effect_of(row: dict) -> str:
    return str(row.get("gene_disease_effect") or row.get("effect") or "").upper()


def gene_graph_feat(gene: str, path_rows: list[dict], cancer: str) -> dict:
    """从 gene_disease_paths 原始行里抽出该基因沿 SLC->Pathway->Disease 链的拆细图特征。

    只读证据字段（gene/pathway/rel_AB/rel_BC/effect/disease），绝不触碰真值标签。
    """
    gene_u = (gene or "").upper()
    cancer_l = (cancer or "").lower()
    f = {k: 0 for k in PAIR_COLS if k.startswith("n_") and "distinct" not in k}
    pw: set[str] = set()
    pwm: set[str] = set()
    for r in path_rows or []:
        if not isinstance(r, dict) or str(r.get("gene", "")).upper() != gene_u:
            continue
        f["n_paths"] += 1
        eff = _effect_of(r)
        ab = str(r.get("rel_AB") or "").upper()
        bc = str(r.get("rel_BC") or "").upper()
        f["n_promote"] += eff == "PROMOTION"
        f["n_suppress"] += eff == "SUPPRESSION"
        f["n_ab_promote"] += ab == "PROMOTION"
        f["n_ab_suppress"] += ab == "SUPPRESSION"
        f["n_bc_promote"] += bc == "PROMOTION"
        f["n_bc_suppress"] += bc == "SUPPRESSION"
        p = str(r.get("pathway") or "").upper()
        matched = bool(cancer_l) and cancer_l in str(r.get("disease", "")).lower()
        if matched:
            f["n_cancer_hit"] += 1
            f["n_promote_matched"] += eff == "PROMOTION"
            f["n_suppress_matched"] += eff == "SUPPRESSION"
            if p:
                pwm.add(p)
        if p:
            pw.add(p)
    f["n_distinct_pathways"] = len(pw)
    f["n_distinct_pathways_matched"] = len(pwm)
    return f


def feat_vec(feat: dict, cols: list[str] = PAIR_COLS) -> list[float]:
    """特征字典 -> 与 cols 同序的数值向量（缺列按 0）。"""
    return [float(feat.get(c, 0) or 0) for c in cols]


def collect_path_rows(kg) -> list[dict]:
    """从 BackgroundKnowledge 里汇集所有 gene_disease_paths 行（含邻居补来的符号行）。"""
    rows: list[dict] = []
    for ev in (getattr(kg, "evidence", []) or []):
        if getattr(ev, "operation", None) != "gene_disease_paths":
            continue
        for r in ev.rows or []:
            if isinstance(r, dict):
                rows.append(r)
    return rows
