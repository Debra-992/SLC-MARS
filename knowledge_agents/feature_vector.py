# -*- coding: utf-8 -*-
"""全特征（图链 + 组学）逐基因向量构造器 —— lever 1「把 v20 忽略的组学通道接进排序头」的单一来源。

设计原则（与 graph_features.py 同源，杜绝 train/serve skew）：
  * 图的 12 列直接复用 gene_graph_feat（pairwise 模型、membership 校准、在线判定三处同一函数）。
  * 组学列（ppr / deg_corrected / in_graph / timer / km / DESeq2 / 旁系冗余）的抽取口径
    与 knowledge_agents.calib_dump._features 逐字一致——calibstore 里落盘的 membership `features`
    就是这套键，训练直接读落盘值、服务用本函数现算，两侧键名/语义完全对齐。
  * 只读证据字段（kg 路径行 / ck 工具产出），绝不触碰真值标签。

FULL_COLS / vec() 与 tools/train_membership_logit.py 的 `_membership_cols(True)` / `_vec`
同序同义，保证「用 calibstore 落盘特征训练的权重」能原样应用到「本函数在线构造的特征向量」上。
"""
from __future__ import annotations

from knowledge_agents.graph_features import gene_graph_feat

# —— 组学列分组（与 train_membership_logit 保持一致，顺序即模型系数顺序）——
NUMERIC_BASE = ["n_paths", "n_promote", "n_suppress", "n_cancer_hit", "ppr",
                "ppr_deg_corrected", "timer_rows", "de_log2fc", "de_sig_count",
                "de_panel_n", "paralog_redundancy"]
PRESENCE = ["in_graph", "timer_ok", "km_ok"]
MISSING_IND = ["ppr", "timer_rows", "de_log2fc"]
NUMERIC_GRAPH = ["n_ab_promote", "n_ab_suppress", "n_bc_promote", "n_bc_suppress",
                 "n_promote_matched", "n_suppress_matched",
                 "n_distinct_pathways", "n_distinct_pathways_matched"]

# 全特征列（图链拆细 + 组学 + 存在性/缺失指示），训练与服务共用同一列序
FULL_COLS = NUMERIC_BASE + PRESENCE + [f"has_{k}" for k in MISSING_IND] + NUMERIC_GRAPH


def vec(feat: dict, cols: list[str] = FULL_COLS) -> list[float]:
    """特征字典 -> 与 cols 同序的数值向量。
    has_X：X 缺失(None)记 0、有值记 1；PRESENCE：布尔/计数直接转 float；其余：None 记 0。
    与 train_membership_logit._vec 逐字同逻辑，杜绝两侧向量化口径漂移。"""
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


def features_for_gene(gene: str, path_rows: list[dict], cancer: str, ck) -> dict:
    """逐基因全特征：图的 12 列走 gene_graph_feat，组学列按 calib_dump._features 同一口径从 ck 抽。

    path_rows：本題候选池的全部 gene_disease_paths 原始行（本函数内部按 gene 过滤 + 截断由调用方负责）。
    ck：ComputedKnowledge（含 graph_computation / expression_by_gene / survival_by_gene /
        differential_expression / paralog_redundancy）。缺失一律记 None/0，绝不伪造、绝不当阴性。
    """
    g = (gene or "").upper()
    feat = dict(gene_graph_feat(g, path_rows, cancer))     # 12 个 n_* 图列

    # --- PPR 网络邻近度（候选内匹配）---
    ppr = deg = None
    in_graph = 0
    gc = getattr(ck, "graph_computation", None)
    if gc:
        for c in gc.candidates or []:
            if str(c.gene).upper() == g:
                ppr = c.ppr_score
                deg = c.degree_corrected_score
                in_graph = 1 if c.in_graph else 0
                break
    feat["ppr"] = ppr
    feat["ppr_deg_corrected"] = deg
    feat["in_graph"] = in_graph

    # --- TIMER 表达（行数 / 是否可用）---
    expr_map = getattr(ck, "expression_by_gene", None) or {}
    exp = expr_map.get(g) or expr_map.get(gene)
    if exp is None:
        exp = getattr(ck, "expression", None)
    feat["timer_rows"] = exp.row_count if exp is not None and getattr(exp, "status", None) else None
    feat["timer_ok"] = 1 if (exp is not None and getattr(exp, "status", None) is not None
                             and exp.status.is_knowledge) else 0

    # --- KM 生存（是否可用）---
    surv_map = getattr(ck, "survival_by_gene", None) or {}
    surv = surv_map.get(g) or surv_map.get(gene)
    if surv is None:
        surv = getattr(ck, "survival", None)
    feat["km_ok"] = 1 if (surv is not None and getattr(surv, "status", None) is not None
                          and surv.status.is_knowledge) else 0

    # --- DESeq2：目标基因带符号 log2FC、显著基因数、面板规模 ---
    de_l2fc = None
    de_sig = de_n = None
    de = getattr(ck, "differential_expression", None)
    if de is not None:
        de_sig = de.significant_count
        de_n = len(de.top_genes or [])
        for row in de.top_genes or []:
            gg = str(row.get("gene_symbol") or row.get("gene") or "").upper()
            if gg == g:
                try:
                    de_l2fc = float(row.get("log2fc"))
                except (TypeError, ValueError):
                    de_l2fc = None
                break
    feat["de_log2fc"] = de_l2fc
    feat["de_sig_count"] = de_sig
    feat["de_panel_n"] = de_n

    # --- 旁系缓冲冗余 ∈ [0,1] ---
    feat["paralog_redundancy"] = round((getattr(ck, "paralog_redundancy", None) or {}).get(g, 0.0), 4)
    return feat
