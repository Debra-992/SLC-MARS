# -*- coding: utf-8 -*-
"""旁系缓冲（paralog buffering）——从『该癌种已测 SLC 面板』估计候选基因的功能冗余。

机理：SLC 基因按亚家族（SLC<编号>，如 SLC7 / SLC25 / SLC12）成簇，同亚家族成员常功能
重叠。若在目标癌种中有多个同亚家族旁系被共表达，则敲除该基因更易被旁系『缓冲』，
CRISPR loss-of-function 更不产生显著必需性表型 -> 提示 membership 偏阴性；反之若该基因
在此癌种近乎孤儿（无共表达旁系），这只是『无缓冲证据』，不能据此臆断其为必需基因。

因此本通道是【单边证据】：只用于削弱『必需/阳性』的判断，绝不凭"无旁系"制造阳性。

数据来源：DESeq2 top_genes（该癌种里显著且被检测的 SLC 面板）。拿不到面板 -> 不写入（未知）。
"""
from __future__ import annotations

import re

_SLC_FAMILY_RE = re.compile(r"^SLC(\d+)", re.I)
# 伪基因 / 反义 / 内含子转录本等非编码符号：它们无法在功能上"缓冲"敲除，需从旁系计数剔除。
#   典型：SLC25A14P1 / SLC25A5P5（P+数字伪基因尾缀）、SLC22A18AS / SLC7A6OS（AS/OS 反义重叠）、
#         SLC7A2-IT1 / SLC25A21-AS1（连字符 IT/AS）。
_PSEUDO_RE = re.compile(r"(P\d+$)|(-)|(?:AS|OS|SN)$")


def is_functional_symbol(gene: str) -> bool:
    """粗略判定是否为『可能编码功能蛋白』的正式 SLC 基因（剔除伪基因/反义/内含子转录本）。"""
    g = (gene or "").strip().upper()
    return bool(g) and not _PSEUDO_RE.search(g)


def subfamily(gene: str) -> str | None:
    """取 SLC 亚家族键（SLC 家族编号为纯数字）。SLC7A11->SLC7, SLC25A1->SLC25,
    SLC12A2->SLC12, SLC3A1->SLC3。非 SLC 命名 -> None（不参与旁系冗余估计）。"""
    m = _SLC_FAMILY_RE.match((gene or "").strip())
    return ("SLC" + m.group(1)) if m else None


_LNCAP = 5.0  # |log2FC| 饱和上限：把差异表达幅度归一到 [0,1] 作为"该旁系在本癌种活跃表达"权重


def panel_from_de(de) -> dict[str, float]:
    """把 DifferentialExpression.top_genes 汇成 {基因(大写): 表达权重}。
    权重 = min(|log2FC|, LNCAP)/LNCAP ∈ [0,1]：不仅看『是否显著』，还看『差异表达幅度』——
    在本癌种强烈失调的旁系更可能实际承担功能、从而更能缓冲敲除。仅保留功能符号，剔除伪基因。"""
    panel: dict[str, float] = {}
    if not de or not getattr(de, "top_genes", None):
        return panel
    for row in de.top_genes or []:
        g = str(row.get("gene_symbol") or row.get("gene") or "").upper()
        if not g or not is_functional_symbol(g):
            continue
        l2 = row.get("log2fc")
        try:
            w = min(abs(float(l2)), _LNCAP) / _LNCAP if l2 is not None else 0.0
        except (TypeError, ValueError):
            w = 0.0
        panel[g] = max(panel.get(g, 0.0), w)   # 去重取最大幅度
    return panel


def redundancy(gene: str, panel: dict[str, float], k_sat: float = 4.0) -> tuple[float, list[str]]:
    """返回 (redundancy∈[0,1], 共表达旁系列表)。
    redundancy = min(1, Σ_旁系 表达权重 / k_sat)：按旁系的『表达幅度』加权累加，
    大家族/强表达旁系自然获得更高缓冲分（family-size 分级内生于累加）。
    - 亚家族无法判定（非 SLC 命名）-> (0.0, [])；
    - 面板为空/仅含自身/旁系表达权重≈0 -> (0.0, [])（无缓冲证据，不臆断必需）。"""
    fam = subfamily(gene)
    if not fam:
        return 0.0, []
    self_g = (gene or "").strip().upper()
    partners = sorted(p for p in panel
                      if p != self_g and subfamily(p) == fam and is_functional_symbol(p))
    if not partners or k_sat <= 0:
        return 0.0, partners
    raw = sum(float(panel.get(p, 0.0)) for p in partners)
    return min(1.0, raw / float(k_sat)), partners


def redundancy_map(candidates: list[str], panel: dict[str, float], k_sat: float = 4.0) -> tuple[dict[str, float], dict[str, list[str]]]:
    """对全部候选基因批量计算冗余。键统一大写，与决策层 gene.upper() 对齐。"""
    red: dict[str, float] = {}
    part: dict[str, list[str]] = {}
    for g in candidates:
        gu = (g or "").strip().upper()
        if not gu:
            continue
        r, ps = redundancy(gu, panel, k_sat)
        red[gu] = r
        part[gu] = ps
    return red, part
