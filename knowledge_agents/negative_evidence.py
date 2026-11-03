# -*- coding: utf-8 -*-
"""v18 ②：membership 负证据通道 —— 文献共现富集/旁系对比判别（结构化事实，非原文堆料）。

动机（§二十二 终表）：B·membership 全场 ≈ 抛硬币（v16 44.4%，最强基线 47.2%），图谱正链
已封顶（§二十）。本通道给决策层提供**判别性**新信号：目标基因×癌种在项目文献库的
共现富集度 R = 观测共现 / 独立期望，并与旁系同源基因横向对比——"别人家的旁系都在谈
这个癌种、就你没有"是**负证据**；"只有你在谈"是正证据。全部为观测数值，不含管线结论
（_facts_digest 原则），且只读公开文献库，不接触 GT。

集成方式：supply 段把结果做成一条合成 KGEvidence(operation="lit_cooc_contrast") 挂进
kg.evidence —— Vega 证据通道(_llm_analogy→_evidence_brief)、法官/主审(_facts_digest)、
triad 事实呈报自动全部可见，决策层零签名改动；启发式按 op 名取数，未知 op 自然忽略。

失败姿态（宁等不污染/v15 教训）：Mongo 不可达或任何异常 => 返回空列表，本题无此通道，
行为逐字节回退 v16，绝不抛异常、绝不伪造行。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

_ON = {"1", "true", "on", "yes"}


def neg_evid_enabled() -> bool:
    """v18 门控：SLC_NEG_EVID（默认关 = 定版 v20 逐字节，v20 不含该通道）。"""
    return os.environ.get("SLC_NEG_EVID", "0").lower() in _ON


def _cancer_terms(cancer: str) -> list[str]:
    """癌种串 -> 检索词（小写、去噪、含缩写与全词；49628 篇 title+abstract 匹配用）。"""
    toks = [t for t in re.split(r"[^A-Za-z0-9]+", (cancer or "").lower()) if len(t) > 2]
    whole = (cancer or "").strip().lower()
    out = list(dict.fromkeys([w for w in [whole, *toks] if w and w not in {"human", "mouse", "cell", "line", "days", "ko"}]))
    return out


@lru_cache(maxsize=256)
def _gene_papers(gene: str) -> tuple[tuple[str, str], ...]:
    """gene 命中的 (title, abstract) 池（mongo_rag 同款正则粗召回，limit 400）。"""
    from pymongo import MongoClient
    uri = os.environ.get("MONGO_URI", "")
    if not uri:
        raise RuntimeError("MONGO_URI 未设置")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    col = client[os.environ.get("MONGO_DB_NAME", "SLCdb")][
        os.environ.get("MONGO_COLLECTION_PAPERS", "papers")]
    rx = re.compile(r"\b" + re.escape(gene) + r"\b", re.I)
    cur = col.find({"$or": [{"title": rx}, {"abstract": rx}]},
                   {"title": 1, "abstract": 1}, limit=400)
    return tuple((_flat(d.get("title")), _flat(d.get("abstract"))) for d in cur)


@lru_cache(maxsize=64)
def _cancer_count(terms_key: str) -> int:
    """癌种背景文献量（全库正则计数，按癌种缓存；terms_key="|".join(terms)）。"""
    from pymongo import MongoClient
    terms = [t for t in terms_key.split("|") if t]
    uri = os.environ.get("MONGO_URI", "")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    col = client[os.environ.get("MONGO_DB_NAME", "SLCdb")][
        os.environ.get("MONGO_COLLECTION_PAPERS", "papers")]
    rx = re.compile("|".join(re.escape(t) for t in terms), re.I)
    return col.count_documents({"$or": [{"title": rx}, {"abstract": rx}]})


@lru_cache(maxsize=1)
def _total_papers() -> int:
    from pymongo import MongoClient
    uri = os.environ.get("MONGO_URI", "")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    return client[os.environ.get("MONGO_DB_NAME", "SLCdb")][
        os.environ.get("MONGO_COLLECTION_PAPERS", "papers")].estimated_document_count()


def _flat(v) -> str:
    """papers 的 title/abstract 可能是 XML 摄入 dict({'#text':..}) -> 归一纯文本（同 mongo_rag）。"""
    if isinstance(v, dict):
        return str(v.get("#text") or v.get("text") or " ".join(str(x) for x in v.values())).strip()
    return str(v or "").strip()


def _contrast(gene: str, terms: list[str]) -> dict[str, Any] | None:
    """单基因 × 癌种的共现富集观测：R = 观测共现 / 独立期望（>=2 显著富集，<=0.5 反关联）。"""
    papers = _gene_papers(gene.upper())
    if not papers:
        return None
    n_g = len(papers)
    rx = re.compile("|".join(re.escape(t) for t in terms), re.I) if terms else None
    n_gC = sum(1 for ti, ab in papers if rx and (rx.search(ti) or rx.search(ab)))
    n_C = _cancer_count("|".join(terms)) if terms else 0
    total = _total_papers()
    exp = n_g * n_C / max(total, 1)
    return {"gene": gene.upper(), "gene_papers": n_g, "cancer_papers": n_C,
            "cooccur_papers": n_gC, "expected_if_independent": round(exp, 2),
            "enrichment_R": round(n_gC / exp, 2) if exp > 0 else None}


def cooc_contrast_rows(gene: str, cancer: str,
                        paralogs: tuple[str, ...] | list[str] = ()) -> list[dict]:
    """membership 题的判别观测行（目标基因 + 至多 2 个旁系对照）。任何异常 -> []（静默退化）。

    旁系来源：ck.paralog_redundancy 的键（图谱同源家族），提供"家族内癌种特异性"对比。
    附 caveat 行防止决策层把"共现缺失"过度解读为"非必要"（文献偏置客观存在）。"""
    try:
        terms = _cancer_terms(cancer)
        if not terms or not gene:
            return []
        rows: list[dict] = []
        tgt = _contrast(gene, terms)
        if tgt is None:
            return []
        rows.append(tgt)
        for p in list(paralogs or [])[:2]:
            if str(p).upper() == str(gene).upper():
                continue
            r = _contrast(str(p), terms)
            if r is not None:
                r["role"] = "paralog_contrast"
                rows.append(r)
        rows.append({"caveat": "co-occurrence is bibliometric signal, not proof; "
                              "absence of co-occurrence may reflect literature bias"})
        return rows
    except Exception as e:  # noqa BLE001 —— 基础设施噪声不得污染判决
        import sys
        print(f"[negative_evidence] 退化(v16 行为): {type(e).__name__}: {str(e)[:120]}",
              file=sys.stderr, flush=True)
        return []
