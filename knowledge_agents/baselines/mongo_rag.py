# -*- coding: utf-8 -*-
"""文献 RAG 检索器：题目 -> MongoDB papers({doi,title,abstract}) 关键词检索 + 词重叠重排。

无 embedding 模型（端点不提供、也不引外部模型），用透明可复核的两级召回：
  1) Mongo 正则粗召回：题面任一 SLC 基因符号出现在 title/abstract 即入候选（按基因缓存）。
  2) Python 重排：题目疾病/癌种词命中数（title×3 + abstract×1）+ 基因出现次数（abstract 每 2 次记 1），
     取 top_k；每篇 abstract 截断，防止 prompt 爆炸。
凭证只从环境变量读（MONGO_URI，run_route_b.ps1 已导出），绝不写进源码。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

STOP = frozenset({
    "gene", "genes", "cell", "cells", "line", "lines", "human", "mouse", "days",
    "day", "week", "weeks", "the", "and", "for", "with", "that", "this", "from",
    "cancer", "cancers", "tumor", "tumour", "cells", "screen", "knockout", "ko",
})


def _disease_terms(question: str, constraints: dict) -> list[str]:
    """从题面/约束提取疾病与癌种词（小写、去停用词、长度>3）。"""
    text = question + " " + " ".join(str(v) for v in (constraints or {}).values())
    words = [w for w in re.split(r"[^a-zA-Z0-9\-]+", text.lower()) if len(w) > 3 and w not in STOP]
    return list(dict.fromkeys(words))


def _text(v) -> str:
    """papers 里 title/abstract 可能是 XML 摄入残留下的 {'sup':..,'#text':..} dict -> 归一成纯文本。"""
    if isinstance(v, dict):
        return str(v.get("#text") or v.get("text") or " ".join(str(x) for x in v.values())).strip()
    return str(v or "").strip()


@lru_cache(maxsize=256)
def _papers_for_gene(gene: str) -> tuple:
    from pymongo import MongoClient  # 延迟导入：无 mongo 环境时不影响其他 baseline
    uri = os.environ.get("MONGO_URI", "")
    if not uri:
        raise RuntimeError("MONGO_URI 未设置（run_route_b.ps1 导出；凭证不进源码）")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    col = client[os.environ.get("MONGO_DB_NAME", "SLCdb")][
        os.environ.get("MONGO_COLLECTION_PAPERS", "papers")]
    rx = re.compile(r"\b" + re.escape(gene) + r"\b", re.I)
    cur = col.find({"$or": [{"title": rx}, {"abstract": rx}]},
                   {"title": 1, "abstract": 1, "doi": 1}, limit=400)
    return tuple((_text(d.get("title")), _text(d.get("abstract"))[:2400], _text(d.get("doi"))) for d in cur)


def retrieve(question: str, candidates: list[str], constraints: dict, top_k: int = 5) -> list[dict]:
    pool: dict[str, tuple] = {}
    for g in candidates[:2]:
        for p in _papers_for_gene(g.upper()):
            pool[p[0] or p[2]] = p
    terms = _disease_terms(question, constraints)
    scored = []
    for title, abstract, doi in pool.values():
        t, a = title.lower(), abstract.lower()
        score = sum(3 * t.count(w) + a.count(w) for w in terms)
        score += sum((a.count(g.lower()) + t.count(g.lower())) // 2 for g in candidates)
        scored.append((score, title, abstract, doi))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [{"score": s, "title": ti, "abstract": ab, "doi": do}
            for s, ti, ab, do in scored[:top_k] if s > 0 or scored]


def format_context(papers: list[dict], max_chars: int = 4000) -> str:
    if not papers:
        return "[文献检索] 未命中任何相关文献。请按你自身知识回答。"
    blocks, used = [], 0
    for i, p in enumerate(papers, 1):
        blk = f"[{i}] {p['title']} (doi:{p['doi']})\n{p['abstract'][:900]}"
        if used + len(blk) > max_chars:
            break
        blocks.append(blk)
        used += len(blk)
    return ("可检索到的文献摘录（来自项目文献库 title/abstract，仅供知识补充）：\n"
            + "\n\n".join(blocks))
