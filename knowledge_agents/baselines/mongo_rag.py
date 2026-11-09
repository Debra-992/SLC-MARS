# -*- coding: utf-8 -*-
"""Literature RAG retriever: question -> keyword search over MongoDB papers ({doi,title,abstract}) plus word-overlap re-ranking.

No embedding model (the endpoint does not offer one and no external model is pulled in), so recall is a transparent,
  1) Mongo regex coarse recall: any SLC gene symbol from the stem appearing in title/abstract qualifies (cached per gene).
  2) Python re-rank: disease/cancer-term hits in the question (title x3 + abstract x1) + gene occurrence count (abstract
     every 2nd occurrence counts once), keep top_k; each abstract is truncated so the prompt cannot blow up.
Credentials are read from the environment only (MONGO_URI, exported by run_route_b.ps1) and never written into source.
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
    """Extract disease and cancer-type words from the question stem / constraints (lowercase, stop words dropped, length > 3)."""
    text = question + " " + " ".join(str(v) for v in (constraints or {}).values())
    words = [w for w in re.split(r"[^a-zA-Z0-9\-]+", text.lower()) if len(w) > 3 and w not in STOP]
    return list(dict.fromkeys(words))


def _text(v) -> str:
    """title/abstract in papers may be an XML ingestion leftover such as {'sup':..,'#text':..} -> flatten to plain text."""
    if isinstance(v, dict):
        return str(v.get("#text") or v.get("text") or " ".join(str(x) for x in v.values())).strip()
    return str(v or "").strip()


@lru_cache(maxsize=256)
def _papers_for_gene(gene: str) -> tuple:
    from pymongo import MongoClient  # lazy import: other baselines keep working without mongo
    uri = os.environ.get("MONGO_URI", "")
    if not uri:
        raise RuntimeError("MONGO_URI is not set (exported by run_route_b.ps1; credentials never live in source)")
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
        return "[literature search] No relevant paper matched. Answer from your own knowledge."
    blocks, used = [], 0
    for i, p in enumerate(papers, 1):
        blk = f"[{i}] {p['title']} (doi:{p['doi']})\n{p['abstract'][:900]}"
        if used + len(blk) > max_chars:
            break
        blocks.append(blk)
        used += len(blk)
    return ("Literature excerpts found by retrieval (title/abstract from the project paper library, background only):\n"
            + "\n\n".join(blocks))
