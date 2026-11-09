# -*- coding: utf-8 -*-
"""2) Membership negative-evidence channel -- literature co-occurrence enrichment against the
paralog contrast, as discriminative structured facts rather than raw text dumps.

Motivation (see the frozen master results table): the membership bucket of the anti-memorization
dataset sat at roughly a coin flip for every system (the released pipeline 44.4%, the strongest
baseline 47.2%), and the graph's positive chain is already saturated. This channel hands the
decision layer a genuinely [discriminative] signal: the co-occurrence enrichment of the target
gene with the cancer type in the project literature corpus, R = observed co-occurrence /
independence expectation, compared across paralogues. "every other paralogue in the family is
discussed in this cancer type except you" is [negative evidence]; "only you are discussed" is
positive evidence. Everything returned is an observed count, never a pipeline conclusion (the
_facts_digest principle), and the channel only reads the public literature corpus -- it never
touches the ground truth.

Integration: the supply stage turns the result into one synthetic
KGEvidence(operation="lit_cooc_contrast") appended to kg.evidence, so the Vega evidence channels
(_llm_analogy -> _evidence_brief), the judge/chief review (_facts_digest) and the triad fact sheet
all see it automatically with zero signature changes in the decision layer; the heuristic reads by
operation name, so an unknown operation is naturally ignored.

Failure posture (rather wait than pollute): Mongo unreachable or any exception => return an empty
list, this question simply has no such channel, behaviour falls back byte for byte to the released
configuration, no exception escapes and no row is fabricated.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

_ON = {"1", "true", "on", "yes"}


def neg_evid_enabled() -> bool:
    """Gate: SLC_NEG_EVID (off by default, i.e. the released SLC-MARS configuration is byte-identical
    and does not contain this channel)."""
    return os.environ.get("SLC_NEG_EVID", "0").lower() in _ON


def _cancer_terms(cancer: str) -> list[str]:
    """Cancer-type string -> search terms (lower-cased, denoised, abbreviation and full name; used to match title+abstract across 49628 papers)."""
    toks = [t for t in re.split(r"[^A-Za-z0-9]+", (cancer or "").lower()) if len(t) > 2]
    whole = (cancer or "").strip().lower()
    out = list(dict.fromkeys([w for w in [whole, *toks] if w and w not in {"human", "mouse", "cell", "line", "days", "ko"}]))
    return out


@lru_cache(maxsize=256)
def _gene_papers(gene: str) -> tuple[tuple[str, str], ...]:
    """(title, abstract) pool matched by the gene (the same coarse regex recall as mongo_rag, limit 400)."""
    from pymongo import MongoClient
    uri = os.environ.get("MONGO_URI", "")
    if not uri:
        raise RuntimeError("MONGO_URI is not set")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    col = client[os.environ.get("MONGO_DB_NAME", "SLCdb")][
        os.environ.get("MONGO_COLLECTION_PAPERS", "papers")]
    rx = re.compile(r"\b" + re.escape(gene) + r"\b", re.I)
    cur = col.find({"$or": [{"title": rx}, {"abstract": rx}]},
                   {"title": 1, "abstract": 1}, limit=400)
    return tuple((_flat(d.get("title")), _flat(d.get("abstract"))) for d in cur)


@lru_cache(maxsize=64)
def _cancer_count(terms_key: str) -> int:
    """Background literature volume for the cancer type (regex count over the whole corpus, cached per
    cancer type; terms_key = "|".join(terms))."""
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
    """A paper's title/abstract may arrive as an XML-ingest dict ({'#text':..}) -> flatten it to plain
    text, exactly as mongo_rag does."""
    if isinstance(v, dict):
        return str(v.get("#text") or v.get("text") or " ".join(str(x) for x in v.values())).strip()
    return str(v or "").strip()


def _contrast(gene: str, terms: list[str]) -> dict[str, Any] | None:
    """Co-occurrence enrichment observation for one gene x cancer type: R = observed co-occurrence /
    independence expectation (>=2 enriched, <=0.5 anti-associated)."""
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
    """Discriminative observation rows for a membership question (the target gene plus at most two
    paralogue controls). Any exception => [] (silent degradation).

    Paralogue source: the keys of ck.paralog_redundancy (the graph's orthologous family), which give
    the "cancer specificity within the family" contrast. A caveat row is appended so the decision
    layer cannot over-read missing co-occurrence as "not essential" -- literature bias is real."""
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
    except Exception as e:  # noqa BLE001 -- infrastructure noise must never pollute the verdict
        import sys
        print(f"[negative_evidence] degraded to the released behaviour: {type(e).__name__}: {str(e)[:120]}",
              file=sys.stderr, flush=True)
        return []
