# -*- coding: utf-8 -*-
"""Expert debate layer (borrowing MDAgents' multi-expert plus majority-ruling idea as a lightweight in-pipeline version).

Differences from MDAgents, deliberate:
  * only ambiguous questions are arbitrated (the evidence channels disagree with the prior, or coverage needs filling), not every question;
  * three experts vote in parallel, one ballot each (MDAgents runs 5 experts x 5 rounds serially, orders of magnitude costlier);
  * the ruling is a deterministic confidence-weighted majority (no second moderator LLM call, reproducible and auditable).
Key fairness property: the experts see the pipeline's evidence summary plus the prior's opinion and are told to vote
only on the supplied evidence and to abstain when it is insufficient -- the prior is an opinion on trial, no longer unconstrained memory answering.

env: SLC_DEBATE=1 enables it; SLC_DEBATE_TIMEOUT sets the per-expert timeout (90s by default).
Honest failure: any expert failure means a missing ballot; too few valid ballots or too close a split returns None (no forced ruling).
"""
from __future__ import annotations

import asyncio
import os

from knowledge_agents.baselines.common import extract_json, llm_settings

_ON = {"1", "true", "on", "yes"}

EXPERTS = [
    ("gene function and paralogue redundancy expert",
     "You specialize in SLC gene family function, paralogous redundancy and CRISPR loss-of-function essentiality screening logic."),
    ("clinical and translational oncology expert",
     "You specialize in tumour typing, cancer-type-specific dependencies from metabolic reprogramming and reading the clinical evidence on resistance."),
    ("statistical genomics expert",
     "You specialize in reading the magnitude of network proximity (PPR), expression (TIMER/DESeq2), survival (KM) and causal-direction evidence,"
     "evidence strength counts, reputation does not."),
]

# margin required by the confidence-weighted majority: the weighted gap between top1 and top2 must reach this to count as a verdict
DEBATE_MARGIN = float(os.environ.get("SLC_DEBATE_MARGIN", "0.15"))


def debate_enabled() -> bool:
    return os.environ.get("SLC_DEBATE", "0").lower() in _ON


def _vote_prompt(persona: str, question: str, bucket: str, candidates: list[str],
                 digest: str, views: str) -> str:
    task = ("Vote for the most likely winner among the candidates" if bucket == "pairwise"
            else "Vote answer=true/false on this claim")
    fmt = ('{"winner": "<one candidate>", "confidence": 0~1, "abstained": false, "rationale": "one sentence"}'
           if bucket == "pairwise" else
           '{"answer": true/false, "confidence": 0~1, "abstained": false, "rationale": "one sentence"}')
    return f"""{persona}
You sit on a multidisciplinary expert panel and vote independently on this SLC gene CRISPR essentiality question: {task}.

Hard rule: vote only on the [pipeline evidence] and the [opinion on trial]. Do not add facts from memory -- if the evidence is insufficient, vote abstain.
Question (bucket={bucket}) candidates={candidates}
Pipeline evidence (graph and statistics channel scores with the reasoning behind them):
{digest}

Opinion on trial (the pure-LLM memory prior's stance, for critical reference only, not evidence):
{views}

Output JSON only: {fmt}"""


async def _one_expert(cfg: dict, prompt: str) -> dict | None:
    from openai import AsyncOpenAI
    timeout = float(os.environ.get("SLC_DEBATE_TIMEOUT", "90"))
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=timeout, max_retries=0)
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(model=cfg["model"],
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0.3),
            timeout=timeout)
        return extract_json(rsp.choices[0].message.content or "")
    except Exception:  # noqa BLE001 -- an expert failure simply means a missing ballot
        return None
    finally:
        try:
            await client.close()
        except Exception:  # noqa BLE001
            pass


async def run_debate(question: str, bucket: str, candidates: list[str],
                     digest: str, views: str) -> dict | None:
    """Three experts vote in parallel -> deterministic confidence-weighted majority. Returns {winner|answer, support, margin,
    tally, rationale} or None when no valid verdict emerged."""
    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        return None
    cands = [c.upper() for c in (candidates or [])]
    prompts = [_vote_prompt(p, question, bucket, cands, digest, views) for _, p in EXPERTS]
    votes = await asyncio.gather(*(_one_expert(cfg, pr) for pr in prompts))

    # confidence-weighted tally (abstain and out-of-range ballots are not counted)
    tally: dict[str, float] = {}
    n_valid = 0
    for v in votes:
        if not v or v.get("abstained"):
            continue
        try:
            conf = max(0.0, min(1.0, float(v.get("confidence", 0) or 0)))
        except (TypeError, ValueError):
            conf = 0.0
        if bucket == "pairwise":
            w = str(v.get("winner") or "").upper()
            if w not in cands:
                continue
            tally[w] = tally.get(w, 0.0) + conf + 0.2  # +0.2 base vote: heads still count when confidence collapses
        else:
            a = v.get("answer")
            if a is None:
                continue
            key = "true" if bool(a) else "false"
            tally[key] = tally.get(key, 0.0) + conf + 0.2
        n_valid += 1
    if n_valid < 2 or not tally:
        return None
    ranked = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    top_key, top_w = ranked[0]
    margin = top_w - (ranked[1][1] if len(ranked) > 1 else 0.0)
    if margin < DEBATE_MARGIN:
        return None  # too close a split -> no forced ruling
    support = (f"{n_valid}/{len(EXPERTS)} experts voted validly, "
               f"top1 weighted={top_w:.2f} (margin={margin:.2f})")
    out = {"support": support, "margin": round(margin, 3),
           "tally": {k: round(w, 2) for k, w in ranked},
           "rationale": f"[expert debate] {bucket} weighted majority="}
    if bucket == "pairwise":
        out["winner"] = top_key
    else:
        out["answer"] = (top_key == "true")
    out["rationale"] += top_key
    return out
