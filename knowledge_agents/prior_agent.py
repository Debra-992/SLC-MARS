# -*- coding: utf-8 -*-
"""Prior agent: promotes the "pure LLM memory prior" into a third parallel evidence channel.

It runs alongside Kira (graph) || Cael (statistics) in the same asyncio.gather and produces a prior
answer {winner|answer, confidence, rationale} on exactly the same question text, endpoint and
prompt as the baseline, which is handed to Vega:
  * when the evidence channels abstain (graph/statistics insufficient) -> the prior fills the
    coverage gap as a fallback, with its provenance labelled honestly;
  * when the evidence channels and the prior conflict -> debate arbitration is triggered (debate.py);
    no conflict means no extra cost.
Environment gates (all off by default, i.e. byte-identical to running without a prior):
  SLC_PRIOR_AGENT=1 enables the channel; SLC_PRIOR_TIMEOUT is the per-call timeout (90s default).
Honesty invariant: a failed prior call or parse yields None -- it never blocks the main evidence
chain and never fabricates a prior.
"""
from __future__ import annotations

import asyncio
import os

from knowledge_agents.baselines.common import BASELINE_PROMPT, extract_json, llm_settings

_ON = {"1", "true", "on", "yes"}


def prior_enabled() -> bool:
    return os.environ.get("SLC_PRIOR_AGENT", "0").lower() in _ON


def prior_retrieve_enabled() -> bool:
    """Gate for decorrelating the second vote: SLC_PRIOR_RETRIEVE hangs a light retrieved-literature
    pack on it. Off by default, i.e. byte-identical to the released configuration."""
    return os.environ.get("SLC_PRIOR_RETRIEVE", "0").lower() in _ON


def build_retrieval_pack(question: str, bucket: str, candidates: list[str],
                         constraints: dict | None = None,
                         top_k: int = 3, max_chars: int = 1200) -> str:
    """Compact [title + excerpt] literature pack recalled by keyword from the local Mongo papers
    collection (the same recall engine as mongo_rag).
    The lit-rag lesson was that stacking 5 papers x 900 characters out of a 4000-character budget
    cost accuracy (46% instead of 57%), so this is deliberately tight: 3 papers, at most 300
    characters of abstract each, at most 1200 characters in total. Any failure (no MONGO_URI, no
    intranet, zero hits, exception) returns "" and the vote degrades to a bare memory vote, and no
    exception is ever raised -- infrastructure noise must not pollute the verdict. Reads only the
    question, the candidates and the constraints; never the ground truth."""
    try:
        from knowledge_agents.baselines import mongo_rag
        papers = [p for p in mongo_rag.retrieve(question,
                  [c.upper() for c in (candidates or [])], constraints or {}, top_k=top_k)
                  if int(p.get("score", 0)) > 0]
        blocks, used = [], 0
        for i, p in enumerate(papers, 1):
            blk = f"[{i}] {p['title'][:160]} - {p['abstract'][:300]}"
            if used + len(blk) > max_chars:
                break
            blocks.append(blk)
            used += len(blk)
        return "\n".join(blocks)
    except Exception:  # noqa BLE001 -- degrade silently but leave an observable trace (silent-failure rule: it has to be findable)
        import sys
        print("[prior_agent] retrieval pack build failed -> the L2 vote degrades to a bare memory vote", file=sys.stderr, flush=True)
        return ""


# Role-heterogeneous personas: one and the same model, different generation distributions (the real
# source of mdagents' single-model diversity)
ROLE_PERSONAS = {
    "L1": ("You are a literature-memory specialist in cancer biology. Answer from what you "
           "recall about these genes, their known phenotypes, and published CRISPR screens."),
    "L2": ("You are a quantitative cancer geneticist. Do NOT rely on recalling specific papers; "
           "reason from first principles: essentiality logic (pathway bottleneck vs redundancy), "
           "gene dosage and buffering, screening conventions (day-18 amplicon depletion = "
           "essential), and mechanistic plausibility in the stated cell context."),
}


def prior_race_n() -> int:
    """Number of racing replicas of the prior call: SLC_PRIOR_RACE (default 1 = a single shot)."""
    try:
        return max(1, int(os.environ.get("SLC_PRIOR_RACE", "1")))
    except (TypeError, ValueError):
        return 1


async def ask_prior(question: str, bucket: str, candidates: list[str]) -> dict | None:
    """Bare LLM memory prior, on exactly the same question text and parameters as the pure-llm
    baseline. Returns None on failure."""
    return await ask_prior_augmented(question, bucket, candidates, digest="")


def prior_ensemble_n() -> int:
    """Gate for the number of draws in the ensemble prior: SLC_PRIOR_ENSEMBLE (default 1 = a single
    draw = byte-identical to the one-shot prior)."""
    try:
        return max(1, int(os.environ.get("SLC_PRIOR_ENSEMBLE", "1")))
    except (TypeError, ValueError):
        return 1


def _spread_temps(n: int) -> list[float]:
    """Diversity by raising the temperature: spread n draws over the 0.5-1.0 temperature band (the
    variance source of self-consistency). n<=1 returns the baseline 0.2, so degrading to a single
    prior leaves the question text and parameters byte-identical."""
    if n <= 1:
        return [0.2]
    return [round(0.5 + 0.5 * i / (n - 1), 2) for i in range(n)]


def _aggregate(bucket: str, draws: list, n_draws: int) -> dict:
    """Aggregate several independent memory draws into one prior dictionary carrying consensus and
    dispersion (the same shape as ask_prior plus votes/n_valid/n_draws/agreement). Abstentions,
    out-of-range votes and failed calls are not counted as valid votes.
    Ties are broken by the sum of the members' confidences. Honest: with zero valid votes this
    returns abstained=True and never invents a verdict."""
    votes = [d for d in draws
             if isinstance(d, dict) and not d.get("abstained")
             and (d.get("winner") if bucket == "pairwise" else d.get("answer")) is not None]
    n_abstain = sum(1 for d in draws if isinstance(d, dict) and d.get("abstained"))
    n_err = sum(1 for d in draws if not isinstance(d, dict))
    n_valid = len(votes)
    if n_valid == 0:
        return {"winner": None, "answer": None, "confidence": 0.0, "abstained": True,
                "votes": {}, "n_valid": 0, "n_draws": n_draws, "agreement": 0.0,
                "rationale": f"ensemble prior has no valid vote (valid=0/{n_draws}, abstained={n_abstain}, failed={n_err})"}
    if bucket == "pairwise":
        pick = lambda v: str(v["winner"]).upper()  # noqa: E731
    else:
        pick = lambda v: bool(v["answer"])  # noqa: E731
    groups: dict = {}
    for v in votes:
        groups.setdefault(pick(v), []).append(v)
    best_key, best_grp = max(
        groups.items(),
        key=lambda kv: (len(kv[1]), sum(x.get("confidence", 0.0) for x in kv[1])))
    agreement = len(best_grp) / n_valid
    mean_conf = sum(x.get("confidence", 0.0) for x in best_grp) / len(best_grp)
    out = {"winner": None, "answer": None, "confidence": round(mean_conf, 3),
           "abstained": False, "votes": {k: len(g) for k, g in groups.items()},
           "n_valid": n_valid, "n_draws": n_draws, "agreement": round(agreement, 3),
           "rationale": (f"ensemble prior n={n_draws} (valid={n_valid}, abstained={n_abstain}, failed={n_err}), "
                         f"consensus={best_key} agreement={agreement:.2f}; "
                         f"{str(best_grp[0].get('rationale', ''))[:200]}")}
    if bucket == "pairwise":
        out["winner"] = best_key
    else:
        out["answer"] = best_key
    return out


async def ask_prior_ensemble(question: str, bucket: str, candidates: list[str],
                             n: int | None = None) -> dict | None:
    """Run the memory-prior channel n times in parallel (diversified by temperature) and aggregate
    the mode into a prior that carries its consensus, then hand it to Vega.
    n<=1 degrades straight to ask_prior (the one-shot prior, byte-identical). This attacks the
    variance weakness of a single pure-llm draw."""
    n = prior_ensemble_n() if n is None else n
    if n <= 1:
        return await ask_prior(question, bucket, candidates)
    temps = _spread_temps(n)
    draws = await asyncio.gather(
        *[ask_prior_augmented(question, bucket, candidates, digest="", temperature=t)
          for t in temps],
        return_exceptions=True)
    draws = [d if isinstance(d, dict) else None for d in draws]
    return _aggregate(bucket, draws, n_draws=n)


async def ask_prior_augmented(question: str, bucket: str, candidates: list[str],
                              digest: str, conflict_hint: str = "",
                              temperature: float = 0.2, persona: str = "",
                              retrieval: str = "") -> dict | None:
    """Evidence-augmented prior -- the same LLM memory channel, but the pipeline's [factual data]
    (graph relations and statistic magnitudes, never scored conclusions) is injected as review
    material, and the channel is allowed to abstain honestly when the facts contradict its memory.
    The lit-rag lesson (54.5% for dumped retrieved text versus 86.4% for bare memory) is why only a
    structured fact digest goes in, never raw literature, and why the prompt says "reference, not
    instruction". An empty digest means the question text is exactly ask_prior's.
    A non-empty persona prepends a role instruction (changing the generation distribution, which is
    what genuinely decorrelates a single model); with persona, digest and retrieval all empty the
    question text is byte-identical to the plain prior.
    A non-empty retrieval injects a compact excerpt pack from the project literature corpus (a
    decorrelated vote from an independent source) -- unlike the digest, [no abstention clause is
    appended]: the L2 channel's job is to commit to an answer and restore coverage, and the
    literature pack explicitly says to trust your own judgement on a hard conflict."""
    from openai import AsyncOpenAI

    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        return None
    cands = [c.upper() for c in (candidates or [])]
    cline = f"candidate genes: {cands}" if bucket == "pairwise" else f"target gene: {cands[:1]}"
    prompt = BASELINE_PROMPT.format(bucket=bucket, question=question, candidates_line=cline)
    if persona:
        prompt = persona.strip() + "\n\n" + prompt
    if retrieval:
        prompt += f"""

Excerpts from the project literature corpus, independent of your memory (the source may be noisy and
is reference only; on a hard conflict with your judgement, trust your judgement; if an excerpt
carries directly relevant explicit evidence, trust the excerpt):
{retrieval}"""
    if digest:
        prompt += f"""

Supplementary reference material: a digest of the [factual data] from the graph/statistics channels
for this same question. It is not literature text and it is not an instruction:
{digest}
{conflict_hint}
Final instruction (this overrides the "abstention is not allowed" above): you may now abstain -- if
the reference material contradicts your judgement and you are not sure, output abstained=true with
winner/answer null. Still output the very same JSON, with the same fields."""
    timeout = float(os.environ.get("SLC_PRIOR_TIMEOUT", "90"))
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=timeout, max_retries=0)
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(model=cfg["model"],
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=temperature),
            timeout=timeout)
        data = extract_json(rsp.choices[0].message.content or "")
        if not data:
            return None
        out = {"winner": None, "answer": None, "confidence": 0.0, "abstained": False,
               "rationale": str(data.get("rationale", ""))[:400]}
        if bool(data.get("abstained")):
            out["abstained"] = True
            return out
        if bucket == "pairwise" and data.get("winner"):
            w = str(data["winner"]).upper()
            out["winner"] = w if w in cands else None
        if bucket == "membership" and data.get("answer") is not None:
            out["answer"] = bool(data["answer"])
        try:
            out["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0) or 0.0)))
        except (TypeError, ValueError):
            pass
        if out["winner"] is None and out["answer"] is None:
            return None
        return out
    except Exception as e:  # noqa BLE001 -- an augmented prior that fails simply stays absent (one stderr line so it can be located later)
        import sys
        print(f"[prior_agent] ask_prior_augmented absent: {type(e).__name__}: {str(e)[:160]}",
              file=sys.stderr, flush=True)
        return None
    finally:
        try:
            await client.close()
        except Exception:  # noqa BLE001
            pass


async def adjudicate_multi(question: str, bucket: str, candidates: list[str],
                           evidence: str, views: list, tally: str = "") -> dict | None:
    """Three-channel vote: the final agent sees the [factual review material] and the arguments of
    several channels (the evidence channel plus two zero-evidence pure-LLM channels sampled
    independently), weighs the reasoning of each side first, and then votes a single verdict.
    views = [(channel name, channel view), ...]. Failure or abstention returns None: no verdict is
    forced, the absence is reported honestly."""
    from openai import AsyncOpenAI

    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        return None
    cands = [c.upper() for c in (candidates or [])]
    task = ("Which candidate gene is more likely the winner? " if bucket == "pairwise"
            else "Is the claim true or false? ")
    chan_block = "\n\n".join(f"Channel {name}:\n{view}" for name, view in views)
    prompt = f"""You are the FINAL AGENT resolving a CRISPR essentiality question (bucket={bucket}) by DEBATING several independent channels and then VOTING one verdict.

Question: {question}
Candidates: {cands}

Factual observations (graph/statistics computed from databases; observations only, not conclusions):
{evidence}

Independent channels and their arguments:
{chan_block}

Current vote tally: {tally or 'none'}

{task}Debate internally: weigh each channel's argument against the factual observations and against the other channels (database evidence is association-level, not a CRISPR readout; a zero-evidence LLM vote may be mistaken or outdated, but two independent LLM votes agreeing is a real signal). Then cast your single verdict. If nothing is convincing, abstain.

Output JSON only: {{"winner": "<candidate or null>", "answer": true/false/null, "confidence": 0~1, "abstained": false, "rationale": "<=50 words"}}"""
    timeout = float(os.environ.get("SLC_JUDGE_TIMEOUT", "120"))
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=timeout, max_retries=0)
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(model=cfg["model"],
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0.2),
            timeout=timeout)
        data = extract_json(rsp.choices[0].message.content or "")
        if not data or bool(data.get("abstained")):
            return None
        out = {"winner": None, "answer": None, "confidence": 0.0,
               "rationale": str(data.get("rationale", ""))[:400]}
        if bucket == "pairwise" and data.get("winner"):
            w = str(data["winner"]).upper()
            out["winner"] = w if w in cands else None
        if bucket == "membership" and data.get("answer") is not None:
            out["answer"] = bool(data["answer"])
        try:
            out["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0) or 0.0)))
        except (TypeError, ValueError):
            pass
        if out["winner"] is None and out["answer"] is None:
            return None
        return out
    except Exception as e:  # noqa BLE001
        import sys
        print(f"[prior_agent] judge absent: {type(e).__name__}: {str(e)[:160]}",
              file=sys.stderr, flush=True)
        return None
    finally:
        try:
            await client.close()
        except Exception:  # noqa BLE001
            pass


async def ask_prior_raced(question: str, bucket: str, candidates: list[str], *,
                          persona: str = "", temperature: float = 0.2,
                          n: int | None = None, retrieval: str = "") -> dict | None:
    """Racing: fire n copies of the same request in parallel and take the first valid answer (this
    absorbs long-tail endpoint timeouts). n<=1 is a single shot, byte-identical to
    ask_prior_augmented. If all copies fail, the first non-None result is returned (possibly an
    abstention, which passes the absence signal on honestly); surplus tasks are cancelled so no
    orphan call is left behind. The retrieval pack is passed through to every copy."""
    n = prior_race_n() if n is None else n
    if n <= 1:
        return await ask_prior_augmented(question, bucket, candidates, digest="",
                                         conflict_hint="", temperature=temperature,
                                         persona=persona, retrieval=retrieval)
    pend = {asyncio.ensure_future(ask_prior_augmented(question, bucket, candidates,
                                                      digest="", conflict_hint="",
                                                      temperature=temperature,
                                                      persona=persona,
                                                      retrieval=retrieval))
            for _ in range(n)}
    fallback: dict | None = None
    try:
        while pend:
            done, pend = await asyncio.wait(pend, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                try:
                    r = t.result()
                except Exception:  # noqa BLE001
                    r = None
                if isinstance(r, dict) and not r.get("abstained"):
                    return r
                if isinstance(r, dict) and fallback is None:
                    fallback = r
        return fallback
    finally:
        for t in pend:
            t.cancel()


async def critique_prior(question: str, bucket: str, candidates: list[str],
                         persona: str, own_view: str, others_view: str) -> dict | None:
    """Cross-critique: each channel keeps its own role persona, first attacks the weakest argument
    (the other side's or its own), then issues a revised final vote -- it switches only if the other
    side's reasoning genuinely defeats its own. Output has the same shape as ask_prior; failure
    returns None and the caller keeps that channel's original vote rather than forcing a change."""
    from openai import AsyncOpenAI

    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        return None
    cands = [c.upper() for c in (candidates or [])]
    prompt = f"""{persona.strip()}

You are in a reasoned disagreement about a CRISPR essentiality question (bucket={bucket}).

Question: {question}
Candidates: {cands}

Your channel's earlier answer: {own_view}
Other channels' answers and arguments:
{others_view}

Step 1: briefly attack the weakest argument (theirs or your own). Step 2: give your REVISED final verdict - keep or change your earlier answer, but only switch if the other side's reasoning genuinely defeats yours.

Output JSON only: {{"winner": "<candidate or null>", "answer": true/false/null, "confidence": 0~1, "abstained": false, "rationale": "<=40 words"}}"""
    timeout = float(os.environ.get("SLC_CRITIQUE_TIMEOUT", "180"))
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=timeout, max_retries=0)
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(model=cfg["model"],
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0.4),
            timeout=timeout)
        data = extract_json(rsp.choices[0].message.content or "")
        if not data or bool(data.get("abstained")):
            return None
        out = {"winner": None, "answer": None, "confidence": 0.0,
               "rationale": str(data.get("rationale", ""))[:400]}
        if bucket == "pairwise" and data.get("winner"):
            w = str(data["winner"]).upper()
            out["winner"] = w if w in cands else None
        if bucket == "membership" and data.get("answer") is not None:
            out["answer"] = bool(data["answer"])
        try:
            out["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0) or 0.0)))
        except (TypeError, ValueError):
            pass
        if out["winner"] is None and out["answer"] is None:
            return None
        return out
    except Exception as e:  # noqa BLE001
        import sys
        print(f"[prior_agent] critique absent: {type(e).__name__}: {str(e)[:160]}",
              file=sys.stderr, flush=True)
        return None
    finally:
        try:
            await client.close()
        except Exception:  # noqa BLE001
            pass


async def adjudicate(question: str, bucket: str, candidates: list[str],
                     evidence: str, prior_view: str) -> dict | None:
    """Conflict arbitration: a single LLM judge sees both the [evidence facts] and the [prior
    argument] and produces the final verdict. Unlike the debate layer there is only one judge, it
    reads the full facts, and abstention is allowed. Failure returns None -- no verdict is forced."""
    from openai import AsyncOpenAI

    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        return None
    cands = [c.upper() for c in (candidates or [])]
    task = ("Which candidate gene is more likely the winner? " if bucket == "pairwise"
            else "Is the claim true or false? ")
    prompt = f"""You must resolve a disagreement between two analysis channels about a CRISPR essentiality question (bucket={bucket}).

Question: {question}
Candidates: {cands}

Channel A (graph/statistical evidence computed from databases):
{evidence}

Channel B (LLM literature-memory prior):
{prior_view}

{task}Weigh the reliability of each channel (database evidence is association-level, not a CRISPR readout; memory prior may be mistaken or outdated). If neither is convincing, abstain.

Output JSON only: {{"winner": "<candidate or null>", "answer": true/false/null, "confidence": 0~1, "abstained": false, "rationale": "<=40 words"}}"""
    timeout = float(os.environ.get("SLC_JUDGE_TIMEOUT", "120"))
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=timeout, max_retries=0)
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(model=cfg["model"],
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0.2),
            timeout=timeout)
        data = extract_json(rsp.choices[0].message.content or "")
        if not data or bool(data.get("abstained")):
            return None
        out = {"winner": None, "answer": None, "confidence": 0.0,
               "rationale": str(data.get("rationale", ""))[:400]}
        if bucket == "pairwise" and data.get("winner"):
            w = str(data["winner"]).upper()
            out["winner"] = w if w in cands else None
        if bucket == "membership" and data.get("answer") is not None:
            out["answer"] = bool(data["answer"])
        try:
            out["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0) or 0.0)))
        except (TypeError, ValueError):
            pass
        if out["winner"] is None and out["answer"] is None:
            return None
        return out
    except Exception as e:  # noqa BLE001
        import sys
        print(f"[prior_agent] judge absent: {type(e).__name__}: {str(e)[:160]}",
              file=sys.stderr, flush=True)
        return None
    finally:
        try:
            await client.close()
        except Exception:  # noqa BLE001
            pass
