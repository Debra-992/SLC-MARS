# -*- coding: utf-8 -*-
"""Baseline 1: pure LLM QA (no MetaGPT framework, direct openai-compatible endpoint, single-shot question answering).

Difference from the stock MetaGPT baseline = only the Role/Action shell is removed; instruction, endpoint and question set
match exactly, which separates framework overhead from raw model ability.

Usage:
  .\\.venv\\Scripts\\python.exe -m knowledge_agents.baselines.run_pure_llm [--limit N] [--system pure_llm]
Resumable: qids already present in answers_<system>.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openai import AsyncOpenAI  # noqa: E402

from knowledge_agents.baselines.common import (  # noqa: E402
    append_answer, build_prompt, extract_json, llm_settings, load_inputs, normalize, pending,
    usage_triplet,
)

SYSTEM = "pure_llm"
TIMEOUT = float(os.environ.get("SLC_BASELINE_LLM_TIMEOUT", "60"))


async def ask_one(client: AsyncOpenAI, model: str, rec: dict) -> tuple[dict | None, str, dict]:
    prompt = build_prompt(rec)
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2),
            timeout=TIMEOUT)
        text = rsp.choices[0].message.content or ""
        p, c, t = usage_triplet(rsp)
        return extract_json(text), "", {"prompt": p, "completion": c, "total": t, "calls": 1}
    except Exception as e:  # noqa BLE001 -- honest failure recorded as abstain
        return None, f"{type(e).__name__}: {e}"[:200], {"prompt": 0, "completion": 0, "total": 0, "calls": 0}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="run only the first N questions (smoke test)")
    ap.add_argument("--system", default=SYSTEM)
    ap.add_argument("--conc", type=int, default=4, help="number of concurrent LLM calls")
    args = ap.parse_args()

    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        raise SystemExit("llm.base_url/api_key not found in config2.yaml")
    import httpx
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=TIMEOUT, max_retries=0,
                         http_client=httpx.AsyncClient(
                             limits=httpx.Limits(max_connections=args.conc + 2)))
    inputs = load_inputs()
    todo = pending(args.system, inputs)
    if args.limit:
        todo = todo[:args.limit]
    print(f"[{args.system}] question set {len(inputs)} | to run {len(todo)} | concurrency {args.conc} | model={cfg['model']}",
          flush=True)
    sem = asyncio.Semaphore(args.conc)
    n_done = 0

    async def worker(rec: dict) -> None:
        nonlocal n_done
        async with sem:
            t0 = time.time()
            data, err, tok = await ask_one(client, cfg["model"], rec)
        row = normalize(rec, data, system=args.system,
                        error=err if err else ("" if data else "no_json_parse"))
        row["elapsed_s"] = round(time.time() - t0, 1)
        row["tokens"] = tok
        append_answer(args.system, row)
        n_done += 1
        print(f"  [{n_done:>2}/{len(todo)}] {row['qid'][:44]:<46} "
              f"win={row['winner']} ans={row['answer']} abst={row['abstained']} tok={tok['total']} {row['elapsed_s']}s {err[:40]}",
              flush=True)

    await asyncio.gather(*(worker(r) for r in todo))
    print(f"[{args.system}] done -> outputs/traces/answers_{args.system}.jsonl", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
