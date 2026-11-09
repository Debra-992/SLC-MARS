# -*- coding: utf-8 -*-
"""Baseline 5: stock MetaGPT (unmodified Role/Action, one plain LLM _aask, no tools, no retrieval, no CRISPR).

Difference from run_pure_llm = the MetaGPT Role/Action shell is kept; instruction, endpoint and question set match exactly
(same config2.yaml qwen endpoint, same _BASELINE_PROMPT, same 489 questions in baselines_input.json).
This measures what the framework alone is worth, without this project's specialized tools.

Token metering: no patching of openai streaming (MetaGPT's provider uses AsyncOpenAI with stream=True);
instead the delta of MetaGPT's own Context.cost_manager is read (one _aask per question, so serial attribution is clean).

Usage:
  .\\.venv\\Scripts\\python.exe -m knowledge_agents.baselines.run_metagpt [--limit N]
Resumption matches the other baselines: qids already successful in answers_metagpt.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# Disable MetaGPT usage reporting (it may otherwise POST to the network and stall the run); must precede import metagpt
os.environ.setdefault("METAGPT_REPORTABLE", "false")
os.environ.setdefault("REDIS_HOST", "")
os.environ.setdefault("SLC_BASELINE_LLM_TIMEOUT", "120")  # BaselineAnswer._aask timeout (absorbs transient congestion)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from knowledge_agents.baselines.common import (  # noqa: E402
    append_answer, install_metagpt_token_tracking, llm_settings, load_inputs, normalize,
    pending, TokenSink,
)

SYSTEM = "metagpt"
_sink = TokenSink()  # accumulated by install_metagpt_token_tracking at CostManager.update_cost


async def one(ctx, rec: dict) -> tuple[dict | None, str, dict]:
    """Run one question through stock MetaGPT: fresh BaselineRole (shared ctx) -> parse -> token snapshot from the sink."""
    from knowledge_agents.baseline_metagpt import BaselineRole
    from knowledge_agents.schema import parse_knowledge
    role = BaselineRole(context=ctx)
    role.set_inputs(bucket=rec["bucket"], candidates=rec.get("candidates") or [],
                    question=rec["question"], task_type=rec.get("task_type", ""))
    _sink.reset()
    try:
        await role.run("answer")
        ans = parse_knowledge(role.rc.memory.get()[-1].content)
    except Exception as e:  # noqa BLE001 -- honest failure -> abstain
        return None, f"{type(e).__name__}: {e}"[:200], _sink.snapshot()
    tok = _sink.snapshot()
    if not getattr(ans, "used_llm", False):  # endpoint unreachable / timeout: record an honest abstain
        return None, (ans.error or "metagpt_llm_unavailable")[:200], tok
    data = {"winner": ans.winner, "answer": ans.answer,
            "confidence": ans.confidence, "abstained": ans.abstained,
            "rationale": ans.rationale}
    return data, "", tok


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--system", default=SYSTEM)
    args = ap.parse_args()

    from metagpt.context import Context
    ctx = Context()  # reads ~/.metagpt/config2.yaml (the framework only understands files)
    # The environment wins: SLC_LLM_* overrides the home configuration, so running this baseline never needs
    # an api_key copied into ~/.metagpt/config2.yaml (missing home file or placeholder values are fine).
    _llm = llm_settings()
    if _llm["api_key"] and _llm["base_url"]:
        _c = ctx.config.llm
        _c.api_key, _c.base_url = _llm["api_key"], _llm["base_url"]
        if _llm["model"]:
            _c.model = _llm["model"]
        print(f"[{SYSTEM}] endpoint={_c.base_url[:48]} model={_c.model} "
              f"(creds from SLC_LLM_* env)", flush=True)
    else:
        print(f"[{SYSTEM}][WARN] SLC_LLM_* unset; falling back to the credentials in ~/.metagpt/config2.yaml",
              flush=True)
    # Force non-streaming: aask goes through _achat_completion -> _update_costs(rsp.usage), giving real usage (same basis as
    # the other baselines). In stream mode a custom qwen model returns no usage chunk and tiktoken cannot resolve it -> 0.
    for holder in (getattr(ctx.config, "llm", None), ctx.config):
        if holder is not None and hasattr(holder, "stream"):
            try:
                holder.stream = False
            except Exception:  # noqa BLE001
                pass
    if not install_metagpt_token_tracking(_sink):
        print(f"[{args.system}][WARN] MetaGPT token interception not installed; tokens will count as 0", flush=True)
    inputs = load_inputs()
    todo = pending(args.system, inputs)
    if args.limit:
        todo = todo[:args.limit]
    print(f"[{args.system}] question set {len(inputs)} | to run {len(todo)} | stock MetaGPT Role/Action (one _aask)",
          flush=True)

    for i, rec in enumerate(todo, 1):
        t0 = time.time()
        data, err, tok = await one(ctx, rec)
        row = normalize(rec, data, system=args.system,
                        error=err if err else ("" if data else "no_json_parse"))
        row["elapsed_s"] = round(time.time() - t0, 1)
        row["tokens"] = tok or {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
        append_answer(args.system, row)
        print(f"  [{i:>3}/{len(todo)}] {row['qid'][:42]:<44} win={row['winner']} "
              f"ans={row['answer']} abst={row['abstained']} tok={row['tokens']['total']} "
              f"{row['elapsed_s']}s {err[:40]}", flush=True)
    print(f"[{args.system}] done -> outputs/traces/answers_{args.system}.jsonl", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
