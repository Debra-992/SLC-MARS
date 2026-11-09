# -*- coding: utf-8 -*-
"""Baseline 3: AutoGen (pyautogen 0.2.x) -- AssistantAgent + UserProxyAgent two-agent chat answering.

Standalone venv: outputs\\autogen-venv (Python 3.11 + pyautogen), keeping the main.venv clean.
Question set / prompt / endpoint / answer schema are identical to pure_llm; the only difference is that answering
goes through the AutoGen multi-agent chat framework (assistant emits the answer + TERMINATE, proxy echoes once,

Usage:
  .\\outputs\\autogen-venv\\Scripts\\python.exe knowledge_agents\\baselines\\run_autogen.py [--limit N]
the assistant's final JSON is taken). Resumption matches pure_llm (error rows are retried).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import autogen  # noqa: E402

from knowledge_agents.baselines.common import (  # noqa: E402
    append_answer, build_prompt, extract_json, install_openai_token_tracking, llm_settings,
    load_inputs, normalize, pending, TokenSink,
)

SYSTEM = "autogen"
TIMEOUT = float(os.environ.get("SLC_BASELINE_LLM_TIMEOUT", "120"))
_lock = threading.Lock()  # serialize the synchronous chat, avoiding 0.2 global-state concurrency issues
_sink = TokenSink()  # token metering (accumulated after patching openai.create)


def make_agents(cfg: dict):
    llm_config = {
        "config_list": [{"model": cfg["model"], "api_key": cfg["api_key"], "base_url": cfg["base_url"]}],
        "timeout": 120,
        "cache_seed": None,  # disable autogen's disk cache: otherwise a repeated prompt hits the cache -> 0 real calls -> tokens undercounted,
                              # and that disagrees with pure_llm/lit_rag, which recompute every time (the cache survives clearing the answers)
    }
    assistant = autogen.AssistantAgent(
        name="assistant", llm_config=llm_config,
        system_message="You are a helpful biomedical assistant. "
                       "After answering, output TERMINATE on a line of its own.",
    )
    proxy = autogen.UserProxyAgent(name="proxy", human_input_mode="NEVER",
                                   code_execution_config=False)
    return assistant, proxy


def one_chat(assistant, proxy, prompt: str) -> tuple[str, str]:
    """Synchronous two-turn dialogue; returns (assistant final text, error).

    In 0.2.x the tail of the chat usually holds an assistant message that is only TERMINATE/an echo, with the JSON earlier:
    scan backwards and take the most recent assistant message that parses as JSON."""
    with _lock:
        try:
            proxy.initiate_chat(assistant, message=prompt, clear_history=True, max_turns=2)
            hist = assistant.chat_messages[proxy]
            best = ""
            for m in reversed(hist):
                if m.get("role") != "assistant":
                    continue
                content = str(m.get("content") or "")
                if extract_json(content):
                    best = content
                    break
            if not best:  # nothing parses as JSON: fall back to the last assistant text (the caller records no_json)
                best = next((str(m.get("content") or "") for m in reversed(hist)
                             if m.get("role") == "assistant"), "")
            return best, ""
        except Exception as e:  # noqa BLE001 -- honest failure -> abstain
            return "", f"{type(e).__name__}: {e}"[:200]


def _chat_task(assistant, proxy, prompt):
    """Runs inside a pool worker: reset -> dialogue -> snapshot, attributing token usage to a single question."""
    _sink.reset()
    text, err = one_chat(assistant, proxy, prompt)
    return text, err, _sink.snapshot()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--system", default=SYSTEM)
    args = ap.parse_args()

    cfg = llm_settings()
    if not install_openai_token_tracking(_sink):
        print("[autogen][WARN] openai token interception not installed (library layout differs); tokens will count as 0", flush=True)
    todo = pending(args.system, load_inputs())
    if args.limit:
        todo = todo[:args.limit]
    print(f"[{args.system}] to run {len(todo)} | model={cfg['model']} | pyautogen 2-agent chat", flush=True)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    for i, rec in enumerate(todo, 1):
        assistant, proxy = make_agents(cfg)
        prompt = build_prompt(rec)
        t0 = time.time()
        fut = pool.submit(_chat_task, assistant, proxy, prompt)
        try:
            text, err, tok = fut.result(timeout=TIMEOUT)
        except concurrent.futures.TimeoutError:
            text, err, tok = "", "TimeoutError: chat exceeded deadline", _sink.snapshot()
        data = extract_json(text) if text else None
        row = normalize(rec, data, system=args.system,
                        error=err if err else ("" if data else "no_json_parse"))
        row["elapsed_s"] = round(time.time() - t0, 1)
        row["tokens"] = tok
        append_answer(args.system, row)
        print(f"  [{i:>2}/{len(todo)}] {row['qid'][:42]:<44} win={row['winner']} "
              f"ans={row['answer']} abst={row['abstained']} tok={tok['total']} calls={tok['calls']} {row['elapsed_s']}s {err[:50]}", flush=True)
    pool.shutdown(wait=False)
    print(f"[{args.system}] done -> outputs/traces/answers_{args.system}.jsonl", flush=True)


if __name__ == "__main__":
    main()
