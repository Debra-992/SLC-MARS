# -*- coding: utf-8 -*-
"""Baseline 1：纯 LLM 问答（不经 MetaGPT 框架，直连 openai 兼容端点，单次问答）。

与原生 MetaGPT baseline 的差别 = 只去掉 Role/Action 框架壳，指令/端点/题集完全一致，
用来分离"框架开销"与"模型本身能力"。

用法：
  .\\.venv\\Scripts\\python.exe -m knowledge_agents.baselines.run_pure_llm [--limit N] [--system pure_llm]
断点续跑：已在 answers_<system>.jsonl 中的 qid 自动跳过。
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
    except Exception as e:  # noqa BLE001 —— 失败诚实记 abstain
        return None, f"{type(e).__name__}: {e}"[:200], {"prompt": 0, "completion": 0, "total": 0, "calls": 0}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--system", default=SYSTEM)
    ap.add_argument("--conc", type=int, default=4, help="并发 LLM 调用数")
    args = ap.parse_args()

    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        raise SystemExit("config2.yaml 未找到 llm.base_url/api_key")
    import httpx
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=TIMEOUT, max_retries=0,
                         http_client=httpx.AsyncClient(
                             limits=httpx.Limits(max_connections=args.conc + 2)))
    inputs = load_inputs()
    todo = pending(args.system, inputs)
    if args.limit:
        todo = todo[:args.limit]
    print(f"[{args.system}] 题集 {len(inputs)} | 待跑 {len(todo)} | 并发 {args.conc} | model={cfg['model']}",
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
    print(f"[{args.system}] 完成 -> outputs/traces/answers_{args.system}.jsonl", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
