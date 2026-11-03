# -*- coding: utf-8 -*-
"""Baseline 2：文献 RAG —— 检索 MongoDB papers 文献 -> 摘录拼进 prompt -> 同一 LLM 作答。

与 pure_llm 唯一差别 = 多一段检索到的文献知识；模型/指令格式/题集一致。
需要 MONGO_URI 环境变量（run_route_b.ps1 已导出）。

用法：
  $env:MONGO_URI=...; .\\.venv\\Scripts\\python.exe -m knowledge_agents.baselines.run_lit_rag [--limit N] [--conc 4]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openai import AsyncOpenAI  # noqa: E402

from knowledge_agents.baselines import mongo_rag  # noqa: E402
from knowledge_agents.baselines.common import (  # noqa: E402
    BASELINE_PROMPT, append_answer, extract_json, llm_settings, load_inputs, normalize, pending,
    usage_triplet,
)

SYSTEM = "lit_rag"
TIMEOUT = float(os.environ.get("SLC_BASELINE_LLM_TIMEOUT", "90"))
RAG_PROMPT = BASELINE_PROMPT.replace(
    "仅凭你自己的知识回答，不使用任何工具、检索或外部数据。",
    "结合文末提供的文献摘录作为知识补充来回答；文献没有提及不等于不存在。")


def build_rag_prompt(rec: dict) -> tuple[str, int]:
    papers = mongo_rag.retrieve(rec["question"], rec.get("candidates") or [],
                               rec.get("constraints") or {})
    ctx = mongo_rag.format_context(papers)
    cands = rec.get("candidates") or []
    cline = f"候选基因: {cands}" if rec["bucket"] == "pairwise" else f"目标基因: {cands[:1]}"
    body = RAG_PROMPT.format(bucket=rec["bucket"], question=rec["question"], candidates_line=cline)
    return ctx + "\n\n" + body, len(papers)


_NOTOK = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}


async def ask_one(client: AsyncOpenAI, model: str, rec: dict) -> tuple[dict | None, str, int, dict]:
    try:
        prompt, n_pap = await asyncio.to_thread(build_rag_prompt, rec)
    except Exception as e:  # noqa BLE001 —— 检索失败即如实 abstain，不静默退化成纯LLM
        return None, f"retrieval {type(e).__name__}: {e}"[:200], -1, _NOTOK
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2),
            timeout=TIMEOUT)
        p, c, t = usage_triplet(rsp)
        return (extract_json(rsp.choices[0].message.content or ""), "", n_pap,
                {"prompt": p, "completion": c, "total": t, "calls": 1})
    except Exception as e:  # noqa BLE001
        return None, f"{type(e).__name__}: {e}"[:200], n_pap, _NOTOK


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--system", default=SYSTEM)
    ap.add_argument("--conc", type=int, default=4)
    args = ap.parse_args()

    cfg = llm_settings()
    import httpx
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=TIMEOUT, max_retries=0,
                         http_client=httpx.AsyncClient(
                             limits=httpx.Limits(max_connections=args.conc + 2)))
    todo = pending(args.system, load_inputs())
    if args.limit:
        todo = todo[:args.limit]
    print(f"[{args.system}] 待跑 {len(todo)} | model={cfg['model']} | mongo_uri={'set' if os.environ.get('MONGO_URI') else 'MISSING!'}",
          flush=True)
    sem = asyncio.Semaphore(args.conc)
    n_done = 0

    async def worker(rec: dict) -> None:
        nonlocal n_done
        async with sem:
            t0 = time.time()
            data, err, n_pap, tok = await ask_one(client, cfg["model"], rec)
        row = normalize(rec, data, system=args.system,
                        error=err if err else ("" if data else "no_json_parse"))
        row["n_papers"] = n_pap
        row["elapsed_s"] = round(time.time() - t0, 1)
        row["tokens"] = tok
        append_answer(args.system, row)
        n_done += 1
        print(f"  [{n_done:>2}/{len(todo)}] {row['qid'][:42]:<44} win={row['winner']} "
              f"ans={row['answer']} abst={row['abstained']} papers={n_pap} tok={tok['total']} {row['elapsed_s']}s {err[:40]}",
              flush=True)

    await asyncio.gather(*(worker(r) for r in todo))
    print(f"[{args.system}] 完成 -> outputs/traces/answers_{args.system}.jsonl", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
