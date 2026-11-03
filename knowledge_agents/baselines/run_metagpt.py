# -*- coding: utf-8 -*-
"""Baseline 5：原生 MetaGPT（stock Role/Action，纯 LLM 一次 _aask，无工具/检索/CRISPR）。

与 run_pure_llm 的差别 = 保留 MetaGPT 的 Role/Action 框架壳，指令/端点/题集完全一致
（都用 config2.yaml qwen 端点、同 _BASELINE_PROMPT、同 baselines_input.json 489 题）。
用来回答"框架本身（不经本项目特化工具）值多少"。

token 计量：不 patch openai 流式（MetaGPT provider 走 AsyncOpenAI + stream=True），
直接读 MetaGPT 原生 Context.cost_manager 的 token 增量（单题一次 _aask，串行归因干净）。

用法：
  .\\.venv\\Scripts\\python.exe -m knowledge_agents.baselines.run_metagpt [--limit N]
断点续跑同其它 baseline：answers_metagpt.jsonl 里已成功 qid 自动跳过。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# 禁 MetaGPT 用量上报（否则可能尝试外网 POST 拖慢/挂起）；须在 import metagpt 前置
os.environ.setdefault("METAGPT_REPORTABLE", "false")
os.environ.setdefault("REDIS_HOST", "")
os.environ.setdefault("SLC_BASELINE_LLM_TIMEOUT", "120")  # BaselineAnswer._aask 超时(拥塞洗瞬时)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from knowledge_agents.baselines.common import (  # noqa: E402
    append_answer, install_metagpt_token_tracking, llm_settings, load_inputs, normalize,
    pending, TokenSink,
)

SYSTEM = "metagpt"
_sink = TokenSink()   # 由 install_metagpt_token_tracking 在 CostManager.update_cost 处累加


async def one(ctx, rec: dict) -> tuple[dict | None, str, dict]:
    """跑一题原生 MetaGPT：fresh BaselineRole(共享 ctx) → parse → sink 快照取 token。"""
    from knowledge_agents.baseline_metagpt import BaselineRole
    from knowledge_agents.schema import parse_knowledge
    role = BaselineRole(context=ctx)
    role.set_inputs(bucket=rec["bucket"], candidates=rec.get("candidates") or [],
                    question=rec["question"], task_type=rec.get("task_type", ""))
    _sink.reset()
    try:
        await role.run("answer")
        ans = parse_knowledge(role.rc.memory.get()[-1].content)
    except Exception as e:  # noqa BLE001 —— 失败诚实 abstain
        return None, f"{type(e).__name__}: {e}"[:200], _sink.snapshot()
    tok = _sink.snapshot()
    if not getattr(ans, "used_llm", False):       # 端点不可达/超时：原生记 abstain
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
    ctx = Context()                              # 读 ~/.metagpt/config2.yaml（框架只认文件）
    # 凭据以环境为准：用 SLC_LLM_* 覆盖 home 配置，这样跑这条 baseline 不需要
    # 把 api_key 抄进 ~/.metagpt/config2.yaml（home 文件缺失或写占位符都不影响）。
    _llm = llm_settings()
    if _llm["api_key"] and _llm["base_url"]:
        _c = ctx.config.llm
        _c.api_key, _c.base_url = _llm["api_key"], _llm["base_url"]
        if _llm["model"]:
            _c.model = _llm["model"]
        print(f"[{SYSTEM}] endpoint={_c.base_url[:48]} model={_c.model} "
              f"(creds from SLC_LLM_* env)", flush=True)
    else:
        print(f"[{SYSTEM}][WARN] SLC_LLM_* 未配置，回退 ~/.metagpt/config2.yaml 的凭据",
              flush=True)
    # 强制非流式：aask 走 _achat_completion → _update_costs(rsp.usage) 拿真实 usage（与
    # 其它 baseline 同口径）。流式时 qwen 自定义模型无 usage chunk 且 tiktoken 不识别→记 0。
    for holder in (getattr(ctx.config, "llm", None), ctx.config):
        if holder is not None and hasattr(holder, "stream"):
            try:
                holder.stream = False
            except Exception:  # noqa BLE001
                pass
    if not install_metagpt_token_tracking(_sink):
        print(f"[{args.system}][WARN] MetaGPT token 拦截未安装，tokens 将记 0", flush=True)
    inputs = load_inputs()
    todo = pending(args.system, inputs)
    if args.limit:
        todo = todo[:args.limit]
    print(f"[{args.system}] 题集 {len(inputs)} | 待跑 {len(todo)} | 原生 MetaGPT Role/Action(单 _aask)",
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
    print(f"[{args.system}] 完成 -> outputs/traces/answers_{args.system}.jsonl", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
