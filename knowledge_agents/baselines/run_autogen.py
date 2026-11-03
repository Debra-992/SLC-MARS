# -*- coding: utf-8 -*-
"""Baseline 3：AutoGen（pyautogen 0.2.x）—— AssistantAgent + UserProxyAgent 双人对话答题。

独立 venv：outputs\\autogen-venv（Python 3.11 + pyautogen，不污染主 .venv）。
题集/prompt/端点/答案规范与 pure_llm 完全一致；差别 = 经由 AutoGen 多智能体对话框架作答
（assistant 出答案+TERMINATE，proxy 回声一次，取 assistant 最终 JSON）。

用法：
  .\\outputs\\autogen-venv\\Scripts\\python.exe knowledge_agents\\baselines\\run_autogen.py [--limit N]
断点续跑同 pure_llm（error 行会重试）。
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
_lock = threading.Lock()          # 串行化同步 chat，避免 0.2 全局状态并发问题
_sink = TokenSink()               # token 计量（patch openai.create 后累加）


def make_agents(cfg: dict):
    llm_config = {
        "config_list": [{"model": cfg["model"], "api_key": cfg["api_key"], "base_url": cfg["base_url"]}],
        "timeout": 120,
        "cache_seed": None,   # 关闭 autogen 磁盘缓存：否则同 prompt 命中缓存→0 真实调用→token 漏计，
                              # 且与每次真算的 pure_llm/lit_rag 口径不一致（缓存跨 answers 清空仍残留）
    }
    assistant = autogen.AssistantAgent(
        name="assistant", llm_config=llm_config,
        system_message="You are a helpful biomedical assistant. "
                       "回答完后单独一行输出 TERMINATE。",
    )
    proxy = autogen.UserProxyAgent(name="proxy", human_input_mode="NEVER",
                                   code_execution_config=False)
    return assistant, proxy


def one_chat(assistant, proxy, prompt: str) -> tuple[str, str]:
    """同步双回合对话；返回 (assistant 最终文本, 错误)。

    0.2.x 对话尾部常有一条纯 TERMINATE/回声的 assistant 消息，JSON 在更早一条里：
    从后往前取【最新一条能解析出 JSON】的 assistant 消息。"""
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
            if not best:  # 没有可解析 JSON：回退最后一条 assistant 文本（让上层记 no_json）
                best = next((str(m.get("content") or "") for m in reversed(hist)
                             if m.get("role") == "assistant"), "")
            return best, ""
        except Exception as e:  # noqa BLE001 —— 失败诚实 abstain
            return "", f"{type(e).__name__}: {e}"[:200]


def _chat_task(assistant, proxy, prompt):
    """在池工作线程内执行：reset → 对话 → snapshot，单题 token 归因。"""
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
        print("[autogen][WARN] openai token 拦截未安装（版本路径不符），tokens 将记 0", flush=True)
    todo = pending(args.system, load_inputs())
    if args.limit:
        todo = todo[:args.limit]
    print(f"[{args.system}] 待跑 {len(todo)} | model={cfg['model']} | pyautogen 2-agent chat", flush=True)
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
    print(f"[{args.system}] 完成 -> outputs/traces/answers_{args.system}.jsonl", flush=True)


if __name__ == "__main__":
    main()
