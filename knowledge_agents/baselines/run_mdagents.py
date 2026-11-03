# -*- coding: utf-8 -*-
"""Baseline 4：MDAgents (mitmedialab, NeurIPS'24) 多智能体自适应协作。

独立 venv：outputs\\mdagents-venv（py3.11 + 官方 requirements）；仓库克隆在 outputs\\MDAgents。
**不改克隆源码**：run 前 monkeypatch ——
  * `openai.OpenAI` 子类化：所有 client 强制指向 config2.yaml 的 qwen 兼容端点，
    并把调用方传的 model 一律替换为 qwen 模型名（原码硬编码 gpt-4o-mini/gpt-3.5-turbo）。
  * `google.generativeai` 用桩替代（不用 gemini 分支，避免多装一个 SDK）。
框架自身的 adaptive difficulty 路由 + basic/intermediate/advanced 三层协作逻辑原样保留。

用法：
  .\\outputs\\mdagents-venv\\Scripts\\python.exe knowledge_agents\\baselines\\run_mdagents.py [--limit N]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MDA = os.path.join(REPO, "outputs", "MDAgents")
sys.path.insert(0, REPO)
sys.path.insert(0, MDA)
# MDAgents 源码往控制台 cprint 医生 emoji —— Windows GBK 标准输出会 UnicodeEncodeError
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa BLE001
        pass

# ---------------------------------------------------------------- 端点/模型 patch（须在 import utils 之前）
from knowledge_agents.baselines.common import (  # noqa: E402
    append_answer, build_prompt, extract_json, llm_settings, load_inputs, normalize, pending,
    TokenSink, usage_triplet,
)

CFG = llm_settings()
SYSTEM = "mdagents"
_sink = TokenSink()               # token 计量（在 QwenClient.create 咽喉处累加）
PER_Q_TIMEOUT = float(os.environ.get("SLC_MDAGENTS_TIMEOUT", "1800"))  # advanced 一题可达数十次调用


def _patch_openai() -> None:
    import openai

    real = openai.OpenAI

    class QwenClient(real):
        def __init__(self, *a, **kw):
            kw["api_key"] = CFG["api_key"]
            kw["base_url"] = CFG["base_url"]
            kw.setdefault("max_retries", 1)
            super().__init__(*a, **kw)
            real_create = self.chat.completions.create

            def create(*ca, **ckw):
                ckw["model"] = CFG["model"]
                ckw.setdefault("timeout", 300)
                rsp = real_create(*ca, **ckw)
                try:
                    if not ckw.get("stream"):
                        p, c, t = usage_triplet(rsp)
                        _sink.add(p, c, t)
                except Exception:  # noqa BLE001 —— 计量不干扰作答
                    pass
                return rsp

            self.chat.completions.create = create

    openai.OpenAI = QwenClient


def _stub_gemini() -> None:
    if "google.generativeai" in sys.modules:
        return
    g = types.ModuleType("google")
    ga = types.ModuleType("google.generativeai")
    ga.configure = lambda **kw: None            # 仅 gemini 分支会调；我们用 gpt 风格分支
    ga.GenerativeModel = object
    g.generativeai = ga
    sys.modules["google"] = g
    sys.modules["google.generativeai"] = ga


os.environ.setdefault("openai_api_key", CFG["api_key"])
_patch_openai()
_stub_gemini()

from utils import (  # noqa: E402  (MDAgents 官方逻辑，克隆未改动)
    determine_difficulty, process_advanced_query, process_basic_query, process_intermediate_query,
)
import utils as _mda  # noqa: E402

# ---- 防御性补丁（只兜住官方解析的崩溃，不改协作算法）----
# 1) parse_hierarchy 假定 hierarchy 严格是 "A > B" 模板；qwen 偏离时 split('>')[1] 越界。
#    该树【仅用于 print_tree 展示】——debate 逻辑用的是扁平 medical_agents，退化=平铺，无算法增益。
_real_parse_hierarchy = _mda.parse_hierarchy


def _safe_parse_hierarchy(info, emojis):
    try:
        return _real_parse_hierarchy(info, emojis)
    except (IndexError, AttributeError):
        from pptree import Node
        moderator = Node('moderator')
        agents = [moderator]
        for k, (expert, _h) in enumerate(info):
            name = "{} ({})".format(expert.split("-")[0].strip(), emojis[k % len(emojis)])
            agents.append(Node(name, moderator))
        return agents


_mda.parse_hierarchy = _safe_parse_hierarchy

# 2) Group.interact：qwen 的 MDT 招募输出不合模板 -> members 解析为空 -> assist_members[0] 越界。
#    该组退化为单专家作答（其余组合照旧协作），不再整题崩溃。
_RealGroup = _mda.Group


class _RobustGroup(_RealGroup):
    def interact(self, comm_type, message=None, img_path=None):
        try:
            return super().interact(comm_type, message=message, img_path=img_path)
        except IndexError:
            solo = _mda.Agent("You are a helpful medical agent.", role='medical expert',
                              model_info=_Args.model)
            solo.chat("You are a helpful medical agent.")
            return solo.chat(self.question)


_mda.Group = _RobustGroup


class _Args:                    # MDAgents 需要的最简 args（dataset 非 medqa -> 不吃它的 examplers 逻辑）
    dataset = "custom"
    difficulty = "adaptive"
    model = "gpt-4o-mini"       # 实际模型名被 QwenClient 统一替换为 qwen


def _decision_text(final) -> str:
    """递归展平 —— intermediate 返回 {'majority': {0.0: text}}，直接 str(v) 会变成
    单引号 repr 让 extract_json 解析失败，须一路取到叶子字符串。"""
    if isinstance(final, dict):
        return "\n".join(_decision_text(v) for v in final.values())
    if isinstance(final, (list, tuple)):
        return "\n".join(_decision_text(v) for v in final)
    return str(final or "")


def run_one(rec: dict) -> tuple[str, str, str, dict]:
    question = build_prompt(rec)          # 与 pure_llm/autogen 同一题面指令（含 JSON 输出要求）
    _sink.reset()
    try:
        diff = determine_difficulty(question, _Args.difficulty) or "basic"
        if diff == "basic":
            final = process_basic_query(question, [], _Args.model, _Args())
        elif diff == "intermediate":
            final = process_intermediate_query(question, [], _Args.model, _Args())
        else:
            final = process_advanced_query(question, _Args.model, _Args())
        return _decision_text(final), "", diff, _sink.snapshot()
    except Exception as e:  # noqa BLE001 —— 失败诚实 abstain
        import traceback
        tb = traceback.format_exc().strip().splitlines()
        loc = " | ".join(l.strip() for l in tb[-4:])
        return "", f"{type(e).__name__}: {e} @@ {loc}"[:400], "?", _sink.snapshot()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--system", default=SYSTEM)
    args = ap.parse_args()

    inputs = load_inputs()
    qfile = os.environ.get("SLC_MDA_QIDFILE", "")
    if qfile and os.path.exists(qfile):
        keep = set(json.loads(open(qfile, encoding="utf-8").read())["qids"])
        inputs = [r for r in inputs if r["qid"] in keep]
        print(f"[{args.system}] 子集模式: 清单 {len(keep)} qids -> 命中题面 {len(inputs)}（已答自动跳过=复用）", flush=True)
    todo = pending(args.system, inputs)
    if args.limit:
        todo = todo[:args.limit]
    print(f"[{args.system}] 待跑 {len(todo)} | adaptive 路由 + 单题超时 {PER_Q_TIMEOUT:.0f}s", flush=True)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    for i, rec in enumerate(todo, 1):
        t0 = time.time()
        fut = pool.submit(run_one, rec)
        try:
            text, err, diff, tok = fut.result(timeout=PER_Q_TIMEOUT)
        except concurrent.futures.TimeoutError:
            text, err, diff, tok = "", "TimeoutError: mdagents query exceeded deadline", "?", _sink.snapshot()
        data = extract_json(text) if text else None
        row = normalize(rec, data, system=args.system,
                        error=err if err else ("" if data else "no_json_parse"))
        row["difficulty"] = diff
        row["elapsed_s"] = round(time.time() - t0, 1)
        row["tokens"] = tok
        append_answer(args.system, row)
        print(f"  [{i:>2}/{len(todo)}] {row['qid'][:40]:<42} diff={diff:<12} win={row['winner']} "
              f"ans={row['answer']} abst={row['abstained']} tok={tok['total']} calls={tok['calls']} {row['elapsed_s']}s {err[:200]}", flush=True)
    pool.shutdown(wait=False)
    print(f"[{args.system}] 完成 -> outputs/traces/answers_{args.system}.jsonl", flush=True)


if __name__ == "__main__":
    main()
