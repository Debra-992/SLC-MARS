# -*- coding: utf-8 -*-
"""Baseline 4: MDAgents (mitmedialab, NeurIPS'24) multi-agent adaptive collaboration.

Standalone venv: outputs\\mdagents-venv (py3.11 + the official requirements); the clone lives in outputs\\MDAgents.
**The cloned source is not edited**: monkeypatches are installed before the run --
  * `openai.OpenAI` subclassed: every client is forced onto the openai-compatible endpoint from config2.yaml,
    and any model name the caller passes is replaced by the qwen model (the upstream code hardcodes gpt-4o-mini/gpt-3.5-turbo).
  * `google.generativeai` is stubbed out (the gemini branch is never used, so no extra SDK is installed).
The framework's own adaptive-difficulty routing and basic/intermediate/advanced collaboration logic stay untouched.

Usage:
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
# MDAgents cprints doctor emojis to the console -- Windows' GBK stdout raises UnicodeEncodeError
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa BLE001
        pass

# ---------------------------------------------------------------- endpoint/model patch (must precede import utils)
from knowledge_agents.baselines.common import (  # noqa: E402
    append_answer, build_prompt, extract_json, llm_settings, load_inputs, normalize, pending,
    TokenSink, usage_triplet,
)

CFG = llm_settings()
SYSTEM = "mdagents"
_sink = TokenSink()  # token metering (accumulated at the QwenClient.create chokepoint)
PER_Q_TIMEOUT = float(os.environ.get("SLC_MDAGENTS_TIMEOUT", "1800"))  # advanced can make dozens of calls per question


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
                except Exception:  # noqa BLE001 -- metering must not disturb answering
                    pass
                return rsp

            self.chat.completions.create = create

    openai.OpenAI = QwenClient


def _stub_gemini() -> None:
    if "google.generativeai" in sys.modules:
        return
    g = types.ModuleType("google")
    ga = types.ModuleType("google.generativeai")
    ga.configure = lambda **kw: None  # only the gemini branch calls it; we use the gpt-style branch
    ga.GenerativeModel = object
    g.generativeai = ga
    sys.modules["google"] = g
    sys.modules["google.generativeai"] = ga


os.environ.setdefault("openai_api_key", CFG["api_key"])
_patch_openai()
_stub_gemini()

from utils import (  # noqa: E402  (official MDAgents logic, the clone is unmodified)
    determine_difficulty, process_advanced_query, process_basic_query, process_intermediate_query,
)
import utils as _mda  # noqa: E402

# ---- defensive patches (they only keep official parsing from crashing; the collaboration algorithm is unchanged) ----
# 1) parse_hierarchy assumes the hierarchy is strictly the "A > B" template; when qwen deviates, split('>')[1] goes out of range.
#    That tree is [only used by print_tree] -- the debate logic uses the flat medical_agents list, so degrading to a flat list costs no algorithm.
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

# 2) Group.interact: qwen's MDT recruitment output does not match the template -> members parses empty -> assist_members[0] out of range.
#    That group degrades to a single specialist answer (the other groups still collaborate), instead of losing the whole question.
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


class _Args:  # the minimal args object MDAgents needs (dataset is not medqa, so its examplers logic is skipped)
    dataset = "custom"
    difficulty = "adaptive"
    model = "gpt-4o-mini"  # the actual model name is replaced with the qwen model by QwenClient


def _decision_text(final) -> str:
    """Recursive flattening -- intermediate returns {'majority': {0.0: text}}, and str(v) would yield
    a single-quoted repr that extract_json cannot parse, so walk down to the leaf strings."""
    if isinstance(final, dict):
        return "\n".join(_decision_text(v) for v in final.values())
    if isinstance(final, (list, tuple)):
        return "\n".join(_decision_text(v) for v in final)
    return str(final or "")


def run_one(rec: dict) -> tuple[str, str, str, dict]:
    question = build_prompt(rec)  # the same stem and instruction as pure_llm/autogen (including the JSON output requirement)
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
    except Exception as e:  # noqa BLE001 -- honest failure -> abstain
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
        print(f"[{args.system}] subset mode: list of {len(keep)} qids -> {len(inputs)} matching stems (already-answered ones skipped = reuse)", flush=True)
    todo = pending(args.system, inputs)
    if args.limit:
        todo = todo[:args.limit]
    print(f"[{args.system}] to run {len(todo)} | adaptive routing + per-question timeout {PER_Q_TIMEOUT:.0f}s", flush=True)
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
    print(f"[{args.system}] done -> outputs/traces/answers_{args.system}.jsonl", flush=True)


if __name__ == "__main__":
    main()
