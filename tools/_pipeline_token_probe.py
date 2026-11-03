# -*- coding: utf-8 -*-
"""管线 per-question LLM token 探针（只测成本，绝不写 tracestore / 不污染缓存）。

背景：compare_eval 的 tracestore 里 systems 只存 answer/outcome，没记 token；管线 LLM
全走 AsyncOpenAI.chat.completions.create（prior/debate/decision/L2）。本探针在进程内
中央 monkeypatch openai 的 create（同 install_temperature_zero 的拦截点），复位 sink ->
跑 compare_eval.trace_one(llm_ok=False)（supply + heuristic decide + auto decide，
不含 MetaGPT baseline）-> 快照 sink，得到该题的管线 token。逐题汇总写
outputs/traces/pipeline_token_probe.json 供"效率—精度"总表引用。

必须在端点空闲时跑（它会真打 qwen）。env 需与 v500 同（slc_mars_env.ps1 dot-source）。
用法: .venv\\Scripts\\python.exe tools\\_pipeline_token_probe.py --n 40 --seed 5
"""
from __future__ import annotations
import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knowledge_agents.answer_pipeline import AnswerTeam, sample_questions  # noqa: E402
from knowledge_agents.compare_eval import trace_one  # noqa: E402  复用完全相同的管线路径
from knowledge_agents.baselines.common import TRACES, usage_triplet  # noqa: E402

_V = [0, 0, 0, 0]   # prompt, completion, total, calls（进程级累计，单进程串行归因干净）


def _add(p: int, c: int, t: int) -> None:
    _V[0] += p; _V[1] += c; _V[2] += t; _V[3] += 1


def _install_tracking() -> bool:
    """monkeypatch 同步+异步 openai chat.completions.create，把 usage 累进 _V。
    不改返回值/异常语义；探针专用，跑完即弃（不写回任何 store）。"""
    try:
        from openai.resources.chat.completions import Completions, AsyncCompletions
    except Exception:  # noqa BLE001
        return False
    real_s = Completions.create
    if not getattr(real_s, "__probe_tok__", False):
        def create(self, *a, **kw):
            rsp = real_s(self, *a, **kw)
            try:
                if not kw.get("stream"):
                    p, c, t = usage_triplet(rsp); _add(p, c, t)
            except Exception:  # noqa BLE001 计量绝不干扰作答
                pass
            return rsp
        create.__probe_tok__ = True
        Completions.create = create
    real_a = AsyncCompletions.create
    if not getattr(real_a, "__probe_tok__", False):
        async def acreate(self, *a, **kw):
            rsp = await real_a(self, *a, **kw)
            try:
                if not kw.get("stream"):
                    p, c, t = usage_triplet(rsp); _add(p, c, t)
            except Exception:  # noqa BLE001
                pass
            return rsp
        acreate.__probe_tok__ = True
        AsyncCompletions.create = acreate
    return True


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--buckets", default="pairwise,membership")
    ap.add_argument("--retries", type=int, default=2, help="单题 calls==0(端点抖动/失败) 时额外重试次数")
    a = ap.parse_args()
    buckets = [b.strip() for b in a.buckets.split(",") if b.strip()]

    if not _install_tracking():
        print(">>> 警告：openai 路径不符，token 插桩安装失败（将记 0）")
    team = AnswerTeam()
    sample = sample_questions(a.n, buckets, a.seed)
    print(f"[管线 token 探针] n={len(sample)} seed={a.seed} 桶={buckets}  "
          f"(supply + heuristic decide + auto decide；不含 MetaGPT)", flush=True)

    per = []
    valid = 0
    for i, rec in enumerate(sample, 1):
        snap = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
        ok = True
        for attempt in range(a.retries + 1):
            _V[0] = _V[1] = _V[2] = _V[3] = 0
            try:
                await trace_one(team, rec, llm_ok=False)   # 丢弃返回，不 append_store
            except Exception as e:  # noqa BLE001
                ok = False
                print(f"    [ERR] {rec['qid'][:40]} {type(e).__name__}: {str(e)[:60]}", flush=True)
            snap = {"prompt": _V[0], "completion": _V[1], "total": _V[2], "calls": _V[3]}
            if snap["calls"] > 0 and snap["total"] > 0:   # 真实冷成本
                ok = True
                break
            if attempt < a.retries:                        # 端点抖动导致 0 调用 → 稍等重试
                await asyncio.sleep(6)
        good = snap["calls"] > 0 and snap["total"] > 0
        valid += 1 if good else 0
        per.append({"qid": rec["qid"], "bucket": rec.get("task_bucket", ""), "ok": ok, **snap})
        print(f"[{i:>3}/{len(sample)}] {rec['qid'][:40]:<40} bucket={rec.get('task_bucket',''):<10} "
              f"tok={snap['total']:>7} calls={snap['calls']:>3} "
              f"{'VALID' if good else 'ZERO(端点废行,弃)'}", flush=True)

    okrows = [r for r in per if r["ok"] and r["total"] > 0]

    def _mean(k, rows=okrows):
        return statistics.mean([r[k] for r in rows]) if rows else 0

    by_bucket = {}
    by_bucket_n = {}
    for b in buckets:
        rb = [r["total"] for r in okrows if r["bucket"] == b]
        if rb:
            by_bucket[b] = round(statistics.mean(rb))
            by_bucket_n[b] = len(rb)

    out = {
        "note": "管线每题 LLM token（supply+heuristic decide+auto decide；不含 MetaGPT baseline）。"
                "tracestore 未插桩，故用独立探针 trace_one 复算实测。calls==0 视为端点废行已剔除。",
        "measured_at": str(__import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        "probe_n": len(sample), "ok_n": len(okrows), "valid_n": valid, "seed": a.seed,
        "mean_total": round(_mean("total")), "mean_prompt": round(_mean("prompt")),
        "mean_completion": round(_mean("completion")), "mean_calls": round(_mean("calls"), 1),
        "mean_total_by_bucket": by_bucket, "n_by_bucket": by_bucket_n,
        "per_q": per,
    }
    p = TRACES / "pipeline_token_probe.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[written] {p}\n  每题均值 total={out['mean_total']}  prompt={out['mean_prompt']} "
          f"completion={out['mean_completion']} calls={out['mean_calls']}  (有效 {valid}/{len(sample)})  "
          f"分桶={by_bucket} 分桶n={by_bucket_n}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
