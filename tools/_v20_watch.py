# -*- coding: utf-8 -*-
"""v20 臂 watcher：轮询 tracestore 直到 tv20=100 或预算耗尽。
用法: .venv\\Scripts\\python.exe tools\\_v20_watch.py [预算秒=1680] [间隔秒=240]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_agents.baselines.merge_baselines import load_jsonl  # noqa: E402

TS = Path("outputs/traces/tracestore.jsonl")
budget = int(sys.argv[1]) if len(sys.argv) > 1 else 1680
gap = int(sys.argv[2]) if len(sys.argv) > 2 else 240


def scan():
    n20 = r2 = err = 0
    for r in load_jsonl(TS):
        if str((r.get("provenance") or {}).get("trace_version") or "") == "20":
            n20 += 1
            a = ((r.get("systems") or {}).get("pipeline_auto") or {}).get("answer") or {}
            if "[R2 必需性重排" in str(a.get("rationale") or ""):
                r2 += 1
            if r.get("error"):
                err += 1
    return n20, r2, err


start = time.time()
while True:
    try:
        n20, r2, err = scan()
    except PermissionError:
        n20 = -1
    el = int(time.time() - start)
    print(f"[+{el}s] tv20={n20}/100 R2fired={r2} err={err}", flush=True)
    if n20 >= 100:
        print("DONE", flush=True)
        break
    if time.time() - start >= budget:
        print("BUDGET_HIT", flush=True)
        break
    time.sleep(gap)
