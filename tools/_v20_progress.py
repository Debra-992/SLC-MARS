# -*- coding: utf-8 -*-
"""v20 臂进度探针：数 tracestore 里 trace_version=20 的行数 + R2 触发数；带锁重试。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_agents.baselines.merge_baselines import load_jsonl  # noqa: E402

TS = Path("outputs/traces/tracestore.jsonl")


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


for _ in range(6):
    try:
        n20, r2, err = scan()
        print(f"tv20 rows={n20}/100  R2 fired={r2}  errored rows={err}")
        break
    except PermissionError:
        time.sleep(3)
else:
    print("tracestore 持续被占用，稍后再看")
