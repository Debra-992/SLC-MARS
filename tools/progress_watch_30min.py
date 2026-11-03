# -*- coding: utf-8 -*-
"""30 分钟自动进度计时器（独立后台进程，不依赖 agent 在线）。

每 30 分钟把一份快照追加写入 outputs/traces/progress_30min.log：
  * v20 臂：tracestore 里 trace_version=20 的行数 / R2 触发 / errored（/100）
  * 4 个 baseline：已成功唯一 qid / 489（口径 = common.pending，与续跑逻辑一致，error 行计待补）
全部 baseline 满 489 且 v20 满 100 → 收尾退出；否则最多跑 MAX_TICKS 个周期后退出。

用法（由 run_baselines_all500.ps1 或手动后台起）：
  .\\.venv\\Scripts\\python.exe tools\\progress_watch_30min.py
"""
from __future__ import annotations

import datetime as _dt
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knowledge_agents.baselines.common import load_inputs, pending  # noqa: E402

TRACES = ROOT / "outputs" / "traces"
WATCH_LOG = TRACES / "progress_30min.log"
TS_PATH = TRACES / "tracestore.jsonl"
SYSTEMS = ["pure_llm", "lit_rag", "autogen", "mdagents"]
INTERVAL = 1800          # 30 分钟
MAX_TICKS = 36           # 上限 18 小时，防永久挂起


def _read_lines(path: Path, tries: int = 6):
    """读 jsonl，容忍 v20/runner 并发写导致的文件占用（PermissionError 退避重试）。"""
    for _ in range(tries):
        try:
            txt = path.read_text(encoding="utf-8")
            return [ln for ln in txt.splitlines() if ln.strip()]
        except (PermissionError, OSError):
            time.sleep(2)
    return []


def v20_scan():
    n20 = r2 = err = 0
    import json
    for ln in _read_lines(TS_PATH):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if str((r.get("provenance") or {}).get("trace_version") or "") == "20":
            n20 += 1
            a = ((r.get("systems") or {}).get("pipeline_auto") or {}).get("answer") or {}
            if "[R2 必需性重排" in str(a.get("rationale") or ""):
                r2 += 1
            if r.get("error"):
                err += 1
    return n20, r2, err


def snapshot(inputs):
    n20, r2, err = v20_scan()
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [f"[{stamp}] v20 {n20}/100 (R2={r2} err={err})"]
    all_bl_done = True
    for s in SYSTEMS:
        pend = len(pending(s, inputs))
        done = len(inputs) - pend
        if pend > 0:
            all_bl_done = False
        parts.append(f"{s} {done}/{len(inputs)}")
    v20_done = n20 >= 100
    line = "  ".join(parts)
    with WATCH_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)
    return all_bl_done and v20_done


def main():
    inputs = [i for i in load_inputs() if i["bucket"] in {"pairwise", "membership"}]
    msg = f"=== progress_watch 起 间隔{INTERVAL}s 目标baseline={len(inputs)} v20=100 ==="
    print(msg, flush=True)
    done = False
    for tick in range(MAX_TICKS):
        done = snapshot(inputs)
        if done:
            with WATCH_LOG.open("a", encoding="utf-8") as f:
                f.write(f"[{_dt.datetime.now():%H:%M:%S}] ALL DONE\n")
            break
        time.sleep(INTERVAL)
    print("progress_watch 退出", flush=True)


if __name__ == "__main__":
    main()
