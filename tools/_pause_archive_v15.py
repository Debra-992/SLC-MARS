# -*- coding: utf-8 -*-
"""断网暂停存档：备份 tracestore，并剔除 v15 首轮（10:11~12:10 本地 = 02:11~04:11Z）被
端点超时污染的 v15 行，保留 v15b 复跑（>=04:50Z）的干净行。续跑时不加 --no-cache，
compare_eval 会复用干净行、只重算剩余题。"""
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TR = Path(__file__).resolve().parents[1] / "outputs" / "traces"
src = TR / "tracestore.jsonl"
bak = TR / "tracestore_before_v15pause.jsonl.bak"
if not bak.exists():
    shutil.copy2(src, bak)
    print(f"备份 -> {bak.name}")

keep, dropped = [], 0
for line in src.open(encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    p = r.get("provenance") or {}
    if str(p.get("trace_version")) == "15" and (p.get("ts") or "") < "2026-10-23T04:50":
        dropped += 1          # 首轮污染行（含其后续被复跑覆盖前的旧写）
        continue
    keep.append(line if line.endswith("\n") else line + "\n")

src.write_text("".join(keep), encoding="utf-8")
clean15 = sum(1 for l in keep if '"trace_version": "15"' in l or '"trace_version":"15"' in l)
print(f"剔除首轮污染 v15 行 {dropped} 条；保留 v15 干净行 {clean15} 题；总行数 {len(keep)}")
