# -*- coding: utf-8 -*-
"""扩测链进度哨兵：chain 日志尾部 + 各基线覆盖/错误数 + 活跃进程。"""
import io
import json
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
T = r"outputs/traces"

print("--- chain log 尾部 ---")
for line in open(T + r"\expansion_chain.log", encoding="utf-8", errors="replace").read().splitlines()[-8:]:
    print(" ", line)

print("--- 基线覆盖（目标=100）---")
for s in ("pure_llm", "lit_rag", "autogen", "mdagents"):
    done, err = {}, set()
    for l in open(T + rf"\answers_{s}.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        done[r["qid"]] = r
        if r.get("error"):
            err.add(r["qid"])
        else:
            err.discard(r["qid"])
    print(f"  {s:10s}: 成功 {len(done) - len(err):3d} / 含error {len(err)}")

mani = json.load(open(T + r"\sample_manifest.json", encoding="utf-8"))
qids = set(mani["qids"])
print(f"--- manifest: n={mani.get('n')} tv={mani.get('trace_version')} ---")
for s in ("pure_llm", "lit_rag", "autogen", "mdagents"):
    got = set()
    for l in open(T + rf"\answers_{s}.jsonl", encoding="utf-8"):
        if l.strip():
            got.add(json.loads(l)["qid"])
    miss = qids - got
    print(f"  {s:10s}: 缺 {len(miss)}  {sorted(miss)[:3]}")

print("--- 活跃 python 进程 ---")
out = subprocess.run(["powershell", "-NoProfile", "-Command",
                      "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                      "ForEach-Object { $_.CommandLine }"],
                     capture_output=True, text=True).stdout
for l in out.splitlines():
    if any(k in l for k in ("run_mdagents", "run_autogen", "merge_", "compare_eval", "label_")):
        print("  RUN:", l.split("\\")[-1][:90])
