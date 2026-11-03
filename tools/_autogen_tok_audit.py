# -*- coding: utf-8 -*-
"""审计 autogen token 是否可信：cache.db(旧遗留) + answers_autogen.jsonl 0-token 行分布。"""
import json
import sqlite3
import statistics
from pathlib import Path

# 1) 旧 cache.db 结构(修复 cache_seed=None 前遗留)
try:
    c = sqlite3.connect(".cache/41/cache.db")
    tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
    print("cache.db tables:", tabs)
    for t in tabs:
        n = c.execute(f"select count(*) from [{t}]").fetchone()[0]
        print(" ", t, "rows=", n)
    if tabs:
        t0 = tabs[0]
        cols = [d[1] for d in c.execute(f"pragma table_info([{t0}])")]
        row = c.execute(f"select * from [{t0}] limit 1").fetchone()
        sample = str(row)[:400] if row else ""
        print("  列:", cols)
        print("  样例含 usage?", "usage" in sample, "| 含 prompt_tokens?", "prompt_tokens" in sample)
except Exception as e:  # noqa BLE001
    print("cache.db 读取失败:", e)

# 2) answers_autogen.jsonl token 健康度(去重后)
rows = {}
for l in Path("outputs/traces/answers_autogen.jsonl").open(encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        rows[r["qid"]] = r
tk = [r.get("tokens") or {} for r in rows.values()]
tot = [t.get("total", 0) or 0 for t in tk]
calls = [t.get("calls", 0) or 0 for t in tk]
z = sum(1 for x in tot if x == 0)
print(f"\nautogen 去重题数={len(rows)}  0-token行={z}")
if tot:
    print(f"total: 均值={statistics.mean(tot):.0f} 中位={statistics.median(tot)} min={min(tot)} max={max(tot)}")
    print(f"calls/题: 均值={statistics.mean(calls):.1f} min={min(calls)} max={max(calls)}")
err = [q for q, r in rows.items() if r.get("error")]
print(f"error行={len(err)}")
