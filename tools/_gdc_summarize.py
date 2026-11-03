# -*- coding: utf-8 -*-
"""把 stdin 里的 GDC POST 结果摘要成 (sample_type, workflow_type) 计数表。"""
import json
import sys
from collections import Counter

d = json.load(sys.stdin)
hits = d.get("data", {}).get("hits", {})
total = hits.get("total")
pag = hits.get("hits", [])
print("paging.total =", total, "| returned =", len(pag))
c = Counter()
for h in pag:
    wf = (h.get("analysis") or {}).get("workflow_type")
    c[(h.get("sample_type"), wf, h.get("experimental_strategy"))] += 1
for k, v in sorted(c.items(), key=lambda x: -x[1]):
    print(f"   {v:4d}  sample_type={k[0]!r:28s} workflow={k[1]!r:26s} strategy={k[2]!r}")
if pag:
    print("   首条样例 file_name:", pag[0].get("file_name"), "| file_id:", pag[0].get("file_id"))
