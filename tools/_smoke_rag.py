# -*- coding: utf-8 -*-
import json
import time

from knowledge_agents.baselines import mongo_rag as mr

inp = json.load(open("outputs/traces/baselines_input.json", encoding="utf-8"))
for r in inp[:4]:
    t0 = time.time()
    p = mr.retrieve(r["question"], r["candidates"], r["constraints"])
    el = time.time() - t0
    top = p[0]["score"] if p else "-"
    print(f"{r['qid'][:40]:<42} hits={len(p)} top={top} {el:.1f}s")
print("ctx sample:", mr.format_context(p)[:160].replace("\n", " | "))
