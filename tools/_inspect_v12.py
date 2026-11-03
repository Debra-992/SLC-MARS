# -*- coding: utf-8 -*-
"""v12 per-fusion-path accuracy: which branch earned the gain (A + B split)."""
import json
import collections
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = Path(__file__).resolve().parents[1] / "outputs" / "traces"
rows = [json.loads(l) for l in (BASE / "tracestore.jsonl").open(encoding="utf-8") if l.strip()]
v12 = [r for r in rows if (r.get("provenance") or {}).get("trace_version") == "12"]

bq = BASE / "dataset_b_qids.json"
b_qids = set(json.loads(bq.read_text(encoding="utf-8"))["b_qids"]) if bq.exists() else set()


def classify(ans):
    rt = ans.get("rationale", "") or ""
    dm = str(ans.get("decision_mode", ""))
    if "一致" in rt:
        return "agree(证据∥先验同向)"
    if "增强LLM" in rt or ("v12" in rt and "法官" not in rt and "fallback" not in rt and "+prior" in dm):
        return "abstain->增强LLM主审"
    if "法官" in rt or "+judge" in dm:
        return "conflict->LLM法官"
    if "fallback" in rt:
        return "aug弃答->裸prior兜底"
    return "其他/直通"


def stat(arm, keep=None):
    by = collections.defaultdict(lambda: [0, 0])  # path -> [correct, total]
    for r in v12:
        if keep is not None and r["qid"] not in keep:
            continue
        s = r["systems"][arm]
        p = classify(s["answer"])
        by[p][1] += 1
        if s.get("outcome") == "correct":
            by[p][0] += 1
    return {k: (v[0], v[1], round(100 * v[0] / v[1], 1)) for k, v in sorted(by.items(), key=lambda x: -x[1][1])}


print("=== v12 pipeline_auto per-path  (A n=40) ===")
for k, v in stat("pipeline_auto").items():
    print(f"  {k:28s} correct={v[0]:2d}/{v[1]:2d}  acc={v[2]}%")
print("=== v12 pipeline_auto per-path  (B n=%d) ===" % len(b_qids))
for k, v in stat("pipeline_auto", b_qids).items():
    print(f"  {k:28s} correct={v[0]:2d}/{v[1]:2d}  acc={v[2]}%")

# B-decay robustness: acc on A vs B for pipeline vs pure-llm
def acc_all(arm, keep):
    c = t = 0
    for r in v12:
        if keep is not None and r["qid"] not in keep:
            continue
        t += 1
        if r["systems"][arm].get("outcome") == "correct":
            c += 1
    return round(100 * c / t, 1) if t else 0


a_qids = {r["qid"] for r in v12}
print("B-decay pipeline-auto: A=%s%% B=%s%%" % (acc_all("pipeline_auto", a_qids), acc_all("pipeline_auto", b_qids)))
