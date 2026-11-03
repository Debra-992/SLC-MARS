# -*- coding: utf-8 -*-
"""v11/v12/v13 结果对照：总体准确率 + v13 分融合态准确率（A + B 拆分）。"""
import json
import collections
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = Path(__file__).resolve().parents[1] / "outputs" / "traces"
rows = [json.loads(l) for l in (BASE / "tracestore.jsonl").open(encoding="utf-8") if l.strip()]
bq = BASE / "dataset_b_qids.json"
b_qids = set(json.loads(bq.read_text(encoding="utf-8"))["b_qids"]) if bq.exists() else set()


def sel(version):
    return [r for r in rows if str((r.get("provenance") or {}).get("trace_version")) == version]


def acc(version, arm, keep=None):
    c = t = 0
    for r in sel(version):
        if keep is not None and r["qid"] not in keep:
            continue
        t += 1
        if r["systems"][arm].get("outcome") == "correct":
            c += 1
    return (c, t, round(100 * c / t, 1) if t else 0)


def classify(ans):
    rt = ans.get("rationale", "") or ""
    dm = str(ans.get("decision_mode", ""))
    if "+ansens" in dm:
        return ("v14 答案层集成(采纳)" if "低共识弃答" not in rt else "v14 答案层集成(低共识弃答)")
    if "一致" in rt:
        return "agree(证据∥先验同向)"
    if "法官" in rt or "+judge" in dm:
        return "conflict->LLM法官"
    if "增强LLM" in rt:
        return "abstain->增强LLM主审"
    if "fallback" in rt:
        return "aug弃答->裸/集成prior兜底"
    return "其他/直通证据"


def perpath(version, arm, keep=None):
    by = collections.defaultdict(lambda: [0, 0])
    for r in sel(version):
        if keep is not None and r["qid"] not in keep:
            continue
        s = r["systems"][arm]
        p = classify(s["answer"])
        by[p][1] += 1
        if s.get("outcome") == "correct":
            by[p][0] += 1
    return {k: (v[0], v[1], round(100 * v[0] / v[1], 1)) for k, v in
            sorted(by.items(), key=lambda x: -x[1][1])}


A = {r["qid"] for r in sel("14")} or {r["qid"] for r in sel("13")}
print("=== pipeline_auto acc(全体):  v11 / v12 / v13 / v14   (A n=40) ===")
for ver in ("11", "12", "13", "14"):
    print(f"  v{ver} A:", acc(ver, "pipeline_auto", A), " B:", acc(ver, "pipeline_auto", b_qids))
print("=== pipeline_heuristic acc: v12 / v13 / v14 ===")
for ver in ("12", "13", "14"):
    print(f"  v{ver} A:", acc(ver, "pipeline_heuristic", A), " B:", acc(ver, "pipeline_heuristic", b_qids))
print("=== v14 pipeline_auto 分融合态 (A) ===")
for k, v in perpath("14", "pipeline_auto", A).items():
    print(f"  {k:34s} {v[0]:2d}/{v[1]:2d} acc={v[2]}%")
print("=== v14 pipeline_auto 分融合态 (B n=%d) ===" % len(b_qids))
for k, v in perpath("14", "pipeline_auto", b_qids).items():
    print(f"  {k:34s} {v[0]:2d}/{v[1]:2d} acc={v[2]}%")
