# -*- coding: utf-8 -*-
"""v12..v15 结果对照：总体准确率 + v15 三通道表决分路径准确率（A + B 拆分）。"""
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
    latest = {}
    for r in rows:
        if str((r.get("provenance") or {}).get("trace_version")) == version:
            latest[r["qid"]] = r          # 同 qid 后写覆盖先写
    return list(latest.values())


def acc(version, arm, keep=None):
    c = t = 0
    for r in sel(version):
        if keep is not None and r["qid"] not in keep:
            continue
        t += 1
        if r["systems"][arm].get("outcome") == "correct":
            c += 1
    return (c, t, round(100 * c / t, 1) if t else 0)


def classify15(ans):
    rt = ans.get("rationale", "") or ""
    if "三通道·v15 一致" in rt:
        return "E==L1 agree(零成本直通)"
    if "三通道表决·v15" in rt:
        return "冲突->最终智能体辩论表决"
    if "纯LLM独立票·v15" in rt:
        return "法官缺席->多数票(L1/L2胜)"
    if "与证据通道一致" in rt:
        return "法官缺席->多数票(E胜)"
    if "无多数退v12规则" in rt:
        return "法官缺席->无多数退prior"
    if "维持证据通道原判" in rt:
        return "法官缺席->全弃权维持E"
    return "其他(证据直通/未触发表决)"


def perpath(version, arm, classify, keep=None):
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


A15 = {r["qid"] for r in sel("15")}
A12 = {r["qid"] for r in sel("12")}
print(f"v15 行数={len(A15)}  与 v12 题集相同={A15 == A12}")
A = A15
print("=== pipeline_auto acc(全体): v12 / v13 / v14 / v15  (A) ===")
for ver in ("12", "13", "14", "15"):
    print(f"  v{ver} A:", acc(ver, "pipeline_auto", A), " B:", acc(ver, "pipeline_auto", b_qids))
print("=== pipeline_heuristic acc: v12 / v15 ===")
for ver in ("12", "15"):
    print(f"  v{ver} A:", acc(ver, "pipeline_heuristic", A), " B:", acc(ver, "pipeline_heuristic", b_qids))
print("=== v15 pipeline_auto 分路径 (A) ===")
for k, v in perpath("15", "pipeline_auto", classify15, A).items():
    print(f"  {k:34s} {v[0]:2d}/{v[1]:2d} acc={v[2]}%")
print("=== v15 pipeline_auto 分路径 (B n=%d) ===" % len(b_qids & A))
for k, v in perpath("15", "pipeline_auto", classify15, b_qids).items():
    print(f"  {k:34s} {v[0]:2d}/{v[1]:2d} acc={v[2]}%")
print("=== v12 pipeline_auto 分路径(旧口径, 对照用 A) ===")
def classify12(ans):
    rt = ans.get("rationale", "") or ""
    if "prior·v11 一致" in rt:
        return "agree"
    if "法官" in rt:
        return "conflict->LLM法官"
    if "证据增强LLM" in rt:
        return "abstain->增强LLM"
    if "fallback" in rt:
        return "prior兜底"
    return "直通证据"
for k, v in perpath("12", "pipeline_auto", classify12, A).items():
    print(f"  {k:34s} {v[0]:2d}/{v[1]:2d} acc={v[2]}%")
