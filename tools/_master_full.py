# -*- coding: utf-8 -*-
"""全任务（membership + pairwise + topK）· 全系统 · 性能 & Token 消耗 一览表。

用法: .venv\Scripts\python.exe tools\_master_full.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "outputs" / "traces"

# ── Ground Truth ──
def load_gt():
    gtm = {}
    for line in (ROOT / "evaluation" / "ground_truth_clean.jsonl").read_text(encoding="utf-8").strip().splitlines():
        r = json.loads(line)
        gtm[r["qid"]] = r
    return gtm

GT = load_gt()

# ── M/P: 答案 JSONL → metrics ──
def eval_answer_file(name: str, label: str):
    path = TRACES / f"answers_{name}.jsonl"
    if not path.exists():
        return None
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().splitlines()]
    by_qid = {}
    for r in lines:
        by_qid[r["qid"]] = r
    buckets = {"membership": [], "pairwise": []}
    total_tok = 0
    for qid, ans in by_qid.items():
        gt = GT.get(qid)
        if not gt:
            continue
        b = gt.get("task_bucket", "")
        if b not in buckets: continue
        tok = ans.get("tokens", {}).get("total", 0)
        total_tok += tok
        if ans.get("abstained"):
            buckets[b].append("abstain")
        elif b == "membership":
            expected = gt["ground_truth"]["answer"]
            got = ans.get("answer")
            buckets[b].append("correct" if got == expected else "wrong")
        elif b == "pairwise":
            expected = gt["ground_truth"]["winner"]
            got = ans.get("winner")
            buckets[b].append("correct" if (got and got.upper() == expected.upper()) else ("wrong" if got else "abstain"))
    def _m(vals):
        c=vals.count("correct"); w=vals.count("wrong"); a=vals.count("abstain"); n=c+w+a; ans_n=c+w
        return (n, c, w, a, ans_n/n if n else 0, c/ans_n if ans_n else 0, c/n if n else 0)
    m=_m(buckets["membership"]); p=_m(buckets["pairwise"])
    return {"label": label, "m": m, "p": p, "all": _m(buckets["membership"]+buckets["pairwise"]), "tok": total_tok}


# ── Pipeline (tracestore.jsonl) filtered by trace_version ──
def eval_pipeline(version: str, label_prefix: str):
    """从 tracestore 中提取指定 trace_version 的 pipeline 结果（最新 per qid 去重）。"""
    store_path = TRACES / "tracestore.jsonl"
    if not store_path.exists(): return []
    # 读入并按 qid 去重（保留最后出现的 = 最新版本）
    latest: dict[str, dict] = {}
    for line in store_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if str((r.get("provenance") or {}).get("trace_version") or "") == version:
            latest[r["qid"]] = r

    results = []
    for sys_key, sys_label in [("pipeline_heuristic", f"{label_prefix}·管线启发式"),
                                ("pipeline_auto", f"{label_prefix}·管线auto类比")]:
        buckets = {"membership": [], "pairwise": []}
        for e in latest.values():
            b = e.get("bucket", "")
            if b not in buckets: continue
            outcome = e.get("systems", {}).get(sys_key, {}).get("outcome", "abstain")
            buckets[b].append(outcome)
        def _m(vals):
            c=vals.count("correct"); w=vals.count("wrong"); a=vals.count("abstain"); n=c+w+a; ans_n=c+w
            return (n, c, w, a, ans_n/n if n else 0, c/ans_n if ans_n else 0, c/n if n else 0)
        m=_m(buckets["membership"]); p=_m(buckets["pairwise"])
        results.append({"label": sys_label, "m": m, "p": p,
                        "all": _m(buckets["membership"]+buckets["pairwise"]), "tok": 0})
    return results


# ── TopK ──
def eval_topk(file_key: str, label: str):
    path = TRACES / f"topk_eval_{file_key}.json"
    if not path.exists(): return None
    d = json.loads(path.read_text(encoding="utf-8"))
    ok = [r for r in d["results"] if not r.get("error")]
    if not ok: return None
    ndcg = sum(r["ndcg5"] for r in ok)/len(ok)
    hit = sum(r["hit5"] for r in ok)/len(ok)
    tok = sum(r.get("tokens",{}).get("total",0) for r in d["results"])
    return {"label": label, "ndcg": ndcg, "hit": hit, "tok": tok}


# ── Formatting ──
def ftok(t):
    if t <= 0: return "--"
    if t >= 1_000_000: return f"{t/1_000_000:.1f}M"
    return f"{t/1_000:.0f}k"


# ═══════════════════════════════
def main():
    # 外部基线（答案 JSONL）
    ans_sys = [eval_answer_file(n, l) for n, l in [
        ("pure_llm", "pure_llm"), ("metagpt", "MetaGPT"), ("autogen", "AutoGen"),
        ("mdagents", "MDAgents"), ("lit_rag", "LitRAG")]]
    ans_sys = [s for s in ans_sys if s]

    # Pipeline: v16 和 v20 分别提取
    v16_pipe = eval_pipeline("16", "v16")
    v20_pipe = eval_pipeline("20", "v20")

    # 合并：外部基线在前，管线在后
    all_mp = ans_sys + v16_pipe + v20_pipe

    # 分区统计（用于标注 N）
    v16_n = v16_pipe[0]["m"][0] + v16_pipe[0]["p"][0] if v16_pipe else 0
    v20_n = v20_pipe[0]["m"][0] + v20_pipe[0]["p"][0] if v20_pipe else 0

    print("=" * 130)
    print("  全任务 · 全系统 · 性能 & Token 一览")
    print("=" * 130)

    # ── Membership ──
    print(f"\n  ── Membership (二分类: 基因是否属指定功能类) ──")
    print(f"  {'系统':<24s} {'N':>5s} {'对':>5s} {'错':>5s} {'弃':>4s} {'覆盖':>6s} {'准确(答)':>8s} {'准确(全)':>8s} {'净胜':>6s}  {'Token':>8s}")
    print(f"  {'─'*109}")
    for s in all_mp:
        n, c, w, a, cov, acc_ans, acc_all = s["m"]
        net = acc_all - 0.5
        print(f"  {s['label']:<24s} {n:>5d} {c:>5d} {w:>5d} {a:>4d} {cov*100:>5.1f}% {acc_ans*100:>7.1f}% {acc_all*100:>7.1f}% {net:>+6.1%}  {ftok(s['tok']):>8s}")

    # ── Pairwise ──
    print(f"\n  ── Pairwise (二选一: 哪个基因更可能为目标基因) ──")
    print(f"  {'系统':<24s} {'N':>5s} {'对':>5s} {'错':>5s} {'弃':>4s} {'覆盖':>6s} {'准确(答)':>8s} {'准确(全)':>8s} {'净胜':>6s}  {'Token':>8s}")
    print(f"  {'─'*109}")
    for s in all_mp:
        n, c, w, a, cov, acc_ans, acc_all = s["p"]
        net = acc_all - 0.5
        print(f"  {s['label']:<24s} {n:>5d} {c:>5d} {w:>5d} {a:>4d} {cov*100:>5.1f}% {acc_ans*100:>7.1f}% {acc_all*100:>7.1f}% {net:>+6.1%}  {ftok(s['tok']):>8s}")

    # ── TopK ──
    tk = [eval_topk(f, l) for f, l in [
        ("v20pergene", "v20pergene·Kira (SOTA)"), ("v20graph_ppr", "v20graph_ppr·Cael+PPR"),
        ("v20graph", "v20graph·Cael"), ("v20rrf_pergene", "v20rrf_pergene·RRF"),
        ("pure_llm", "pure_llm"), ("pipeline", "pipeline·PPR→LLM"),
        ("metagpt", "MetaGPT"), ("autogen", "AutoGen"), ("mdagents", "MDAgents")]]
    tk = [t for t in tk if t]
    sota = max(t["ndcg"] for t in tk)
    print(f"\n  ── TopK (CRISPR SLC top-5 排序, NDCG@5/hit@5, 11题) ──")
    print(f"  {'系统':<32s} {'NDCG@5':>7s} {'hit@5':>7s} {'ΔSOTA':>7s} {'Token':>8s}")
    print(f"  {'─'*70}")
    for t in tk:
        d = t["ndcg"] - sota; ds = f"{d:+.4f}"
        tok_s = ftok(t["tok"]) if t["tok"] > 0 else ("perGene" if "pergene" in t["label"] else "0")
        mark = " ★" if abs(d) < 0.001 else ""
        print(f"  {t['label']:<32s} {t['ndcg']:>7.4f} {t['hit']:>7.3f} {ds:>7s} {tok_s:>8s}{mark}")

    print(f"\n  {'='*130}")
    print(f"  注:")
    print(f"    · 外部基线(前5行): 同一批 489~631 题, N 不同因各系统运行覆盖不同, 可直接对比。")
    print(f"    · v16 管线(n≈{v16_n}): 冠军定版(triad vote+critique+race), 缓存复用全部已跑题。")
    print(f"    · v20 管线(n≈{v20_n}): R2 membership 必需性重排, 全体 73.0% 反超 v16 的 70.0%。")
    print(f"    · TopK Token: perGene=~18-20次/题独立基因评分; 0=纯图/规则无LLM。")
    print(f"  {'='*130}")


if __name__ == "__main__":
    main()