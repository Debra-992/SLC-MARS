# -*- coding: utf-8 -*-
"""v20 版本 · 全量样本 · 三任务性能 & Token + 消融实验。

用法: .venv\Scripts\python.exe tools\_v20_master_report.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "outputs" / "traces"

# ═══════════════════════════════════════════════════════════════
#  Ground Truth
# ═══════════════════════════════════════════════════════════════
def load_gt():
    gtm = {}
    for line in (ROOT / "evaluation" / "ground_truth_clean.jsonl").read_text(encoding="utf-8").strip().splitlines():
        r = json.loads(line)
        gtm[r["qid"]] = r
    return gtm

GT = load_gt()


# ═══════════════════════════════════════════════════════════════
#  M/P: 答案 JSONL → metrics
# ═══════════════════════════════════════════════════════════════
def eval_answer_file(name: str):
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
        c = vals.count("correct"); w = vals.count("wrong"); a = vals.count("abstain")
        n = c + w + a; ans_n = c + w
        return (n, c, w, a, ans_n / n if n else 0, c / ans_n if ans_n else 0, c / n if n else 0)
    m = _m(buckets["membership"]); p = _m(buckets["pairwise"])
    return {"m": m, "p": p, "all": _m(buckets["membership"] + buckets["pairwise"]), "tok": total_tok}


# ═══════════════════════════════════════════════════════════════
#  Pipeline (tracestore.jsonl) filtered by trace_version
# ═══════════════════════════════════════════════════════════════
def eval_pipeline(version: str):
    store_path = TRACES / "tracestore.jsonl"
    if not store_path.exists(): return {}
    latest: dict[str, dict] = {}
    for line in store_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if str((r.get("provenance") or {}).get("trace_version") or "") == version:
            latest[r["qid"]] = r
    result = {}
    for sys_key in ("pipeline_heuristic", "pipeline_auto"):
        buckets = {"membership": [], "pairwise": []}
        for e in latest.values():
            b = e.get("bucket", "")
            if b not in buckets: continue
            outcome = e.get("systems", {}).get(sys_key, {}).get("outcome", "abstain")
            buckets[b].append(outcome)
        def _m(vals):
            c = vals.count("correct"); w = vals.count("wrong"); a = vals.count("abstain")
            n = c + w + a; ans_n = c + w
            return (n, c, w, a, ans_n / n if n else 0, c / ans_n if ans_n else 0, c / n if n else 0)
        result[sys_key] = {"m": _m(buckets["membership"]), "p": _m(buckets["pairwise"]),
                           "all": _m(buckets["membership"] + buckets["pairwise"])}
    return result


# ═══════════════════════════════════════════════════════════════
#  TopK
# ═══════════════════════════════════════════════════════════════
def eval_topk(file_key: str):
    path = TRACES / f"topk_eval_{file_key}.json"
    if not path.exists(): return None
    d = json.loads(path.read_text(encoding="utf-8"))
    ok = [r for r in d["results"] if not r.get("error")]
    if not ok: return None
    ndcg = sum(r["ndcg5"] for r in ok) / len(ok)
    hit = sum(r["hit5"] for r in ok) / len(ok)
    tok = sum(r.get("tokens", {}).get("total", 0) for r in d["results"])
    return {"ndcg": ndcg, "hit": hit, "tok": tok, "n": len(ok)}


# ═══════════════════════════════════════════════════════════════
#  Formatting
# ═══════════════════════════════════════════════════════════════
def ftok(t):
    if t <= 0: return "--"
    if t >= 1_000_000: return f"{t / 1_000_000:.1f}M"
    return f"{t / 1_000:.0f}k"

def fmt_mp_row(label, tup, tok, label_w=28):
    n, c, w, a, cov, acc_ans, acc_all = tup
    net = acc_all - 0.5
    return (f"  {label:<{label_w}s} {n:>5d} {c:>5d} {w:>5d} {a:>4d} "
            f"{cov*100:>5.1f}% {acc_ans*100:>7.1f}% {acc_all*100:>7.1f}% {net:>+6.1%}  {ftok(tok):>8s}")

MP_HEADER = (f"  {'系统':<28s} {'N':>5s} {'对':>5s} {'错':>5s} {'弃':>4s} "
             f"{'覆盖':>6s} {'准确(答)':>8s} {'准确(全)':>8s} {'净胜':>6s}  {'Token':>8s}")
MP_SEP = f"  {'─'*113}"


# ═══════════════════════════════════════════════════════════════
def main():
    # ── 外部基线 (答案 JSONL) ──
    BASELINES = [
        ("pure_llm",  "pure_llm"),
        ("lit_rag",   "LitRAG"),
        ("metagpt",   "MetaGPT"),
        ("autogen",   "AutoGen"),
        ("mdagents",  "MDAgents"),
    ]
    base_mp = []
    for fname, label in BASELINES:
        r = eval_answer_file(fname)
        if r:
            base_mp.append({"label": label, **r})

    # ── v20 管线 ──
    v20p = eval_pipeline("20")
    v20_heur = v20p.get("pipeline_heuristic", {})
    v20_auto = v20p.get("pipeline_auto", {})

    # ── TopK: 基线 + v20 核心系统 ──
    TOPK_BASE = [
        ("pure_llm",   "pure_llm · 直接LLM排序"),
        ("pipeline",   "pipeline · PPR→LLM重排"),
        ("metagpt",    "MetaGPT"),
        ("autogen",    "AutoGen"),
        ("mdagents",   "MDAgents"),
    ]
    TOPK_CORE = [
        ("v20pergene",   "v20pergene · Kira 逐基因LLM评分"),
        ("v20graph_ppr", "v20graph_ppr · Cael 图谱+PPR"),
        ("v20graph",     "v20graph · Cael 纯图谱"),
        ("v20rrf_llm",   "v20rrf_llm · Kira+Cael RRF融合"),
    ]

    tk_all = {}
    for fk, label in TOPK_BASE + TOPK_CORE:
        r = eval_topk(fk)
        if r:
            tk_all[fk] = {"label": label, **r}

    # ── TopK 消融: 图谱变体 ──
    GRAPH_ABL = [
        ("v20graph",        "Cael 纯图谱 (pairwise logistic)"),
        ("v20graph_ppr",    "+ PPR 提纯候选池"),
        ("v20graph_fam",    "+ 同家族特征扩展"),
        ("v20graph_cap32",  "候选池截断 cap=32"),
        ("v20graph_cap48",  "候选池截断 cap=48"),
        ("v20full",         "Cael 全特征 cap=16"),
    ]
    # ── TopK 消融: Kira / RRF ──
    KIRA_ABL = [
        ("v20pergene",       "Kira 纯LLM 逐基因 1-5 评分"),
        ("v20rrf_pergene",   "RRF pergene+graph"),
        ("v20rrf_llm",       "RRF Kira+Cael (k=60)"),
    ]
    # ── TopK: LLM Pool vs Graph Pool ──
    POOL_CMP = [
        ("v20graph",     "v20graph_llmpool",    "Cael 图谱"),
        ("v20graph_ppr", "v20graph_ppr_llmpool","Cael 图谱+PPR"),
        ("v20rrf_llm",   "v20rrf_llm_llmpool",  "RRF Kira+Cael"),
    ]

    # ═══════════════════════════════════════════════════════════
    width = 135
    sep = "=" * width

    print(sep)
    print("  v20 版本 · 全量样本 · 三任务 (Membership + Pairwise + TopK) 性能 & Token + 消融")
    print(sep)

    # ──────────────────────────────────────
    #  表1: 全量三任务总表
    # ──────────────────────────────────────
    print(f"\n  {'─'*120}")
    print(f"  表1 · 全量性能总表（全部基线 + v20 管线）")
    print(f"  {'─'*120}")

    # Membership
    print(f"\n  ▸ Membership (二分类: 基因是否属指定功能类)")
    print(MP_HEADER)
    print(MP_SEP)
    for s in base_mp:
        print(fmt_mp_row(f"  [基线] {s['label']}", s["m"], s["tok"]))
    print(MP_SEP)
    print(fmt_mp_row("  v20 · 管线启发式 (heuristic)", v20_heur["m"], 0))
    print(fmt_mp_row("  v20 · 管线auto类比 (LLM兜底)", v20_auto["m"], 0))

    # Pairwise
    print(f"\n  ▸ Pairwise (二选一: 哪个基因更可能为目标基因)")
    print(MP_HEADER)
    print(MP_SEP)
    for s in base_mp:
        print(fmt_mp_row(f"  [基线] {s['label']}", s["p"], s["tok"]))
    print(MP_SEP)
    print(fmt_mp_row("  v20 · 管线启发式 (heuristic)", v20_heur["p"], 0))
    print(fmt_mp_row("  v20 · 管线auto类比 (LLM兜底)", v20_auto["p"], 0))

    # TopK
    sota_ndcg = max(t["ndcg"] for t in tk_all.values())
    print(f"\n  ▸ TopK (CRISPR SLC top-5 排序, NDCG@5 / hit@5, 11题)")
    print(f"  {'系统':<44s} {'N':>4s} {'NDCG@5':>7s} {'hit@5':>7s} {'ΔSOTA':>7s} {'Token':>8s}")
    print(f"  {'─'*84}")
    for fk, _ in TOPK_BASE:
        if fk not in tk_all: continue
        t = tk_all[fk]
        d = t["ndcg"] - sota_ndcg
        print(f"  [基线] {t['label']:<38s} {t['n']:>4d} {t['ndcg']:>7.4f} {t['hit']:>7.3f} {d:>+7.4f} {ftok(t['tok']):>8s}")
    print(f"  {'─'*84}")
    for fk, _ in TOPK_CORE:
        if fk not in tk_all: continue
        t = tk_all[fk]
        d = t["ndcg"] - sota_ndcg
        mark = " ★" if abs(d) < 0.001 else ""
        tok_s = ftok(t["tok"]) if t["tok"] > 0 else ("perGene" if "pergene" in fk else "0")
        print(f"  [v20] {t['label']:<39s} {t['n']:>4d} {t['ndcg']:>7.4f} {t['hit']:>7.3f} {d:>+7.4f} {tok_s:>8s}{mark}")

    # ═══════════════════════════════════════════════════════════
    #  表2: 消融实验
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n  {'─'*120}")
    print(f"  表2 · 消融实验")
    print(f"  {'─'*120}")

    # ── 2a: M/P 模式消融 ──
    print(f"\n  ▸ 2a · M/P 模式消融: heuristic(纯证据) vs auto(LLM类比兜底)")
    print(f"  {'系统':<28s} {'任务':<12s} {'N':>5s} {'对':>5s} {'错':>5s} {'弃':>4s} {'覆盖':>6s} {'准确(答)':>8s} {'准确(全)':>8s} {'净胜':>6s}")
    print(f"  {'─'*105}")
    for label, data in [("v20·启发式(heuristic)", v20_heur), ("v20·auto类比(LLM兜底)", v20_auto)]:
        for task, tup in [("membership", data["m"]), ("pairwise", data["p"]), ("OVERALL", data["all"])]:
            n, c, w, a, cov, acc_ans, acc_all = tup
            net = acc_all - 0.5
            print(f"  {label:<28s} {task:<12s} {n:>5d} {c:>5d} {w:>5d} {a:>4d} {cov*100:>5.1f}% {acc_ans*100:>7.1f}% {acc_all*100:>7.1f}% {net:>+6.1%}")

    # ── 2b: TopK 计算智能体 (Cael) 消融 ──
    print(f"\n  ▸ 2b · TopK 消融: 计算智能体 (Cael) 组件增益")
    print(f"  {'配置':<44s} {'NDCG@5':>7s} {'hit@5':>7s} {'vs Cael-only':>12s} {'Token':>8s}")
    print(f"  {'─'*85}")
    cael_base = None
    for fk, label in GRAPH_ABL:
        r = eval_topk(fk)
        if not r: continue
        if cael_base is None:
            cael_base = r["ndcg"]
        delta = r["ndcg"] - cael_base
        delta_s = f"{delta:+.4f}" if delta != 0 else "baseline"
        tok_s = ftok(r["tok"]) if r["tok"] > 0 else "0"
        print(f"  {label:<44s} {r['ndcg']:>7.4f} {r['hit']:>7.3f} {delta_s:>12s} {tok_s:>8s}")

    # ── 2c: TopK 知识智能体 (Kira) 消融 ──
    print(f"\n  ▸ 2c · TopK 消融: 知识智能体 (Kira) 组件增益")
    print(f"  {'配置':<44s} {'NDCG@5':>7s} {'hit@5':>7s} {'vs Cael-only':>12s} {'vs Kira-only':>12s} {'Token':>8s}")
    print(f"  {'─'*100}")
    kira_base = None
    for fk, label in KIRA_ABL:
        r = eval_topk(fk)
        if not r: continue
        if fk == "v20pergene" and kira_base is None:
            kira_base = r["ndcg"]
        vs_cael = f"{r['ndcg'] - cael_base:+.4f}" if cael_base else "--"
        vs_kira = f"{r['ndcg'] - kira_base:+.4f}" if kira_base else "--"
        if fk == "v20pergene": vs_kira = "baseline"
        tok_s = ftok(r["tok"]) if r["tok"] > 0 else ("perGene" if "pergene" in fk else "0")
        print(f"  {label:<44s} {r['ndcg']:>7.4f} {r['hit']:>7.3f} {vs_cael:>12s} {vs_kira:>12s} {tok_s:>8s}")

    # ── 2d: TopK 外部基线对比 ──
    print(f"\n  ▸ 2d · TopK 消融: 外部基线 (不同Agent框架)")
    print(f"  {'系统':<44s} {'NDCG@5':>7s} {'hit@5':>7s} {'ΔSOTA':>7s} {'Token':>8s}")
    print(f"  {'─'*80}")
    for fk, _ in TOPK_BASE:
        if fk not in tk_all: continue
        t = tk_all[fk]
        d = t["ndcg"] - sota_ndcg
        print(f"  {t['label']:<44s} {t['ndcg']:>7.4f} {t['hit']:>7.3f} {d:>+7.4f} {ftok(t['tok']):>8s}")

    # ── 2e: LLM Pool vs Graph Pool 候选池对比 ──
    print(f"\n  ▸ 2e · TopK 消融: 候选池来源对比 (知识图谱 vs LLM生成)")
    print(f"  {'模型':<20s} {'图谱池 (PPR∪GT)':>14s} {'LLM生成池':>12s} {'Δ (LLM-图谱)':>13s} {'倍数':>7s}")
    print(f"  {'─'*72}")
    for graph_fk, pool_fk, label in POOL_CMP:
        gr = eval_topk(graph_fk)
        pr = eval_topk(pool_fk)
        if not gr or not pr: continue
        delta = pr["ndcg"] - gr["ndcg"]
        ratio = pr["ndcg"] / gr["ndcg"] if gr["ndcg"] > 0 else 0
        print(f"  {label:<20s} {gr['ndcg']:>14.4f} {pr['ndcg']:>12.4f} {delta:>+13.4f} {ratio:>6.1f}x")

    # ── 注记 ──
    print(f"\n  {sep}")
    print(f"  注:")
    print(f"    · v20 管线 N=489 (membership 230 + pairwise 259), 外部基线 N=489~631 (运行覆盖不同)。")
    print(f"    · TopK 均为 11 题 (同一套 CRISPR SLC top-5 排序题), PPR∪GT 候选池。")
    print(f"    · Token: perGene=逐基因LLM评分(~18-20次/题); 0=纯图谱/规则无LLM; --=管线无独立token统计。")
    print(f"    · M/P 消融: heuristic=纯证据打分(可弃答), auto=证据不足时LLM跨癌种类推(尽量不弃答)。")
    print(f"    · v20 核心改进: R2 membership 必需性重排 (仅影响 membership 兜底路径, pairwise 不动)。")
    print(f"    · LLM Pool 变体使用 LLM 自行生成的候选池(非 PPR∪GT),用于说明 LLM 生成候选池的劣势。")
    print(f"  {sep}")


if __name__ == "__main__":
    main()