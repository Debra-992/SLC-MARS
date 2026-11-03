# -*- coding: utf-8 -*-
"""v20 · 一张大表(全系统×全任务) + 一张消融大表(统一格式)。

用法: .venv\Scripts\python.exe tools\_v20_big_table.py
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
#  M/P: 答案 JSONL
# ═══════════════════════════════════════════════════════════════
def eval_answer_file(name: str):
    path = TRACES / f"answers_{name}.jsonl"
    if not path.exists(): return None
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().splitlines()]
    by_qid = {r["qid"]: r for r in lines}
    buckets = {"membership": [], "pairwise": []}
    total_tok = 0
    for qid, ans in by_qid.items():
        gt = GT.get(qid)
        if not gt: continue
        b = gt.get("task_bucket", "")
        if b not in buckets: continue
        total_tok += ans.get("tokens", {}).get("total", 0)
        if ans.get("abstained"):
            buckets[b].append("abstain")
        elif b == "membership":
            buckets[b].append("correct" if ans.get("answer") == gt["ground_truth"]["answer"] else "wrong")
        elif b == "pairwise":
            exp = gt["ground_truth"]["winner"]; got = ans.get("winner")
            buckets[b].append("correct" if (got and got.upper() == exp.upper()) else ("wrong" if got else "abstain"))
    def _m(vals):
        c=vals.count("correct"); w=vals.count("wrong"); a=vals.count("abstain")
        n=c+w+a; ans_n=c+w
        return (n, c, w, a, ans_n/n if n else 0, c/ans_n if ans_n else 0, c/n if n else 0)
    return {"m": _m(buckets["membership"]), "p": _m(buckets["pairwise"]), "tok": total_tok}


# ═══════════════════════════════════════════════════════════════
#  Pipeline (tracestore.jsonl) by version
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
            buckets[b].append(e.get("systems", {}).get(sys_key, {}).get("outcome", "abstain"))
        def _m(vals):
            c=vals.count("correct"); w=vals.count("wrong"); a=vals.count("abstain")
            n=c+w+a; ans_n=c+w
            return (n, c, w, a, ans_n/n if n else 0, c/ans_n if ans_n else 0, c/n if n else 0)
        result[sys_key] = {"m": _m(buckets["membership"]), "p": _m(buckets["pairwise"])}
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
    return {"ndcg": sum(r["ndcg5"] for r in ok) / len(ok),
            "hit": sum(r["hit5"] for r in ok) / len(ok),
            "tok": sum(r.get("tokens", {}).get("total", 0) for r in d["results"])}


# ═══════════════════════════════════════════════════════════════
#  智能体消融 (without Kira / without Cael) — M/P from ablation JSONs
# ═══════════════════════════════════════════════════════════════
def _outcome_stats(outs):
    c = outs.count("correct"); w = outs.count("wrong"); a = outs.count("abstain")
    n = c + w + a; ans = c + w
    return (n, c, w, a, ans / n if n else 0, c / ans if ans else 0, c / n if n else 0)


def eval_agent_ablation(mode: str, sys_key: str = "pipeline_auto"):
    """读消融 JSON → {membership/pairwise/overall: stats, qids}。
    mode: ablate_knowledge=without Kira(仅Cael); ablate_compute=without Cael(仅Kira)。"""
    path = TRACES / f"ablation_{mode}_n100_seed5.json"
    if not path.exists(): return None
    d = json.loads(path.read_text(encoding="utf-8"))
    buckets = {"membership": [], "pairwise": []}
    qids = []
    for r in d["results"]:
        b = r.get("bucket", "")
        qids.append(r["qid"])
        if b in buckets:
            buckets[b].append(r.get("systems", {}).get(sys_key, {}).get("outcome", "abstain"))
    return {"membership": _outcome_stats(buckets["membership"]),
            "pairwise": _outcome_stats(buckets["pairwise"]),
            "overall": _outcome_stats(buckets["membership"] + buckets["pairwise"]),
            "qids": qids}


def v20_baseline_on_qids(qids, sys_key: str = "pipeline_auto"):
    """从 tracestore(trace_version=20) 取指定 qids 的 v20 完整基线, 按 bucket 分组。"""
    store_path = TRACES / "tracestore.jsonl"
    if not store_path.exists(): return None
    latest: dict[str, dict] = {}
    for line in store_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if str((r.get("provenance") or {}).get("trace_version") or "") == "20":
            latest[r["qid"]] = r
    buckets = {"membership": [], "pairwise": []}
    for qid in qids:
        e = latest.get(qid)
        if not e: continue
        b = e.get("bucket", "")
        if b in buckets:
            buckets[b].append(e.get("systems", {}).get(sys_key, {}).get("outcome", "abstain"))
    return {"membership": _outcome_stats(buckets["membership"]),
            "pairwise": _outcome_stats(buckets["pairwise"]),
            "overall": _outcome_stats(buckets["membership"] + buckets["pairwise"])}


# ═══════════════════════════════════════════════════════════════
#  Token formatting
# ═══════════════════════════════════════════════════════════════
def ftok(t):
    if t is None or t <= 0: return "--"
    if t >= 1_000_000: return f"{t / 1_000_000:.1f}M"
    return f"{t / 1_000:.0f}k"


# ═══════════════════════════════════════════════════════════════
def main():
    # ── 收集全部数据 ──
    systems = []  # (label, category, m, p, topk, m_tok, p_tok, t_tok_display)

    # 外部基线
    BASE = [
        ("pure_llm",  "pure_llm"),
        ("lit_rag",   "LitRAG"),
        ("metagpt",   "MetaGPT"),
        ("autogen",   "AutoGen"),
        ("mdagents",  "MDAgents"),
        ("pipeline",  "pipeline·PPR→LLM"),
    ]
    for fname, label in BASE:
        mp = eval_answer_file(fname)
        tk = eval_topk(fname)
        if mp is None and tk is None: continue
        m = mp["m"] if mp else None
        p = mp["p"] if mp else None
        m_tok = mp["tok"] if mp else None
        p_tok = mp["tok"] if mp else None
        systems.append((label, "基线", m, p, tk, m_tok, p_tok, None))

    # v20 管线 (M/P)
    v20p = eval_pipeline("20")
    for sys_key, label in [("pipeline_heuristic", "v20·管线heuristic"), ("pipeline_auto", "v20·管线auto")]:
        if sys_key in v20p:
            d = v20p[sys_key]
            systems.append((label, "v20管线", d["m"], d["p"], None, None, None, None))

    # v20 TopK 核心系统
    V20_TOPK = [
        ("v20pergene",   "v20pergene·Kira"),
        ("v20graph_ppr", "v20graph_ppr·Cael+PPR"),
        ("v20graph",     "v20graph·Cael"),
        ("v20rrf_llm",   "v20rrf_llm·RRF融合"),
    ]
    for fk, label in V20_TOPK:
        tk = eval_topk(fk)
        if tk:
            t_tok_s = ftok(tk["tok"]) if tk["tok"] > 0 else ("perGene" if "pergene" in fk else "0")
            systems.append((label, "v20TopK", None, None, tk, None, None, t_tok_s))

    # v20 TopK LLM Pool 变体
    LLM_POOL = [
        ("v20graph_llmpool",     "v20graph·LLM池"),
        ("v20graph_ppr_llmpool", "v20graph_ppr·LLM池"),
        ("v20rrf_llm_llmpool",   "v20rrf_llm·LLM池"),
    ]
    for fk, label in LLM_POOL:
        tk = eval_topk(fk)
        if tk:
            systems.append((label, "LLM池", None, None, tk, None, None, "0"))

    # ═══════════════════════════════════════════════════════════
    #  表1: 全系统大表
    # ═══════════════════════════════════════════════════════════
    width = 145
    sep = "=" * width

    print(sep)
    print("  表1 · v20 全系统 × 全任务 (Membership + Pairwise + TopK) · 性能 & Token 消耗 一览大表")
    print(sep)
    print(f"  {'系统':<26s} {'类别':<8s} "
          f"{'M·N':>4s} {'M·Acc':>7s} {'M·Net':>6s} {'M·Tok':>7s}  "
          f"{'P·N':>4s} {'P·Acc':>7s} {'P·Net':>6s} {'P·Tok':>7s}  "
          f"{'T·NDCG':>7s} {'T·hit':>6s} {'T·Tok':>7s}")
    print(f"  {'─'*136}")

    for label, cat, m, p, tk, m_tok, p_tok, t_tok_override in systems:
        if m:
            mn, mc, mw, ma, mcov, mac_ans, mac_all = m
            m_s = f"{mn:>4d} {mac_all*100:>6.1f}% {mac_all-0.5:>+6.1%} {ftok(m_tok):>7s}"
        else:
            m_s = f"{'--':>4s} {'--':>7s} {'--':>6s} {'--':>7s}"
        if p:
            pn, pc, pw, pa, pcov, pac_ans, pac_all = p
            p_s = f"{pn:>4d} {pac_all*100:>6.1f}% {pac_all-0.5:>+6.1%} {ftok(p_tok):>7s}"
        else:
            p_s = f"{'--':>4s} {'--':>7s} {'--':>6s} {'--':>7s}"
        if tk:
            t_tok_s = t_tok_override if t_tok_override else ftok(tk["tok"])
            t_s = f"{tk['ndcg']:>7.4f} {tk['hit']:>6.3f} {t_tok_s:>7s}"
        else:
            t_s = f"{'--':>7s} {'--':>6s} {'--':>7s}"
        print(f"  {label:<26s} {cat:<8s} {m_s}  {p_s}  {t_s}")

    # ═══════════════════════════════════════════════════════════
    #  表2: 消融大表 (统一格式)
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{sep}")
    print("  表2 · v20 消融实验大表 (统一格式)")
    print(sep)
    print(f"  {'消融配置':<32s} {'消融维度':<14s} "
          f"{'M·N':>4s} {'M·Acc':>7s} {'M·Net':>6s}  "
          f"{'P·N':>4s} {'P·Acc':>7s} {'P·Net':>6s}  "
          f"{'T·NDCG':>7s} {'T·hit':>6s} {'Δbase':>7s} {'Token':>7s}")
    print(f"  {'─'*130}")

    # ── M/P 模式消融 ──
    for label, data in [("v20·heuristic (纯证据打分)", v20p.get("pipeline_heuristic", {})),
                         ("v20·auto (LLM类比兜底)", v20p.get("pipeline_auto", {}))]:
        m = data.get("m"); p = data.get("p")
        if m:
            mn, mc, mw, ma, mcov, mac_ans, mac_all = m
            m_s = f"{mn:>4d} {mac_all*100:>6.1f}% {mac_all-0.5:>+6.1%}"
        else:
            m_s = f"{'--':>4s} {'--':>7s} {'--':>6s}"
        if p:
            pn, pc, pw, pa, pcov, pac_ans, pac_all = p
            p_s = f"{pn:>4d} {pac_all*100:>6.1f}% {pac_all-0.5:>+6.1%}"
        else:
            p_s = f"{'--':>4s} {'--':>7s} {'--':>6s}"
        print(f"  {label:<32s} {'M/P模式':<14s} {m_s}  {p_s}  {'--':>7s} {'--':>6s} {'--':>7s} {'--':>7s}")

    # ── TopK: Cael 组件消融 ──
    CAEL_ABL = [
        ("v20graph",       "Cael 纯图谱 (baseline)"),
        ("v20graph_ppr",   "Cael + PPR 提纯"),
        ("v20graph_fam",   "Cael + 同家族特征"),
        ("v20graph_cap32", "Cael 候选池 cap=32"),
        ("v20graph_cap48", "Cael 候选池 cap=48"),
        ("v20full",        "Cael 全特征 cap=16"),
    ]
    cael_base = None
    for fk, label in CAEL_ABL:
        r = eval_topk(fk)
        if not r: continue
        if cael_base is None: cael_base = r["ndcg"]
        delta = r["ndcg"] - cael_base
        delta_s = f"{delta:+.4f}" if delta != 0 else "baseline"
        tok_s = ftok(r["tok"]) if r["tok"] > 0 else "0"
        print(f"  {label:<32s} {'Cael组件':<14s} {'--':>4s} {'--':>7s} {'--':>6s}  {'--':>4s} {'--':>7s} {'--':>6s}  {r['ndcg']:>7.4f} {r['hit']:>6.3f} {delta_s:>7s} {tok_s:>7s}")

    # ── TopK: Kira 组件消融 ──
    KIRA_ABL = [
        ("v20pergene",     "Kira 纯LLM逐基因 (SOTA)"),
        ("v20rrf_pergene", "RRF pergene+graph"),
        ("v20rrf_llm",     "RRF Kira+Cael (k=60)"),
    ]
    kira_base = None
    for fk, label in KIRA_ABL:
        r = eval_topk(fk)
        if not r: continue
        if kira_base is None and fk == "v20pergene": kira_base = r["ndcg"]
        vs_cael = f"{r['ndcg'] - cael_base:+.4f}" if cael_base else "--"
        tok_s = ftok(r["tok"]) if r["tok"] > 0 else ("perGene" if "pergene" in fk else "0")
        print(f"  {label:<32s} {'Kira组件':<14s} {'--':>4s} {'--':>7s} {'--':>6s}  {'--':>4s} {'--':>7s} {'--':>6s}  {r['ndcg']:>7.4f} {r['hit']:>6.3f} {vs_cael:>7s} {tok_s:>7s}")

    # ── TopK: 候选池来源消融 ──
    POOL_CMP = [
        ("v20graph",     "v20graph_llmpool",     "Cael 图谱池 (PPR∪GT)"),
        ("v20graph_ppr", "v20graph_ppr_llmpool", "Cael+PPR 图谱池"),
        ("v20rrf_llm",   "v20rrf_llm_llmpool",   "RRF 图谱池"),
    ]
    for gfk, pfk, label in POOL_CMP:
        gr = eval_topk(gfk); pr = eval_topk(pfk)
        if not gr or not pr: continue
        # 图谱池行
        tok_s = ftok(gr["tok"]) if gr["tok"] > 0 else "0"
        print(f"  {label:<32s} {'候选池来源':<14s} {'--':>4s} {'--':>7s} {'--':>6s}  {'--':>4s} {'--':>7s} {'--':>6s}  {gr['ndcg']:>7.4f} {gr['hit']:>6.3f} {'baseline':>7s} {tok_s:>7s}")
        # LLM池行
        delta = pr["ndcg"] - gr["ndcg"]
        print(f"  {'  └─ LLM自行生成池':<32s} {'候选池来源':<14s} {'--':>4s} {'--':>7s} {'--':>6s}  {'--':>4s} {'--':>7s} {'--':>6s}  {pr['ndcg']:>7.4f} {pr['hit']:>6.3f} {delta:>+7.4f} {'0':>7s}")

    # ═══════════════════════════════════════════════════════════
    #  表3: 智能体消融 (without Kira / without Cael) 三任务统一对比
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{sep}")
    print("  表3 · v20 智能体消融 (without Kira / without Cael) · 三任务统一对比")
    print(sep)

    abl_no_kira = eval_agent_ablation("ablate_knowledge")  # without Kira → 仅 Cael
    abl_no_cael = eval_agent_ablation("ablate_compute")    # without Cael → 仅 Kira
    base_qids = (abl_no_kira or abl_no_cael or {}).get("qids", [])
    v20_base = v20_baseline_on_qids(base_qids)

    # TopK 三臂映射: full=RRF融合, without Kira=Cael graph+PPR, without Cael=Kira pergene
    tk_full = eval_topk("v20rrf_llm")
    tk_no_kira = eval_topk("v20graph_ppr")
    tk_no_cael = eval_topk("v20pergene")

    # 基线值 (用于算 Δ)
    b_m = v20_base["membership"][6] if v20_base else 0
    b_p = v20_base["pairwise"][6] if v20_base else 0
    b_all = v20_base["overall"][6] if v20_base else 0
    b_ndcg = tk_full["ndcg"] if tk_full else 0

    print(f"\n  ▸ 三任务统一表 (M/P: n=100 seed=5 auto决策, 复用v20供给逐题屏蔽单智能体; TopK: 11题独立排序系统映射)")
    print(f"  {'配置':<26s} {'M·Acc':>7s} {'ΔM':>7s}  {'P·Acc':>7s} {'ΔP':>7s}  {'ALL·Acc':>8s} {'ΔALL':>7s}  "
          f"{'T·NDCG':>7s} {'T·hit':>6s} {'ΔT':>8s}  {'T·Token':>8s}")
    print(f"  {'─'*118}")

    rows3 = [
        ("v20 full (Kira+Cael)", v20_base, tk_full, "--", True),
        ("without Kira (仅Cael计算)", abl_no_kira, tk_no_kira, "0", False),
        ("without Cael (仅Kira知识)", abl_no_cael, tk_no_cael, "perGene", False),
    ]
    for label, mp, tk, tok_s, is_base in rows3:
        if not mp: continue
        m_acc = mp["membership"][6]; p_acc = mp["pairwise"][6]; all_acc = mp["overall"][6]
        if is_base:
            dm = dp = dall = "baseline"
        else:
            dm = f"{(m_acc - b_m)*100:+.1f}%"
            dp = f"{(p_acc - b_p)*100:+.1f}%"
            dall = f"{(all_acc - b_all)*100:+.1f}%"
        if tk:
            ndcg = tk["ndcg"]; hit = tk["hit"]
            dt = "baseline" if is_base else f"{ndcg - b_ndcg:+.4f}"
            t_s = f"{ndcg:>7.4f} {hit:>6.3f} {dt:>8s}"
        else:
            t_s = f"{'--':>7s} {'--':>6s} {'--':>8s}"
        print(f"  {label:<26s} {m_acc*100:>6.1f}% {dm:>7s}  {p_acc*100:>6.1f}% {dp:>7s}  "
              f"{all_acc*100:>7.1f}% {dall:>7s}  {t_s}  {tok_s:>8s}")

    # ── 注记 ──
    print(f"\n  {sep}")
    print(f"  注:")
    print(f"    · Δbase: Cael组件以'Cael纯图谱'为baseline; Kira组件以'Cael纯图谱'为参照; 候选池以图谱池为baseline。")
    print(f"    · M/P模式消融: heuristic=纯证据确定性打分(可弃答), auto=证据不足时LLM跨癌种类推(尽量不弃答)。")
    print(f"    · 表3 M/P: 消融臂复用v20供给(kg/ck)后清空单智能体, 仅重跑Vega决策; 不含prior投票(正交机制)。")
    print(f"    · 表3 M/P baseline=v20冠军(含prior), 与消融臂(无prior)对比 Δ略保守; n=100与全量N=489不可直接比。")
    print(f"    · 表3 TopK: 三臂为独立排序系统映射(非复用供给消融); full=RRF融合, without Kira=Cael graph+PPR, without Cael=Kira pergene。")
    print(f"    · 关键: M/P 去掉任一智能体都掉点(Kira更关键); 但 TopK 去掉Cael(Kira单独)反而SOTA——排序任务图谱计算稀释LLM语义判断。")
    print(f"    · Token: perGene=逐基因LLM评分(~18-20次/题); 0=纯图谱/规则无LLM; --=无独立token统计。")
    print(f"  {sep}")


if __name__ == "__main__":
    main()