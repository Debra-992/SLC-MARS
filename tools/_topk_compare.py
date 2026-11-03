# -*- coding: utf-8 -*-
"""topk 多系统对比汇总：读 topk_eval_<system>.json → 三臂对比表 + 分桶切片。

三臂：
  1. 纯PPR图谱基线   —— 不经过 LLM，直接拿 pack 里的 PPR 降序投影到候选池
                        （逐题 ppr 字段；图谱零提名的题 ppr=null，从基线汇总中剔除）
  2. 纯LLM裸排       —— topk_eval_pure_llm.json   (--hint none，整池乱序)
  3. 管线·PPR+LLM重排 —— topk_eval_pipeline.json  (--hint ppr_order，图谱序+LLM重排)

口径注意：LLM 臂用全部可答题，PPR 臂只用"图谱有提名"的题集；两者题目集不同，
故结论行会把 LLM 也 restrict 到 PPR 可测子集上再算 Δ（同题集才可比）。

切片：正例/负例（qid 前缀）× 物种（Human/Mouse/Synthetic construct，
Synthetic≈防记忆代理）。注意 00002/00003 是同一 synthetic 题的正/负反向，
候选池==GT 全 5 基因，PR@5 恒 1.0，只能读 NDCG/τ。
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "outputs" / "traces"

DISPLAY = {"pipeline": "管线·PPR+LLM重排", "pure_llm": "纯LLM裸排",
           "v20graph": "v20·图logistic排序头", "v20graph_ppr": "v20·图头+PPR融合",
           "v20full": "v20·全特征排序头(lever1)",
           "v20full_ext": "v20full+DESeq2旁路面板(B)",
           "v20graph_fam": "v20·图头+家族传播(lever3)",
           "v20graph_cap32": "v20·CAP=32",
           "v20graph_cap48": "v20·CAP=48",
           "v20rrf_llm": "v20·图+PPR+LLM融合",
           "v20rrf_llm_full": "v20·全LLM排序+RRF融合"}
METRICS = ("ndcg5", "kendall_tau", "precision5", "hit5", "mrr")


def load_systems() -> dict[str, dict]:
    out = {}
    for p in sorted(TRACES.glob("topk_eval_*.json")):
        sys_name = p.name[len("topk_eval_"):-len(".json")]
        if sys_name in ("dryrun", "pack"):
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("dry_run"):
            continue
        out[sys_name] = data
    return out


def mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def ppr_block_field(r: dict, key: str):
    return (r.get("ppr") or {}).get(key)


def slice_key(qid: str) -> tuple[str, str]:
    pol = "positive" if "_positive_" in qid else "negative"
    for sp in ("Human", "Mouse", "Synthetic"):
        if f"_{sp}" in qid or f"_{sp}_" in qid:
            return pol, sp
    return pol, "?"


def main():
    systems = load_systems()
    if not systems:
        raise SystemExit("没有 topk_eval_<system>.json，先跑 _topk_eval.py")

    ref = next(iter(systems.values()))
    ref_ok = [r for r in ref["results"] if not r["error"]]
    # 图谱基线可测题集（零提名题剔除）
    ppr_measurable = [r["qid"] for r in ref_ok if r.get("ppr")]

    print("=== topk 三臂对比（候选池 = PPR top-15 ∪ GT，同池同题）===")
    hdr = f"{'系统':<24s} {'n':>3s} " + " ".join(f"{m:>9s}" for m in
            ("NDCG@5", "Kendallτ", "PR@5", "hit@5", "MRR")) + "  token"
    print(hdr)
    print("-" * len(hdr))

    # 1) 纯图谱基线（只在可测题集上）
    ppr_rows = [r for r in ref_ok if r.get("ppr")]
    line = f"{'纯PPR图谱基线':<23s} {len(ppr_rows):>3d} "
    for m in METRICS:
        line += f" {mean([{m: ppr_block_field(r, m)} for r in ppr_rows], m):>+8.3f}" if m == "kendall_tau" \
            else f" {mean([{m: ppr_block_field(r, m)} for r in ppr_rows], m):>8.3f}"
    print(line + "  0")

    # 2) 各 LLM 臂（全部可答题）
    for sys_name, data in systems.items():
        ok = [r for r in data["results"] if not r["error"]]
        line = f"{DISPLAY.get(sys_name, sys_name):<23s} {len(ok):>3d} "
        for m in METRICS:
            v = mean(ok, m)
            line += f" {v:>+8.3f}" if m == "kendall_tau" else f" {v:>8.3f}"
        tok = sum(r["tokens"].get("total", 0) for r in data["results"])
        print(line + f"  {tok}")

    # 3) 同题集对齐：LLM 臂 restrict 到图谱可测子集，再算 Δ
    print(f"\n=== 同题集对齐（仅图谱可测 {len(ppr_measurable)}/11 题）===")
    line = f"{'纯PPR图谱基线':<23s} {len(ppr_rows):>3d} "
    for m in METRICS:
        v = mean([{m: ppr_block_field(r, m)} for r in ppr_rows], m)
        line += f" {v:>+8.3f}" if m == "kendall_tau" else f" {v:>8.3f}"
    print(line)
    for sys_name, data in systems.items():
        ok = [r for r in data["results"] if not r["error"] and r["qid"] in ppr_measurable]
        line = f"{DISPLAY.get(sys_name, sys_name):<23s} {len(ok):>3d} "
        for m in METRICS:
            v = mean(ok, m)
            line += f" {v:>+8.3f}" if m == "kendall_tau" else f" {v:>8.3f}"
        print(line)
        d_ndcg = mean(ok, "ndcg5") - mean([{ "ndcg5": ppr_block_field(r, "ndcg5")} for r in ppr_rows], "ndcg5")
        print(f"    → vs 图谱基线 ΔNDCG@5 = {d_ndcg:+.3f}")

    # 逐题明细
    print("\n=== 逐题 NDCG@5 ===")
    head = f"{'qid':<28s} {'pool':>4s} {'图谱提名':>6s} {'PPR':>7s}"
    for sys_name in systems:
        head += f" {DISPLAY.get(sys_name, sys_name):>14s}"
    print(head)
    for r in ref["results"]:
        ppr_s = f"{r['ppr']['ndcg5']:>7.3f}" if r.get("ppr") else "    N/A"
        line = f"{r['qid'].replace('topk_', ''):<28s} {r['pool_size']:>4d} {r.get('ppr_named', 0):>6d} {ppr_s}"
        for sys_name, data in systems.items():
            m = next((x for x in data["results"] if x["qid"] == r["qid"]), None)
            v = f"{m['ndcg5']:>14.3f}" if m and not m["error"] else f"{'ERR':>14s}"
            line += f" {v}"
        print(line)

    # 分桶切片
    print("\n=== 切片（正负例 × 物种；synthetic≈防记忆代理）===")
    keys = sorted({slice_key(r["qid"]) for r in ref["results"]})
    head = f"{'桶':<24s} {'n':>2s} {'PPR':>7s}"
    for sys_name in systems:
        head += f" {sys_name:>12s}"
    print(head)
    for pol, sp in keys:
        sub = [r for r in ref["results"] if slice_key(r["qid"]) == (pol, sp)]
        sub_ppr = [r for r in sub if r.get("ppr")]
        ppr_s = f"{mean([{'ndcg5': ppr_block_field(r, 'ndcg5')} for r in sub_ppr], 'ndcg5'):.3f}" if sub_ppr else "N/A"
        line = f"{pol + '/' + sp:<24s} {len(sub):>2d} {ppr_s:>7s}"
        for sys_name, data in systems.items():
            vals = [next(x for x in data["results"] if x["qid"] == r["qid"])["ndcg5"] for r in sub]
            line += f" {sum(vals)/len(vals):>12.3f}"
        print(line)


if __name__ == "__main__":
    # 直接自写 UTF-8，绕开 PowerShell 5.1 的 1> 重定向把 UTF-8 stdout 误按 GBK 解码再存 UTF-16 的坑
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    text = buf.getvalue()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    (TRACES / "topk_eval_summary.txt").write_text(text, encoding="utf-8")
