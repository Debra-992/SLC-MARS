# -*- coding: utf-8 -*-
"""TopK 全部系统性能 & 消融 & Token 消耗 — 一张汇总表。

用法: .venv\Scripts\python.exe tools\_topk_master_table.py
"""
from __future__ import annotations
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "outputs" / "traces"


def load(name: str):
    p = TRACES / f"topk_eval_{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    ok = [r for r in d["results"] if not r.get("error")]
    if not ok:
        return None
    ndcg = sum(r["ndcg5"] for r in ok) / len(ok)
    hit = sum(r["hit5"] for r in ok) / len(ok)
    tok = sum(r.get("tokens", {}).get("total", 0) for r in d["results"])
    cap = d.get("cap", "")
    return ndcg, hit, tok, cap


# ── 主表定义 ──
ROWS = [
    # (section, label, file_key, note)
    ("── 消融 · 知识智能体 (Kira/LLM) ──", "", "", ""),
    ("", "v20pergene · Kira 纯 LLM 逐基因 1-5 评分", "v20pergene", "← SOTA"),

    ("── 消融 · 计算智能体 (Cael/Graph) ──", "", "", ""),
    ("", "v20graph · Cael 纯图谱 pairwise logistic", "v20graph", ""),
    ("", "v20graph_ppr · +PPR 提纯候选池", "v20graph_ppr", ""),
    ("", "v20graph_fam · +同家族特征扩展", "v20graph_fam", ""),
    ("", "v20graph_cap32 · 候选池截断 32", "v20graph_cap32", ""),
    ("", "v20graph_cap48 · 候选池截断 48", "v20graph_cap48", ""),

    ("── 消融 · 融合 (Kira+Cael RRF) ──", "", "", ""),
    ("", "v20rrf_pergene · RRF k=60 (pergene+graph)", "v20rrf_pergene", ""),
    ("", "v20rrf_pergene_purellm · RRF (pure_llm+graph)", "v20rrf_pergene_purellm", ""),
    ("", "v20rrf_llm · RRF LLM-only 通道", "v20rrf_llm", ""),
    ("", "v20rrf_llm_full · RRF LLM full 通道", "v20rrf_llm_full", ""),
    ("", "v20full · Cael 全特征 (cap=16)", "v20full", ""),
    ("", "v20full_ext · Cael 全特征扩展", "v20full_ext", ""),

    ("── 外部基线 · 排序头 ──", "", "", ""),
    ("", "pure_llm · 直接 LLM 全池乱序排序", "pure_llm", ""),
    ("", "pipeline · PPR 图谱序 → LLM 重排", "pipeline", ""),
    ("", "MetaGPT · Role/Action + LLM 单轮", "metagpt", ""),
    ("", "AutoGen · 2-turn self-critique", "autogen", ""),
    ("", "MDAgents · adaptive multi-agent debate", "mdagents", ""),

    ("── LLM Pool 变体 (不同候选池) ──", "", "", ""),
    ("", "pure_llm_llmpool · LLM 生成池", "pure_llm_llmpool", ""),
    ("", "pipeline_llmpool · LLM 生成池 + PPR", "pipeline_llmpool", ""),
    ("", "v20graph_llmpool · 图谱 on LLM 池", "v20graph_llmpool", ""),
    ("", "v20graph_ppr_llmpool · 图谱+PPR on LLM 池", "v20graph_ppr_llmpool", ""),
    ("", "v20rrf_llm_llmpool · RRF-LLM on LLM 池", "v20rrf_llm_llmpool", ""),
]


def fmt_tok(tok: int, file_key: str) -> str:
    if tok > 0:
        if tok >= 1_000_000:
            return f"{tok/1_000_000:.2f}M"
        elif tok >= 1_000:
            return f"{tok/1_000:.1f}k"
        return str(tok)
    if file_key.startswith("v20pergene") or file_key.startswith("v20rrf_pergene"):
        return "perGene"
    if file_key.startswith("v20graph") or file_key.startswith("v20full") or file_key.startswith("v20rrf"):
        return "0"
    return "--"


def main():
    sota_r = load("v20pergene")
    sota_ndcg = sota_r[0] if sota_r else 0

    sep = "─" * 95
    header = f"{'系统':<48s} {'NDCG@5':>7s} {'hit@5':>7s} {'Δ SOTA':>8s} {'Token':>9s}  备注"
    print(sep)
    print("  TopK 全部系统 · 消融 & 基线 · 性能 & Token 消耗 一览")
    print(f"  (11 题 CRISPR SLC top-5 排序, PPR∪GT 候选池; SOTA = v20pergene {sota_ndcg:.4f})")
    print(sep)
    print(header)
    print(sep)

    for sec, label, fk, note in ROWS:
        if not fk:
            if sec:
                print(f"\n  \033[1;36m{sec}\033[0m")
            continue
        r = load(fk)
        if not r:
            print(f"  {label:<48s} {'--':>7s} {'--':>7s} {'--':>8s} {'--':>9s}  (未跑)")
            continue
        ndcg, hit, tok, cap = r
        delta = ndcg - sota_ndcg
        delta_s = f"{delta:+.4f}"
        tok_s = fmt_tok(tok, fk)
        mark = " ★" if note else ""
        cap_s = f" cap={cap}" if cap else ""
        print(f"  {label:<48s} {ndcg:>7.4f} {hit:>7.3f} {delta_s:>8s} {tok_s:>9s}{mark}{cap_s}")

    print(sep)
    print(f"  Token 说明: 0 = 纯图谱/规则模型无LLM调用; perGene = ~18-20次独立基因评分调用/题,不可比")
    print(f"  LLM Pool 变体使用不同候选池(LLM生成),与主表候选池(PPR∪GT)不可直接比")
    print(sep)


if __name__ == "__main__":
    main()