# -*- coding: utf-8 -*-
"""科学合规审查：剔除『处理实验(KO)基因 ≠ 提问基因』的不成立题目，另存干净数据集。

判定规则（唯一且可复核）：
  若某题 treatment 是单基因敲除实验 `GENE.KO vs WT`（即该实验只扰动了 GENE，
  ground-truth 由该 GENE 的敲除表型推出），而问题实际询问的基因（membership 的
  "Is <X> ..." / pairwise 的 "between <X> and <Y>"）不包含 GENE，
  则 (question, answer) 语义错位、科学上不成立 -> 删除。

输出：
  data/questions_clean.jsonl          干净题目
  evaluation/ground_truth_clean.jsonl 与之一一对应的答案
  data/questions_removed.jsonl        被删题目 + removal_reason（供审计追溯）
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Q_FULL = ROOT / "data" / "questions.jsonl"
G_FULL = ROOT / "evaluation" / "ground_truth.jsonl"
Q_CLEAN = ROOT / "data" / "questions_clean.jsonl"
G_CLEAN = ROOT / "evaluation" / "ground_truth_clean.jsonl"
Q_REMOVED = ROOT / "data" / "questions_removed.jsonl"

ASK = re.compile(r"\b(?:Is|Does)\s+(SLC[A-Z0-9]+)\b", re.I)
BETWEEN = re.compile(r"between\s+(SLC[A-Z0-9]+)\s+and\s+(SLC[A-Z0-9]+)", re.I)
# 只认 "SLCxxx.KO" 这类『单基因敲除 vs WT』的实验主体基因。
# p53.KO 等背景/时序筛选（T0 vs endpoint，全基因组分组）不算主体敲除，保留其 SLC 问题。
KOTOK = re.compile(r"\b(SLC[A-Z0-9]+)\.KO", re.I)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def asked_genes(rec: dict) -> set[str]:
    """问题真正在判定/选择的基因集合。topk 没有显式候选 -> 返回空集。"""
    q = rec["question"]
    bucket = rec.get("task_bucket", "")
    if bucket == "membership":
        return {g.upper() for g in ASK.findall(q)}
    if bucket == "pairwise":
        m = BETWEEN.findall(q)
        return {m[0][0].upper(), m[0][1].upper()} if m else set()
    return set()


def perturbed_genes(rec: dict) -> set[str]:
    """treatment 里被敲除/扰动的基因（GENE.KO）。"""
    return {g.upper() for g in KOTOK.findall(rec["question"])}


def main() -> None:
    qs = load_jsonl(Q_FULL)
    gt = {r["qid"]: r for r in load_jsonl(G_FULL)}

    keep: list[dict] = []
    removed: list[dict] = []
    for rec in qs:
        perturb = perturbed_genes(rec)
        asked = asked_genes(rec)
        reason = None
        if perturb and asked and not (perturb & asked):
            # 实验只敲打了 perturb，却在问 asked -> 真值无法对应所问基因
            reason = (f"treatment 扰动基因 {sorted(perturb)} 与提问基因 "
                      f"{sorted(asked)} 不一致：真值来自前者敲除实验，无法判定后者")
        if reason:
            r = dict(rec)
            r["removal_reason"] = reason
            removed.append(r)
        else:
            keep.append(rec)

    def dump(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump(Q_CLEAN, keep)
    dump(G_CLEAN, [gt[r["qid"]] for r in keep if r["qid"] in gt])
    dump(Q_REMOVED, removed)

    before = Counter(x["task_bucket"] for x in qs)
    after = Counter(x["task_bucket"] for x in keep)
    drop = Counter(x["task_bucket"] for x in removed)
    print(f"总题数  {sum(before.values())} -> {sum(after.values())}  (删除 {len(removed)})")
    print(f"{'bucket':<11}{'before':>8}{'removed':>9}{'after':>8}")
    for b in ["pairwise", "membership", "topk"]:
        print(f"{b:<11}{before[b]:>8}{drop[b]:>9}{after[b]:>8}")
    print(f"\n写出：\n  {Q_CLEAN}\n  {G_CLEAN}\n  {Q_REMOVED}")


if __name__ == "__main__":
    main()
