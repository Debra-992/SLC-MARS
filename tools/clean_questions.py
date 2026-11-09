# -*- coding: utf-8 -*-
"""Scientific-soundness audit: drop the questions where the treatment (KO) gene is not the gene the
question asks about, and write the clean dataset separately.

Decision rule (single, and re-checkable):
  If a question's treatment is a single-gene knockout experiment `GENE.KO vs WT` (that is, the
  experiment perturbed only GENE and the ground truth is inferred from that GENE's knockout
  phenotype), while the gene the question actually asks about (membership's "Is <X> ..." /
  pairwise's "between <X> and <Y>") does not include GENE,
  then (question, answer) are semantically misaligned and scientifically unsound -> delete.

Outputs:
  data/questions_clean.jsonl          the clean questions
  evaluation/ground_truth_clean.jsonl the answers, in one-to-one correspondence
  data/questions_removed.jsonl        the removed questions + removal_reason (for audit trails)
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
# Only accept a "SLCxxx.KO" style primary-knockout gene, i.e. a single-gene-ko-vs-WT experiment.
# p53.KO and similar background/time-course screens (T0 vs endpoint, whole-genome grouping) are not
# the primary knockout, so their SLC questions are kept.
KOTOK = re.compile(r"\b(SLC[A-Z0-9]+)\.KO", re.I)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def asked_genes(rec: dict) -> set[str]:
    """The set of genes the question is actually judging or choosing between. topk has no explicit
    candidates, so it returns an empty set."""
    q = rec["question"]
    bucket = rec.get("task_bucket", "")
    if bucket == "membership":
        return {g.upper() for g in ASK.findall(q)}
    if bucket == "pairwise":
        m = BETWEEN.findall(q)
        return {m[0][0].upper(), m[0][1].upper()} if m else set()
    return set()


def perturbed_genes(rec: dict) -> set[str]:
    """The genes knocked out / perturbed in the treatment (GENE.KO)."""
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
            # the experiment only knocked out `perturb` but the question asks about `asked`, so the
            # ground truth cannot be attributed to the gene being asked about
            reason = (f"treatment perturbed genes {sorted(perturb)} do not match the genes asked about "
                      f"{sorted(asked)}: the ground truth comes from the former's knockout experiment and cannot decide the latter")
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
    print(f"total questions  {sum(before.values())} -> {sum(after.values())}  (removed {len(removed)})")
    print(f"{'bucket':<11}{'before':>8}{'removed':>9}{'after':>8}")
    for b in ["pairwise", "membership", "topk"]:
        print(f"{b:<11}{before[b]:>8}{drop[b]:>9}{after[b]:>8}")
    print(f"\nwritten:\n  {Q_CLEAN}\n  {G_CLEAN}\n  {Q_REMOVED}")


if __name__ == "__main__":
    main()
