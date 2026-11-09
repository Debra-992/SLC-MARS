# -*- coding: utf-8 -*-
"""Draw one test question at random, run the knowledge-retrieval agent and then the computation
agent, and print the whole process.

Usage:
  python -m knowledge_agents.run_single_question           # draw one question at random
  python -m knowledge_agents.run_single_question --qid membership_negative_ALL_Human_CUTLL1_Low-2V_00002
  python -m knowledge_agents.run_single_question --seed 42  # reproducible draw

Output:
  1) the question text + its constraints + the ground_truth answer
  2) knowledge-retrieval agent (Kira / project_knowledge + text2cypher): one line per intermediate
     receipt, plus the final status
  3) computation agent (Cael / PPR+TIMER+KM+DESeq2): one line per tool, plus the final status
  4) the evidence-integrity note: unknown / rejected != negative; both agents only supply evidence
     and never decide the truth value themselves
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path

from knowledge_agents.roles.compute_role import ComputeRole
from knowledge_agents.roles.knowledge_query_role import KnowledgeQueryRole
from knowledge_agents.schema import (
    BackgroundKnowledge,
    ComputedKnowledge,
    parse_knowledge,
)

ROOT = Path(__file__).resolve().parent.parent


def _resolve_dataset() -> tuple[Path, Path]:
    """SLC_DATASET=clean|full (defaults to clean whenever the cleaned files exist).

    clean: the dataset after dropping questions that are scientifically unsound because the knockout
    treatment gene is not the gene the question is actually asking about.
    """
    choice = os.environ.get("SLC_DATASET", "clean").lower()
    clean_q = ROOT / "data" / "questions_clean.jsonl"
    clean_g = ROOT / "evaluation" / "ground_truth_clean.jsonl"
    if choice == "clean" and clean_q.exists() and clean_g.exists():
        return clean_q, clean_g
    return ROOT / "data" / "questions.jsonl", ROOT / "evaluation" / "ground_truth.jsonl"


QUESTIONS, GTRUTH = _resolve_dataset()

_GENE_RE = re.compile(r"\b(SLC[A-Z0-9]+)\b")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_genes(question: str, constraints: dict) -> list[str]:
    """Every SLC candidate gene present in the question text (pairwise has two, topk has several)."""
    genes = _GENE_RE.findall(question)
    if not genes and constraints.get("gene"):
        genes = [constraints["gene"]]
    # drop duplicates, keep first-occurrence order
    return list(dict.fromkeys(genes))


def format_ground_truth(gt: dict, task_bucket: str) -> str:
    """The three buckets have different ground-truth shapes: membership->answer / pairwise->winner /
    topk->ranked list."""
    if task_bucket == "membership":
        return f"answer={gt.get('answer')}"
    if task_bucket == "pairwise":
        return f"winner={gt.get('winner')}"
    if task_bucket == "topk":
        # for topk the ground_truth is already the ranked gene list
        return f"ranking={gt.get('ranking') or gt}"
    return json.dumps(gt, ensure_ascii=False)


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


async def run_knowledge(qid: str, question: str) -> BackgroundKnowledge:
    hr("[agent 1] knowledge retrieval -- Kira (KnowledgeQueryRole)")
    role = KnowledgeQueryRole()
    # pass a qid= prefix so _infer_qid resolves the whitelisted query exactly (no regex on the text)
    await role.run(f"qid={qid} {question}")
    kg = parse_knowledge(role.rc.memory.get()[-1].content)
    assert isinstance(kg, BackgroundKnowledge)

    print(f"  primary status = {kg.status.value}  source = {kg.source}")
    print(f"  candidate genes = {kg.candidates}")
    print(f"  question constraints requested_context = {json.dumps(kg.requested_context, ensure_ascii=False)}")
    print("  --- intermediates: one receipt per evidence item ---")
    if not kg.evidence:
        print("    (no evidence receipts: the primary knowledge was not retrieved, so it stays unknown)")
    for ev in kg.evidence:
        rows = ev.rows or []
        head = json.dumps(rows[0], ensure_ascii=False, default=str)[:120] if rows else "-"
        print(f"    * op={ev.operation:<16} status={ev.status.value:<12} rows={len(rows):<3} first_row={head}")
        if ev.cypher:
            print(f"        cypher={ev.cypher[:140]}")
    if kg.cypher:
        print(f"  --- text2cypher supplement ---")
        print(f"    cypher={kg.cypher[:200]}")
        print(f"    rows_returned={len(kg.cypher_rows)}")
    print(f"  final summary = {kg.summary}")
    print(f"  public_web_used={kg.public_web_used}  source_independence={kg.source_independence}")
    return kg


async def run_compute(qid: str, genes: list[str], cancer: str, question: str, bucket: str) -> ComputedKnowledge:
    hr("[agent 2] computation -- Cael (ComputeRole)")
    role = ComputeRole()
    role.set_task(qid=qid, gene=(genes[0] if genes else None), genes=genes, cancer=cancer, task_bucket=bucket)
    await role.run(f"gene={genes[0] if genes else ''} cancer={cancer} bucket={bucket} {question}")
    ck = parse_knowledge(role.rc.memory.get()[-1].content)
    assert isinstance(ck, ComputedKnowledge)

    print(f"  routing task_bucket = {bucket!r}  genes={genes} cancer={cancer}")
    print("  --- intermediates: one line per computation tool ---")
    gc = ck.graph_computation
    if gc:
        top = ", ".join(f"{c.gene}={c.degree_corrected_score}" for c in (gc.candidates or [])[:5]) or "none"
        print(f"    * PPR: status={gc.status.value} seeds={gc.seed_count} "
              f"nodes={gc.node_count} edges={gc.edge_count} top5=[{top}]")
    if ck.expression_by_gene:
        for g, ex in ck.expression_by_gene.items():
            print(f"    * TIMER[{g}]: status={ex.status.value} row_count={ex.row_count} "
                  f"scope={ex.evidence_scope} boxplot={ex.boxplot_path}")
    elif ck.expression:
        ex = ck.expression
        print(f"    * TIMER: status={ex.status.value} row_count={ex.row_count} "
              f"scope={ex.evidence_scope} boxplot={ex.boxplot_path}")
    de = ck.differential_expression
    if de:
        top = ", ".join(f"{g.get('gene')}({g.get('log2fc'):.2f})" for g in (de.top_genes or [])[:5]
                        if g.get("log2fc") is not None) or "none"
        print(f"    * DESeq2: status={de.status.value} significant={de.significant_count} "
              f"sample_counts={de.sample_counts} top5=[{top}]")
    if ck.survival_by_gene:
        for g, km in ck.survival_by_gene.items():
            stats = (km.stats_text or "").splitlines()
            print(f"    * KM[{g}]: status={km.status.value} scope={km.evidence_scope} "
                  f"stats_head={stats[0] if stats else ''!r} plot={km.plot_path}")
    elif ck.survival:
        km = ck.survival
        stats = (km.stats_text or "").splitlines()
        print(f"    * KM: status={km.status.value} scope={km.evidence_scope} "
              f"stats_head={stats[0] if stats else ''!r} plot={km.plot_path}")
    if ck.skipped_tools:
            print(f"    (not routed / skipped: {ck.skipped_tools})")
    print(f"  final status = {ck.status.value}")
    print(f"  final summary = {ck.summary}")
    return ck


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default=None, help="qid of the question to run (default: random)")
    ap.add_argument("--seed", type=int, default=None, help="random seed (for reproducibility)")
    args = ap.parse_args()

    questions = load_jsonl(QUESTIONS)
    gt_index = {r["qid"]: r for r in load_jsonl(GTRUTH)}

    if args.qid:
        q = next((r for r in questions if r["qid"] == args.qid), None)
        if q is None:
            raise SystemExit(f"question not found for qid={args.qid}")
    else:
        rng = random.Random(args.seed)
        q = rng.choice(questions)

    qid = q["qid"]
    constraints = q.get("constraints", {})
    genes = extract_genes(q["question"], constraints)
    gene = genes[0] if genes else None
    cancer = constraints.get("cancer", "")
    bucket = q.get("task_bucket", "")
    task_type = q.get("task_type", "")
    gt = gt_index.get(qid, {}).get("ground_truth", {})

    hr(f"[question] random draw (pool of {len(questions)} questions)")
    print(f"  qid        = {qid}")
    print(f"  task_bucket= {bucket}   task_type={task_type}")
    print(f"  constraints= {json.dumps(constraints, ensure_ascii=False)}")
    print(f"  question   = {q['question']}")
    print(f"  candidate genes in the question text = {genes}  (routing primary gene={gene}  cancer={cancer})")
    print(f"  ground_truth -> {format_ground_truth(gt, bucket)}")

    kg = await run_knowledge(qid, q["question"])
    ck = await run_compute(qid, genes, cancer, q["question"], bucket)

    hr("[summary] the two agent outputs vs the correct answer")
    print(f"  knowledge retrieval Kira: status={kg.status.value:<12} evidence_items={len(kg.evidence)}")
    print(f"  computation Cael: status={ck.status.value:<12} "
          f"(ppr={'Y' if ck.graph_computation else '-'} timer={'Y' if ck.expression else '-'} "
          f"de={'Y' if ck.differential_expression else '-'} km={'Y' if ck.survival else '-'})")
    print(f"  ground_truth: {format_ground_truth(gt, bucket)}")
    print("  note: both agents only supply [background knowledge + candidate numeric evidence]; the membership")
    print("        truth value needs the downstream decision role and the CRISPR screen readout; a tool reporting unknown or rejected is never read as negative.")


if __name__ == "__main__":
    asyncio.run(main())
