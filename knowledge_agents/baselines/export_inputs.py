# -*- coding: utf-8 -*-
"""Export baseline inputs on exactly the question set of the frozen SLC-MARS comparison (seed5, clean, n=489).

Writes two strictly separated files so nothing leaks:
  outputs/traces/baselines_input.json -- readable by any runner: qid/bucket/question/candidates/constraints
  outputs/traces/baselines_gt.json    -- read only by merge_baselines.py for scoring; baseline runners must not read it

The candidate-gene rule matches the pipeline: pairwise = the two genes named in the stem; membership = the gene that
'Is/Does <GENE>' actually asks about (the knocked-out gene of a 'vs SLCxxx.KO' treatment is excluded).

Usage:
  .\\.venv\\Scripts\\python.exe -m knowledge_agents.baselines.export_inputs  # default: the batch pinned by the frozen SLC-MARS manifest (dataset A, byte for byte unchanged)
  .\\.venv\\Scripts\\python.exe -m knowledge_agents.baselines.export_inputs --all  # full sweep: every pairwise/membership question in the clean set (topk excluded), bypassing the manifest
"""
from __future__ import annotations

import argparse
import json
import re

from knowledge_agents.baselines.common import GT_PATH, INPUT_PATH, TRACES

_GENE_RE = re.compile(r"\b(SLC[A-Z0-9]+)\b")
_ROOT = TRACES.parents[1]


def load_jsonl(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def candidates_for(rec: dict) -> list[str]:
    question = rec["question"]
    cons = rec.get("constraints") or {}
    genes = _GENE_RE.findall(question)
    if not genes and cons.get("gene"):
        genes = [cons["gene"]]
    genes = list(dict.fromkeys(g.upper() for g in genes))
    if rec.get("task_bucket") == "membership":
        m = re.search(r"\b(?:Is|Does)\s+(SLC[A-Z0-9]+)\b", question, re.I)
        if m:
            return [m.group(1).upper()]
    return genes


def _build_inputs(qids, questions, gt):
    inputs, missing = [], []
    for q in qids:
        rec = questions.get(q)
        if not rec:
            missing.append(q)
            continue
        inputs.append({
            "qid": q,
            "bucket": rec.get("task_bucket", ""),
            "task_type": rec.get("task_type", ""),
            "question": rec["question"],
            "candidates": candidates_for(rec),
            "constraints": rec.get("constraints", {}),
        })
    if missing:
        raise SystemExit(f"{len(missing)} qids from the manifest are missing in questions_clean: {missing[:3]}")
    bad = [i["qid"] for i in inputs if not i["candidates"]]
    if bad:
        raise SystemExit(f"{len(bad)} questions yield no candidate genes; fix the export first: {bad[:3]}")
    return inputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="export every pairwise/membership question from the clean set (topk excluded), bypassing the frozen manifest")
    args = ap.parse_args()

    questions = {r["qid"]: r for r in load_jsonl(_ROOT / "data" / "questions_clean.jsonl")}
    gt = {r["qid"]: r.get("ground_truth", {}) for r in load_jsonl(_ROOT / "evaluation" / "ground_truth_clean.jsonl")}

    if args.all:
        # Full sweep: enumerate the evaluation buckets in the clean set directly (topk is neither a two-choice nor a yes/no baseline format, so it is excluded).
        # Never reads or writes sample_manifest.json -- zero overlap with the SLC-MARS run (compare_eval).
        qids = [q for q, r in questions.items() if r.get("task_bucket") in {"pairwise", "membership"}]
    else:
        mani = json.loads((TRACES / "sample_manifest.json").read_text(encoding="utf-8"))
        # Frozen question set: manifest.trace_version must be the frozen SLC-MARS value (one shared evidence backbone keeps the
        # stems and candidates aligned question by question with the baselines; the arm changes only the decision layer).
        assert str(mani.get("trace_version")) == "20" and mani.get("dataset") == "clean", \
            f"manifest is not the frozen SLC-MARS/clean line: {mani.get('trace_version')}/{mani.get('dataset')}"
        qids = mani["qids"]

    inputs = _build_inputs(qids, questions, gt)

    INPUT_PATH.write_text(json.dumps(inputs, ensure_ascii=False, indent=1), encoding="utf-8")
    GT_PATH.write_text(json.dumps({q: gt.get(q, {}) for q in qids}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    n_pw = sum(1 for i in inputs if i["bucket"] == "pairwise")
    print(f"exported {len(inputs)} questions (pairwise {n_pw} / membership {len(inputs) - n_pw}) -> {INPUT_PATH.name}, {GT_PATH.name}")


if __name__ == "__main__":
    main()
