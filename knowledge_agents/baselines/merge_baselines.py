# -*- coding: utf-8 -*-
"""Frozen SLC-MARS release: seven systems x two datasets summary table (plus one in-harness diagnostic arm).

Data sources:
  * pipeline-heuristic / pipeline-auto:tracestore.jsonl,
    the latest row per qid with trace_version in TRACE_VERSIONS;
  * pure_llm / lit_rag / metagpt / autogen / mdagents:answers_<system>.jsonl
    (stem-driven and version independent; A and B reuse the same answer slice) -- these five are the paper's Table 1 baselines;
  * metaGPT-native (in-harness): tracestore systems.baseline -- the stock MetaGPT Role of
    knowledge_agents/baseline_metagpt.py, answered from parametric knowledge only (no tools, no retrieval) but
    evaluated inside compare_eval's own harness on the shared sample, with a short per-call budget. Its low
    coverage is failed/unparseable calls recorded honestly as abstentions, NOT deliberate refusal, and it is not
    evidence-conditioned: the pipeline's assembled evidence is never handed to it.
    It is a different run of the same idea from the standalone MetaGPT in answers_metagpt.jsonl (batch runner,
    longer call timeout, ~99.6% coverage); cite that one as the MetaGPT baseline and never conflate the two rows.
Scoring follows answer_pipeline._score exactly (error => honest abstain).

Dataset B follows the double-probe labelling of tools/label_antimemorization.py:
  B:= class in {novel, unstable, probe_error} (questions where pure-LLM memory is unreliable or blind).
  Note: B reuses pure-llm probe1 for its labelling, so pure-llm scoring lower on B is a property of the construction
  (unstable includes "probe1 wrong"); it is expected and must be stated when interpreting.

Usage:   .\\.venv\\Scripts\\python.exe -m knowledge_agents.baselines.merge_baselines
Output: outputs/traces/baselines_table_master.md (plus the console)
"""
from __future__ import annotations

import json
import math
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa BLE001
        pass

from knowledge_agents.baselines.common import GT_PATH, INPUT_PATH, TRACES

PIPE_ARMS = [("pipeline-heuristic", "tracestore", "pipeline_heuristic"),
             ("pipeline-auto", "tracestore", "pipeline_auto")]
NEW_ARMS = [("pure-llm", "answers", "pure_llm"),
            ("lit-rag", "answers", "lit_rag"),
            ("MetaGPT", "answers", "metagpt"),
            ("autogen", "answers", "autogen"),
            ("mdagents", "answers", "mdagents")]
# Separate diagnostic arm: the stock MetaGPT Role answered inside compare_eval's harness (tracestore systems.baseline).
# Same parametric-only system as the standalone MetaGPT above, but a different run: it inherits the harness's short
# per-call timeout, so its abstentions are call failures. Comparing the two rows measures harness reliability, not
# the value of evidence -- the in-harness arm never sees the pipeline's evidence view.
EXTRA_ARMS = [("metaGPT-native(in-harness)", "tracestore", "baseline")]
ALL_ARMS = PIPE_ARMS + NEW_ARMS + EXTRA_ARMS
TRACE_VERSIONS = {"20"}  # internal arm id of the frozen SLC-MARS rows (the whole clean pool, n=489, is complete, so no earlier arm is needed as a fallback); earlier experiment arms stay in tracestore but never enter the table


def load_jsonl(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def outcome_of(row: dict, gt: dict, bucket: str) -> str:
    """Dictionary form of answer_pipeline._score, same semantics."""
    if row.get("error") or row.get("abstained"):
        return "abstain"
    if bucket == "pairwise":
        w = str(row.get("winner") or "").upper()
        if not w:
            return "abstain"
        return "correct" if w == str(gt.get("winner", "")).upper() else "wrong"
    a = row.get("answer")
    if a is None:
        return "abstain"
    return "correct" if bool(a) == bool(gt.get("answer")) else "wrong"


def collect(qids: set[str], gt: dict, bucket_of: dict[str, str]) -> dict:
    """-> {arm: {qid: outcome}}; where a qid was appended more than once, the last row wins."""
    latest: dict[str, dict] = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        v = str((r.get("provenance") or {}).get("trace_version") or "")
        if v not in TRACE_VERSIONS:
            continue
        old = latest.get(r["qid"])
        if old is not None and str((old.get("provenance") or {}).get("trace_version") or "0") > v:
            continue  # a higher-version row already exists -> never backfill from a lower one
        latest[r["qid"]] = r
    trace_rows = {q: r for q, r in latest.items() if q in qids}

    outs: dict[str, dict[str, str]] = {}
    for name, kind, key in ALL_ARMS:
        per_q: dict[str, str] = {}
        if kind == "tracestore":
            for q in qids:
                r = trace_rows.get(q)
                if r is None:
                    per_q[q] = "missing"
                    continue
                o = (r.get("systems", {}).get(key) or {}).get("outcome")
                per_q[q] = o if o in ("correct", "wrong", "abstain") else "abstain"
        else:
            p = TRACES / f"answers_{key}.jsonl"
            rows: dict[str, dict] = {}
            if p.exists():
                for r in load_jsonl(p):
                    rows[r["qid"]] = r
            for q in qids:
                r = rows.get(q)
                per_q[q] = outcome_of(r, gt.get(q, {}), bucket_of[q]) if r else "missing"
        outs[name] = per_q
    return outs


def metrics(per_q: dict, keep: set[str]) -> dict:
    c = w = a = m = 0
    for q, o in per_q.items():
        if q not in keep:
            continue
        if o == "missing":
            m += 1
        elif o == "correct":
            c += 1
        elif o == "wrong":
            w += 1
        else:
            a += 1
    n = c + w + a
    ans = c + w
    return dict(n=n, c=c, w=w, ab=a, missing=m,
                acc_all=(c / n if n else float("nan")),
                acc_ans=(c / ans if ans else float("nan")),
                cov=(ans / n if n else float("nan")))


def main() -> None:
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((TRACES / "sample_manifest.json").read_text(encoding="utf-8"))
    a_qids = set(manifest["qids"])
    bucket_of = {i["qid"]: i["bucket"] for i in inputs}

    # Dataset B: the anti-memorization question set (if labels are missing, the report falls back to dataset A only)
    # Frozen definition = the full-pool labelling dataset_b_qids_all489.json (|B|=265);
    # the older dataset_b_qids.json is a leftover from the n=100 era, used only if all489 is absent; never mix the two definitions.
    b_path = TRACES / "dataset_b_qids_all489.json"
    if not b_path.exists():
        b_path = TRACES / "dataset_b_qids.json"
    b_qids = set(json.loads(b_path.read_text(encoding="utf-8"))["b_qids"]) if b_path.exists() else None

    datasets = [("A", a_qids)]
    if b_qids is not None:
        datasets.append(("B", b_qids & a_qids))
    outs_all = {tag: collect(keep, gt, bucket_of) for tag, keep in datasets}

    lines = [f"# SLC-MARS: seven systems x two datasets, the same {len(a_qids)} questions (seed {manifest.get('seed')}, one _score definition)",
             "",
             f"- dataset A = every question (pipeline arm trace_version={sorted(TRACE_VERSIONS)}); "
             f"dataset B = anti-memorization set (pure-LLM double probe disagreeing: novel + unstable + probe_error, list {b_path.name})"
             + (f",|B|={len(datasets[1][1])}" if b_qids is not None else " (labels not generated; this run reports dataset A only)"),
             "- Note: B was constructed with a pure-llm probe1 pass, so pure-llm scoring lower on B is by construction (expected) and needs stating.",
             ""]
    for tag, keep in datasets:
        outs = outs_all[tag]
        lines += [f"# ===== dataset {tag} (n={len(keep)}) =====", ""]
        for bucket in ("pairwise", "membership"):
            sub_keep = {q for q in keep if bucket_of.get(q) == bucket}
            lines += _render(f"{tag}   {bucket}(n={len(sub_keep)})", outs, sub_keep)
        lines += _render(f"{tag}   OVERALL(n={len(keep)})", outs, keep)

    # Run health
    lines.append("## Run health (error rows in answers_*.jsonl / questions missing from A or B)")
    for name, kind, key in NEW_ARMS:
        p = TRACES / f"answers_{key}.jsonl"
        rows: dict[str, dict] = {}
        if p.exists():
            for r in load_jsonl(p):
                rows[r["qid"]] = r
        err = sum(1 for r in rows.values() if r.get("error"))
        lines.append(f"- {name}: answered {len(rows)}, of which {err} error")

    md = "\n".join(lines) + "\n"
    out = TRACES / "baselines_table_master.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"[merge] -> {out}")


def _render(title: str, outs: dict, keep: set[str]) -> list[str]:
    lines = [f"## {title}", "",
             "| system | N | correct | wrong | abstain | coverage | acc (all) | acc (answered) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, *_ in ALL_ARMS:
        m = metrics(outs[name], keep)
        miss = f" (missing {m['missing']})" if m["missing"] else ""
        cov = f"{m['cov']:.1%}" if m['n'] else "-"
        aa = f"{m['acc_all']:.1%}" if m['n'] else "-"
        an = f"{m['acc_ans']:.1%}" if m['n'] and not math.isnan(m['acc_ans']) else "-"
        lines.append(f"| {name} | {m['n']}{miss} | {m['c']} | {m['w']} | {m['ab']} | {cov} | {aa} | {an} |")
    lines.append("")
    return lines


if __name__ == "__main__":
    main()
