# -*- coding: utf-8 -*-
"""Fair comparison: run three systems on the same clean question sample and emit a multi-metric
comparison table plus a full per-question trace archive.

The three systems (stored separately):
  - pipeline heuristic  Kira||Cael grounded evidence -> pure heuristic scoring (may abstain)
  - pipeline auto       the same evidence, with cross-cancer-type analogy by LLM when evidence is
                        insufficient (tries not to abstain)
  - native MetaGPT      stock MetaGPT Role, pure LLM (no tools, no retrieval, no CRISPR data)

What makes the comparison fair:
  - one (n, buckets, seed) draw -> all three systems answer exactly the same questions.
  - coverage / accuracy (answered) / accuracy (overall) / net-vs-chance are reported together, so a
    system cannot buy accuracy by abstaining en masse.

Full-process retention and incremental reuse:
  - every question's full process (each agent output + the intermediate kg/ck evidence + the
    reasoning rationale + gene_scores) is written to outputs/traces/tracestore.jsonl (deduplicated by
    qid, so a run can resume).
  - each question is additionally written as outputs/traces/by_question/<qid>.json (human readable).
  - questions already cached under a matching configuration are not recomputed, so growing n only
    runs the new ones.

Usage:
  python -m knowledge_agents.compare_eval --n 100 --seed 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from knowledge_agents.answer_pipeline import AnswerTeam, _score, sample_questions
from knowledge_agents.baseline_metagpt import BaselineRole, preflight_llm
from knowledge_agents.baselines.common import install_temperature_zero
from knowledge_agents.run_single_question import ROOT, extract_genes
from knowledge_agents.schema import parse_knowledge

SUPPORTED = ("pairwise", "membership")
SYSTEMS = ("pipeline_heuristic", "pipeline_auto", "baseline")
DISPLAY = {"pipeline_heuristic": "pipeline heuristic", "pipeline_auto": "pipeline auto (analogy)",
           "baseline": "native MetaGPT"}
TRACE_VERSION = os.environ.get("SLC_TRACE_VERSION", "20")  # Frozen trace-schema/cache key of the released configuration (the literal "20" matches the shipped tracestore records, trace_version=20, covering the whole clean pool of n=489 questions). Bump it only when the decision or evidence structure changes, which invalidates the old cache. The released configuration = the role-heterogeneous prior triad (triad vote + triad critique + two racing prior draws with the second persona supplied in parallel) + the R2 membership essentiality evidence re-ranking. SLC_TRACE_VERSION can be overridden temporarily to run a comparison arm.
CHANCE = 0.5                 # both pairwise and membership are a forced binary choice
TRACES_DIR = ROOT / "outputs" / "traces"
STORE_PATH = TRACES_DIR / "tracestore.jsonl"
BYQ_DIR = TRACES_DIR / "by_question"
MANIFEST_PATH = TRACES_DIR / "sample_manifest.json"


# ------------------------------------------------------------------ serialisation
def _dump(model) -> dict | None:
    """pydantic model -> plain JSON dict (enums as strings); None passes through."""
    if model is None:
        return None
    return json.loads(model.model_dump_json())


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ per-question full trace
async def trace_one(team: AnswerTeam, rec: dict, llm_ok: bool) -> dict:
    """Run one question: Kira||Cael supply once -> heuristic / auto decisions + the baseline; returns
    the full-process dict."""
    bucket = rec.get("task_bucket", "")
    genes, _ = team._effective_genes(rec)
    supplied = await team.supply(rec)                     # tool evidence fetched once, shared by all three systems

    out: dict = {
        "qid": rec["qid"], "bucket": bucket, "task_type": rec.get("task_type", ""),
        "question": rec["question"], "constraints": rec.get("constraints", {}),
        "ground_truth": rec.get("_gt", {}), "effective_genes": genes,
        "supply": {"kg": _dump(supplied["kg"]), "ck": _dump(supplied["ck"])},
        "systems": {},
    }

    for mode, key in (("heuristic", "pipeline_heuristic"), ("auto", "pipeline_auto")):
        ans = await team.decide(supplied, mode=mode)
        out["systems"][key] = {"answer": _dump(ans), "outcome": _score(ans, rec["_gt"], bucket)}

    # native MetaGPT baseline (pure LLM)
    if llm_ok:
        bgenes = extract_genes(rec["question"], rec.get("constraints", {}))
        role = BaselineRole()
        role.set_inputs(bucket=bucket, candidates=bgenes, question=rec["question"],
                        task_type=rec.get("task_type", ""))
        try:
            await role.run("answer")
            bans = parse_knowledge(role.rc.memory.get()[-1].content)
            out["systems"]["baseline"] = {"answer": _dump(bans), "outcome": _score(bans, rec["_gt"], bucket)}
        except Exception as e:  # noqa BLE001
            out["systems"]["baseline"] = {"answer": None, "outcome": "abstain", "error": type(e).__name__}
    else:
        out["systems"]["baseline"] = {"answer": None, "outcome": "abstain", "error": "llm_unavailable"}

    out["provenance"] = {"dataset": os.environ.get("SLC_DATASET", "clean"),
                         "trace_version": TRACE_VERSION, "baseline_llm": llm_ok, "ts": _now()}
    return out


def _reusable(entry: dict, dataset: str, llm_ok: bool) -> bool:
    p = entry.get("provenance", {})
    if entry.get("error") or p.get("dataset") != dataset or p.get("trace_version") != TRACE_VERSION:
        return False
    if bool(p.get("baseline_llm")) != bool(llm_ok):
        return False
    systems = entry.get("systems") or {}
    if set(systems) < set(SYSTEMS):
        return False
    # must be a genuinely produced answer (it carries an "answer" field); the exception fallback stub
    # only has an outcome and must not be reused
    return all("answer" in systems[s] for s in SYSTEMS)


# ------------------------------------------------------------------ trace store
def load_store() -> dict[str, dict]:
    store: dict[str, dict] = {}
    if STORE_PATH.exists():
        for line in STORE_PATH.open(encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    store[r["qid"]] = r
                except json.JSONDecodeError:
                    continue
    return store


def append_store(entry: dict) -> None:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    with STORE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    BYQ_DIR.mkdir(parents=True, exist_ok=True)
    (BYQ_DIR / f"{entry['qid']}.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ metrics
def _metrics(outcomes: list[str]) -> dict:
    c = outcomes.count("correct"); w = outcomes.count("wrong"); a = outcomes.count("abstain")
    n = c + w + a; ans = c + w
    acc_overall = c / n if n else 0.0
    return dict(n=n, correct=c, wrong=w, abstain=a,
                coverage=ans / n if n else 0.0,
                acc_answered=c / ans if ans else 0.0,
                acc_overall=acc_overall, net_vs_chance=acc_overall - CHANCE)


def _fmt_row(name: str, m: dict) -> str:
    return (f"| {name:<16} | {m['n']:>3} | {m['correct']:>3} | {m['wrong']:>3} | {m['abstain']:>3} "
            f"| {m['coverage']:>6.1%} | {m['acc_answered']:>7.1%} | {m['acc_overall']:>7.1%} "
            f"| {m['net_vs_chance']:>+8.1%} |")


def _table(title: str, rows: list[tuple[str, dict]]) -> str:
    n = rows[0][1]["n"] if rows else 0
    lines = [f"\n### {title}  (n={n})",
             "| system | N | correct | wrong | abstain | coverage | accuracy (answered) | accuracy (overall) | net vs chance |",
             "|------|----:|---:|---:|---:|------:|--------:|--------:|--------:|"]
    for name, m in rows:
        lines.append(_fmt_row(name, m))
    return "\n".join(lines)


# ------------------------------------------------------------------ main flow
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--buckets", default="pairwise,membership")
    ap.add_argument("--no-cache", action="store_true", help="ignore the cache and recompute everything")
    args = ap.parse_args()
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    tz = install_temperature_zero()   # with SLC_FORCE_TEMP0=1 every LLM call is pinned to temperature 0, so the R2 on/off A/B attributes cleanly

    sample = sample_questions(args.n, buckets, args.seed)
    dataset = os.environ.get("SLC_DATASET", "clean")
    llm_ok = await preflight_llm()
    print("=" * 84, flush=True)
    print(f"[comparison eval] dataset={dataset} sample={len(sample)} questions seed={args.seed} buckets={buckets} "
          f"| baseline LLM {'available' if llm_ok else 'unavailable (recorded as abstain)'} | temp0={'forced' if tz else 'off'}", flush=True)

    store = {} if args.no_cache else load_store()
    team = AnswerTeam()
    results: list[dict] = []
    n_new = n_cached = 0
    for i, rec in enumerate(sample, 1):
        entry = store.get(rec["qid"])
        if entry and _reusable(entry, dataset, llm_ok):
            n_cached += 1
            tag = "CACHE"
        else:
            try:
                entry = await trace_one(team, rec, llm_ok)
            except Exception as e:  # noqa BLE001
                entry = {"qid": rec["qid"], "bucket": rec.get("task_bucket"),
                         "systems": {s: {"outcome": "abstain"} for s in SYSTEMS},
                         "provenance": {"dataset": dataset, "trace_version": TRACE_VERSION,
                                        "baseline_llm": llm_ok, "ts": _now()},
                         "error": f"{type(e).__name__}: {str(e)[:80]}"}
            append_store(entry)
            store[rec["qid"]] = entry
            n_new += 1
            tag = "NEW"
        results.append(entry)
        oc = {s: entry.get("systems", {}).get(s, {}).get("outcome", "abstain") for s in SYSTEMS}
        print(f"[{i:>3}/{len(sample)}] {tag:<5} {rec['qid'][:42]:<42} "
              f"heur={oc['pipeline_heuristic']:<7} auto={oc['pipeline_auto']:<7} "
              f"base={oc['baseline']}", flush=True)

    # summary over the current sample only; unrelated cached questions are not mixed in
    outcomes = {s: {b: [] for b in SUPPORTED} for s in SYSTEMS}
    for entry in results:
        b = entry.get("bucket")
        if b not in SUPPORTED:
            continue
        for s in SYSTEMS:
            outcomes[s][b].append(entry.get("systems", {}).get(s, {}).get("outcome", "abstain"))

    md = ["[fair comparison table] coverage = answered/total | accuracy (answered) = correct/answered | accuracy (overall) = correct/total (an abstention counts as wrong) "
          f"| net vs chance = accuracy (overall) - {CHANCE:.0%}",
          f"sample: n={len(sample)} seed={args.seed} dataset={dataset} buckets={buckets} "
          f"(newly computed {n_new} / reused from cache {n_cached})"]
    for b in SUPPORTED:
        md.append(_table(b, [(DISPLAY[s], _metrics(outcomes[s][b])) for s in SYSTEMS]))
    md.append(_table("OVERALL", [
        (DISPLAY[s], _metrics(outcomes[s]["pairwise"] + outcomes[s]["membership"])) for s in SYSTEMS]))
    table_text = "\n".join(md)

    print("\n" + "=" * 84, flush=True)
    print(table_text, flush=True)
    print(f"\nfull traces: {STORE_PATH}  |  per question: {BYQ_DIR}\\<qid>.json", flush=True)

    # manifest: records which questions this run used, so a later run can grow the sample without recomputing
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps({
        "updated_at": _now(), "dataset": dataset, "seed": args.seed, "buckets": buckets,
        "n": len(sample), "trace_version": TRACE_VERSION, "baseline_llm": llm_ok,
        "qids": [r["qid"] for r in results],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    run_table = TRACES_DIR / f"compare_table_n{len(sample)}_seed{args.seed}.md"
    run_table.write_text(table_text, encoding="utf-8")
    print(f"manifest:      {MANIFEST_PATH}\ntable archive: {run_table}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
