# -*- coding: utf-8 -*-
"""Builds the conformal "calibration pool": runs only pipeline-heuristic (pure scoring, no LLM) over a
large question set and emits per-question (confidence, correct/wrong) for conformal_report to fit
thresholds and measure risk-coverage.

Why a separate dumper:
  - Calibrating conformal abstention needs many answered samples before the threshold gets tight
    enough to deploy, and the heuristic decision is deterministic and LLM-free, so a few hundred
    questions cost only tool/retrieval time and make that pool large.
  - It writes its own calibstore.jsonl (deduplicated over the whole file) instead of polluting the
    three-system trace cache tracestore.jsonl that compare_eval owns.
  - Calibration must see the raw confidence distribution before any conformal filter, so this script clears
    SLC_CONFORMAL_FILE explicitly and never truncates by its own threshold (feeding a threshold into
    the fit that produces it would be circular).

Usage:
  python -m knowledge_agents.calib_dump --n 250 --seed 5
  python -m knowledge_agents.conformal_report --traces outputs/traces/calibstore.jsonl \
      --version "$env:SLC_TRACE_VERSION" --system pipeline_heuristic --alpha 0.10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone

from knowledge_agents.answer_pipeline import AnswerTeam, _score, sample_questions
from knowledge_agents.compare_eval import TRACE_VERSION, _dump
from knowledge_agents.run_single_question import ROOT

DATASET = os.environ.get("SLC_DATASET", "clean")
CALIB_DIR = ROOT / "outputs" / "traces"
CALIB_PATH = CALIB_DIR / "calibstore.jsonl"
SUPPORTED = ("pairwise", "membership")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_calib() -> dict[str, dict]:
    store: dict[str, dict] = {}
    if CALIB_PATH.exists():
        for line in CALIB_PATH.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                store[r["qid"]] = r
            except json.JSONDecodeError:
                continue
    return store


def write_calib(store: dict[str, dict]) -> None:
    """Rewrite the whole file: exactly one line per qid, so conformal_report counting line by line
    cannot double-count."""
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CALIB_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for qid in sorted(store):
            f.write(json.dumps(store[qid], ensure_ascii=False) + "\n")
    tmp.replace(CALIB_PATH)


def _reusable(entry: dict) -> bool:
    if entry.get("error"):
        return False
    p = entry.get("provenance", {})
    if p.get("dataset") != DATASET or p.get("trace_version") != TRACE_VERSION:
        return False
    sysobj = entry.get("systems", {}).get("pipeline_heuristic") or {}
    return "answer" in sysobj


def _features(supplied: dict) -> dict:
    """Extract a leakage-free feature vector from the raw tool output (evidence only, never the
    ground truth).

    The target gene is the routed primary gene supplied['gene'] (single gene for membership,
    primary gene for pairwise). These features feed the logistic calibration, which learns its
    weights from the clean labels; deciding and scoring still use only the heuristic score, so the
    two never contaminate each other.
    """
    kg, ck = supplied["kg"], supplied["ck"]
    gene = (supplied.get("gene") or "").upper()
    cancer = str((supplied.get("constraints") or {}).get("cancer") or "").lower()

    # --- KG gene_disease_paths: split the SLC->Pathway->Disease chain finely instead of
    #     collapsing it into a single composite polarity ---
    #   promotion/suppression counts of the composite effect gene_disease_effect(rel_AB⊗rel_BC)
    #   (the legacy feature, kept for backward compatibility)
    #   + separate direction counts for the two legs rel_AB (gene->pathway) and rel_BC (pathway->disease)
    #   + promotion/suppression split restricted to the disease matching this cancer
    #   + number of distinct mediating pathways (pathway diversity)
    n_paths = n_promote = n_suppress = n_cancer_hit = 0
    n_ab_promote = n_ab_suppress = n_bc_promote = n_bc_suppress = 0
    n_promote_matched = n_suppress_matched = 0
    pathways: set[str] = set()
    pathways_matched: set[str] = set()

    def _sign(v):
        u = str(v or "").upper()
        return 1 if u == "PROMOTION" else (-1 if u == "SUPPRESSION" else 0)

    for ev in (getattr(kg, "evidence", []) or []):
        if ev.operation != "gene_disease_paths":
            continue
        for row in ev.rows or []:
            if str(row.get("gene", "")).upper() != gene:
                continue
            n_paths += 1
            eff = str(row.get("gene_disease_effect", "")).upper()
            if eff == "PROMOTION":
                n_promote += 1
            elif eff == "SUPPRESSION":
                n_suppress += 1
            ab, bc = _sign(row.get("rel_AB")), _sign(row.get("rel_BC"))
            n_ab_promote += ab > 0
            n_ab_suppress += ab < 0
            n_bc_promote += bc > 0
            n_bc_suppress += bc < 0
            p = (row.get("pathway") or "").upper()
            if p:
                pathways.add(p)
            d = str(row.get("disease", "")).lower()
            matched = bool(cancer) and cancer in d
            if matched:
                n_cancer_hit += 1
                if eff == "PROMOTION":
                    n_promote_matched += 1
                elif eff == "SUPPRESSION":
                    n_suppress_matched += 1
                if p:
                    pathways_matched.add(p)

    # --- PPR network proximity ---
    ppr = deg = None
    in_graph = 0
    gc = getattr(ck, "graph_computation", None)
    if gc:
        for c in gc.candidates or []:
            if str(c.gene).upper() == gene:
                ppr = c.ppr_score
                deg = c.degree_corrected_score
                in_graph = 1 if c.in_graph else 0
                break

    # --- TIMER expression (row count / availability) ---
    exp = getattr(ck, "expression_by_gene", {}).get(gene) or ck.expression
    timer_rows = exp.row_count if exp is not None and getattr(exp, "status", None) else None
    timer_ok = 1 if (exp is not None and getattr(exp, "status", None) is not None
                     and exp.status.is_knowledge) else 0

    # --- KM survival (availability) ---
    surv = getattr(ck, "survival_by_gene", {}).get(gene) or ck.survival
    km_ok = 1 if (surv is not None and getattr(surv, "status", None) is not None
                  and surv.status.is_knowledge) else 0

    # --- DESeq2: signed log2FC of the target gene, significant-gene count, panel size ---
    de_l2fc = None
    de_sig = de_n = None
    de = getattr(ck, "differential_expression", None)
    if de is not None:
        de_sig = de.significant_count
        de_n = len(de.top_genes or [])
        for row in de.top_genes or []:
            g = str(row.get("gene_symbol") or row.get("gene") or "").upper()
            if g == gene:
                try:
                    de_l2fc = float(row.get("log2fc"))
                except (TypeError, ValueError):
                    de_l2fc = None
                break

    return {
        "gene": gene, "n_paths": n_paths, "n_promote": n_promote,
        "n_suppress": n_suppress, "n_cancer_hit": n_cancer_hit,
        # fine-grained chain features (rel_AB/rel_BC leg directions, pathway polarity inside this
        # cancer, mediating-pathway diversity)
        "n_ab_promote": n_ab_promote, "n_ab_suppress": n_ab_suppress,
        "n_bc_promote": n_bc_promote, "n_bc_suppress": n_bc_suppress,
        "n_promote_matched": n_promote_matched, "n_suppress_matched": n_suppress_matched,
        "n_distinct_pathways": len(pathways), "n_distinct_pathways_matched": len(pathways_matched),
        "ppr": ppr, "ppr_deg_corrected": deg, "in_graph": in_graph,
        "timer_rows": timer_rows, "timer_ok": timer_ok, "km_ok": km_ok,
        "de_log2fc": de_l2fc, "de_sig_count": de_sig, "de_panel_n": de_n,
        "paralog_redundancy": round((ck.paralog_redundancy or {}).get(gene, 0.0), 4),
    }


def _kg_paths(supplied: dict) -> list[dict]:
    """Persist the raw path rows of gene_disease_paths (every candidate gene, not only the primary
    one) so that later graph-feature enrichment can be recomputed offline in seconds without rerunning
    the whole pool. Evidence fields only; never the ground truth."""
    rows: list[dict] = []
    for ev in (getattr(supplied["kg"], "evidence", []) or []):
        if ev.operation != "gene_disease_paths":
            continue
        for r in ev.rows or []:
            if not isinstance(r, dict):
                continue
            rows.append({
                "gene": str(r.get("gene", "")).upper(),
                "pathway": (r.get("pathway") or ""),
                "rel_AB": (r.get("rel_AB") or "").upper() or None,
                "rel_BC": (r.get("rel_BC") or "").upper() or None,
                "effect": (r.get("gene_disease_effect") or "").upper() or None,
                "disease": str(r.get("disease", "")),
            })
    return rows


async def _run_one(team: AnswerTeam, rec: dict) -> dict:
    bucket = rec.get("task_bucket", "")
    supplied = await team.supply(rec)
    ans = await team.decide(supplied, mode="heuristic")
    ck = supplied["ck"]
    return {
        "qid": rec["qid"], "bucket": bucket, "task_type": rec.get("task_type", ""),
        "question": rec["question"], "ground_truth": rec.get("_gt", {}),
        "effective_genes": supplied["genes"],
        "systems": {"pipeline_heuristic": {
            "answer": _dump(ans), "outcome": _score(ans, rec["_gt"], bucket)}},
        # compact evidence summary (human-readable; the full kg/ck is not stored to keep the file small)
        "evidence_view": {
            "paralog_redundancy": {k: round(v, 3) for k, v in (ck.paralog_redundancy or {}).items()},
            "kg_evidence_n": len(getattr(supplied["kg"], "evidence", []) or []),
            "de": bool(ck.differential_expression),
        },
        # leakage-free feature vector + the membership ground-truth label (essential=True -> 1)
        "features": _features(supplied),
        # raw SLC->Pathway->Disease path rows (offline graph-feature enrichment, no pool rerun needed)
        "kg_paths": _kg_paths(supplied),
        "cancer": str((supplied.get("constraints") or {}).get("cancer") or ""),
        "label": (1 if bool((rec.get("_gt") or {}).get("answer")) else 0) if bucket == "membership" else None,
        "provenance": {"dataset": DATASET, "trace_version": TRACE_VERSION, "ts": _now()},
    }


async def main() -> None:
    # calibration needs the untruncated raw confidences: switch off the in-flight conformal
    # filter explicitly so it cannot become self-fulfilling
    os.environ.pop("SLC_CONFORMAL_FILE", None)

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--buckets", default="pairwise,membership")
    ap.add_argument("--no-cache", action="store_true", help="ignore the existing pool and recompute everything")
    args = ap.parse_args()
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]

    sample = sample_questions(args.n, buckets, args.seed)
    store = {} if args.no_cache else load_calib()
    team = AnswerTeam()
    n_new = n_cached = n_err = 0
    print("=" * 84, flush=True)
    print(f"[conformal calibration pool] dataset={DATASET} version={TRACE_VERSION} sampled={len(sample)} "
          f"seed={args.seed} buckets={buckets} (already pooled {len(store)})", flush=True)

    for i, rec in enumerate(sample, 1):
        entry = store.get(rec["qid"])
        if entry and _reusable(entry):
            n_cached += 1
            tag = "CACHE"
        else:
            try:
                entry = await _run_one(team, rec)
            except Exception as e:  # noqa BLE001
                entry = {"qid": rec["qid"], "bucket": rec.get("task_bucket"),
                         "systems": {"pipeline_heuristic": {"outcome": "abstain"}},
                         "provenance": {"dataset": DATASET, "trace_version": TRACE_VERSION, "ts": _now()},
                         "error": f"{type(e).__name__}: {str(e)[:80]}"}
                n_err += 1
            store[rec["qid"]] = entry
            n_new += 1
            tag = "NEW"
        oc = entry.get("systems", {}).get("pipeline_heuristic", {}).get("outcome", "abstain")
        conf = ((entry.get("systems", {}).get("pipeline_heuristic", {}).get("answer") or {})
                .get("confidence"))
        print(f"[{i:>3}/{len(sample)}] {tag:<5} {rec['qid'][:42]:<42} {oc:<7} "
              f"conf={conf if conf is not None else '-'}", flush=True)
        write_calib(store)                          # flushed after every question, so an interruption resumes

    answered = sum(1 for e in store.values()
                   if (e.get("systems", {}).get("pipeline_heuristic") or {}).get("outcome")
                   in ("correct", "wrong") and not e.get("error"))
    print("\n" + "=" * 84, flush=True)
    print(f"[done] new {n_new} / reused {n_cached} / errored {n_err} | "
          f"pool size={len(store)} (answered {answered})", flush=True)
    print(f"pool file: {CALIB_PATH}", flush=True)
    print("next step: python -m knowledge_agents.conformal_report "
          f"--traces {CALIB_PATH} --version {TRACE_VERSION} --system pipeline_heuristic", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
