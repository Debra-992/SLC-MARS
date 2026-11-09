# -*- coding: utf-8 -*-
"""Conformal-abstention calibration and risk-coverage evaluation: read (confidence, correct/wrong)
per question from the trace store, split it into a DISJOINT calibration / test pair by qid, fit the
conformal threshold on the calibration half, then report coverage and answered-error-rate on the
test half against the un-calibrated operating point, and export a threshold file the pipeline can
load directly.

No LLM / no tool dependency: it only consumes cached traces (the tracestore.jsonl that compare_eval
writes).

Usage:
  python -m knowledge_agents.conformal_report                 # curves for 3 systems x 2 buckets
  python -m knowledge_agents.conformal_report --system pipeline_heuristic --alpha 0.10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from knowledge_agents import conformal as cf
from knowledge_agents.run_single_question import ROOT

SYSTEMS = ("pipeline_heuristic", "pipeline_auto", "baseline")
BUCKETS = ("pairwise", "membership")
DEFAULT_TRACES = ROOT / "outputs" / "traces" / "tracestore.jsonl"
ALPHA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def _load(traces_path: Path, version: str | None) -> list[dict]:
    rows: list[dict] = []
    with traces_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("error"):
                continue
            if version and (e.get("provenance") or {}).get("trace_version") != version:
                continue
            rows.append(e)
    return rows


def _in_calibration(qid: str, seed: int, frac: float) -> bool:
    h = int(hashlib.sha1(f"{seed}|{qid}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def _pred_from(entry: dict, system: str) -> cf.Pred | None:
    """Extract (score = 1 - conf, loss = 1 if wrong else 0) from one system's answer.
    Only defined when that system actually answered."""
    sysobj = entry.get("systems", {}).get(system) or {}
    outcome = sysobj.get("outcome")
    if outcome not in ("correct", "wrong"):
        return None
    conf = float((sysobj.get("answer") or {}).get("confidence", 0.0) or 0.0)
    return cf.Pred(score=cf.nonconformity(conf), loss=1 if outcome == "wrong" else 0)


def _split(rows: list[dict], system: str, bucket: str, seed: int, frac: float):
    calib: list[cf.Pred] = []
    test: list[cf.Pred] = []
    test_total = 0
    for e in rows:
        if e.get("bucket") != bucket:
            continue
        is_cal = _in_calibration(e["qid"], seed, frac)
        if not is_cal:
            test_total += 1              # the whole test side of this bucket, including items the baseline itself abstained on
        p = _pred_from(e, system)
        if p is None:
            continue
        (calib if is_cal else test).append(p)
    return calib, test, test_total


def _fmt_curve(pairs: list[cf.Pred], alphas: list[float], n_total: int) -> list[str]:
    lines = ["  | α (target, whole-bucket scale) | τ̂ | coverage | answered-error-rate | wrong/total |",
             "  |---:|---:|---:|---:|---:|"]
    for a in alphas:
        tau = cf.calibrate(pairs, a)
        r = cf.apply_threshold(pairs, tau, n_total)
        tautxt = "—" if tau is None else f"{tau:.3f}"
        lines.append(f"  | {a:.2f} | {tautxt} | {r['coverage']:.1%} | {r['risk']:.1%} | "
                     f"{r['wrong']}/{r['n_total']} |")
    return lines


def main() -> None:
    # The Windows console defaults to GBK and the report contains ⇒/≥, which would raise
    # UnicodeEncodeError; switch stdout to UTF-8 first (unencodable characters fall back to ?).
    # The artefact files are written as UTF-8 regardless.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(DEFAULT_TRACES))
    ap.add_argument("--version", default=None, help="use only this trace_version (e.g. 6)")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--alpha", type=float, default=0.10, help="target whole-bucket-scale risk written into the threshold file")
    ap.add_argument("--system", default="pipeline_heuristic",
                    choices=list(SYSTEMS) + ["all"], help="which system to export the deployable threshold for")
    args = ap.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"[conformal] no trace file found: {traces_path}")
        return
    rows = _load(traces_path, args.version)
    print("=" * 84)
    print(f"[conformal abstention calibration] trace rows={len(rows)} version={args.version or 'any'} "
          f"calibration fraction={args.calib_frac} seed={args.seed} deploy α={args.alpha}")

    thresholds_out: dict[str, float] = {}
    report: list[str] = ["[conformal abstention] calibration/test split is disjoint by qid; coverage uses the whole "
                         "test side as denominator (baseline abstentions included); "
                         "E[wrong and selected] ≤ α ⇒ answered-error-rate ≤ α/coverage", ""]

    systems = SYSTEMS if args.system == "all" else (args.system,)
    for system in systems:
        report.append(f"### system: {system}")
        for bucket in BUCKETS:
            calib, test, test_total = _split(rows, system, bucket, args.seed, args.calib_frac)
            n_all = test_total if test_total else len(test)
            base = cf.apply_threshold(test, tau=1.0, n_total=n_all)  # accept-all = the un-calibrated operating point
            report.append(f"\n- {bucket}  calibration n={len(calib)} / test n={n_all} "
                          f"(answered {len(test)}, baseline abstained on {n_all - len(test)})")
            report.append(f"  baseline (accept-all): coverage={base['coverage']:.1%} answered-error-rate={base['risk']:.1%}")
            if calib:
                tau = cf.calibrate(calib, args.alpha)
                r = cf.apply_threshold(test, tau, n_all)
                conf_thr = None if tau is None else round(1.0 - tau, 4)
                report.append(f"  conformal (α={args.alpha}, τ̂={'—' if tau is None else format(tau,'.3f')}"
                              f" ⇒ conf≥{'—' if conf_thr is None else conf_thr}): "
                              f"coverage={r['coverage']:.1%} answered-error-rate={r['risk']:.1%} "
                              f"[wrong/total={r['wrong']}/{r['n_total']}]")
                if system == (args.system if args.system != "all" else "pipeline_heuristic") and conf_thr is not None:
                    thresholds_out[bucket] = conf_thr
            if calib:
                report += ["  risk-coverage curve (self-consistent on the calibration half, illustrates the trade-off):"] + _fmt_curve(calib, ALPHA_GRID, len(calib))
            else:
                report.append("  (calibration half is empty: no threshold can be fitted, the plain decision stands)")
            # ECE (does the confidence mean what it claims) -- on the answered test samples
            cc = [(1.0 - p.score, p.loss) for p in test]
            if cc:
                report.append(f"  ECE(test-answered, confidence vs actual)={cf.ece(cc):.3f}")
        report.append("")

    text = "\n".join(report)

    # Write the artefacts first (their generation must never depend on the console encoding), then print
    out_dir = traces_path.parent
    (out_dir / "conformal_report.md").write_text(text, encoding="utf-8")
    thresholds_path = out_dir / "conformal_thresholds.json"
    if thresholds_out:
        payload = {"alpha": args.alpha, "system": args.system, "seed": args.seed,
                   "calib_frac": args.calib_frac, "thresholds": thresholds_out,
                   "note": "conf < threshold -> abstain; loaded into the decision layer via SLC_CONFORMAL_FILE"}
        thresholds_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(text)
    if thresholds_out:
        print(f"\nthreshold file: {thresholds_path}  thresholds={thresholds_out}")
    else:
        print(f"\n(α={args.alpha} yields no certifiable threshold: the calibration half is too small, or its risk floor "
              f"1/(N+1) already exceeds α, so conformal correctly abstains on everything)")
    print(f"report: {out_dir / 'conformal_report.md'}")


if __name__ == "__main__":
    main()
