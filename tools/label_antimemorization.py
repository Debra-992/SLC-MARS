# -*- coding: utf-8 -*-
"""Anti-memorization labelling tool: split the Dataset A questions into "recallable" and
"anti-memorization (Dataset B)".

Motivation: the baseline audit found that pure-llm / autogen / mdagents all answer from bare memory
with zero retrieval, so on a corpus derived from the same sources as DepMap they may have "memorised
the answers". Dataset B is the set the model memory does not cover; only evaluating on it exposes
the real increment of the pipeline (graph / statistical evidence) over the memory prior.

Two probes (both with zero retrieval, same endpoint and same temperature as the baselines):
  probe1 = the stored answer to the original question text from
           outputs/traces/answers_pure_llm.jsonl (no paraphrase);
  probe2 = a fresh call that [drops the original question phrasing], gives only the experimental
           constraints plus the candidates, and asks through a blank-fill template "which gene
           drives the differential essentiality" -- this disrupts verbatim memory cues and probes
           the actual knowledge.
Decision (the ground truth is read only by this tool for comparison and never enters any answering
path):
  stable   = both probes correct (memory is solid, excluded from B);
  novel    = both probes wrong (a pure blind spot, the core of B);
  unstable = exactly one correct (the memory is unreliable, which also counts as "memory
             unavailable", so it enters B and is listed separately).
  B := novel union unstable.

env: SLC_PROBE2_TIMEOUT is the per-call timeout (120s default); credentials come from
baselines.common.llm_settings().
Usage: $env:PYTHONPATH=<repo>; .\.venv\Scripts\python.exe tools\label_antimemorization.py
Artifacts: outputs/traces/antimemorization_labels.json (per-question verdict) + dataset_b_qids.json (the B list)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_agents.baselines.common import (  # noqa: E402
    GT_PATH, INPUT_PATH, TRACES, extract_json, llm_settings)

MANIFEST = TRACES / os.environ.get("SLC_LABEL_MANIFEST", "sample_manifest.json")   # the current Dataset A qid manifest (before the sample was expanded this pointed at a hard-coded 40-question set; since 2026-10-25 it reads the live sample manifest; override it with SLC_LABEL_MANIFEST to switch to the full 489)
PURE_ROWS = TRACES / "answers_pure_llm.jsonl"
LABEL_OUT = TRACES / os.environ.get("SLC_LABEL_OUT", "antimemorization_labels.json")     # default matches the older artifact; a full run points SLC_LABEL_OUT at a separate file so nothing is overwritten
BQIDS_OUT = TRACES / os.environ.get("SLC_BQIDS_OUT", "dataset_b_qids.json")              # likewise: the full-run B list is stored separately

# probe2 blank-fill template: it never restates the original question, it only gives the experimental
# setting plus the candidates, which disrupts verbatim memory cues
BLANK_TEMPLATE = """Complete the assessment below using only established domain knowledge of {cancer} CRISPR screens. Do NOT guess from question phrasing.

Experimental setting: {setting}
Candidates: {cands}

The differential essentiality observed in this screen is primarily driven by ____ .

Output JSON only: {{"fill": "<one candidate gene>", "confidence": 0~1, "rationale": "<=25 words"}}"""

# membership-specific template: the predicate sentence must be preserved verbatim (otherwise filling
# in a gene name cannot be folded into true/false), but the experimental-setting header of the
# original sentence is stripped and the question is rewritten as a claim under audit -- which
# disrupts verbatim memory cues just the same.
MEMBERSHIP_TEMPLATE = """You are auditing a claimed research finding about {cancer} CRISPR screens, using only established domain knowledge. Do NOT guess from phrasing.

Claim under audit: "{predicate}"
Gene in question: {gene}

Is this claim correct, given what is known about {gene} and {cancer}?

Output JSON only: {{"fill": "true" or "false", "confidence": 0~1, "rationale": "<=25 words"}} (fill="true" means the claim holds, "false" means it does not)."""


def _setting(rec: dict) -> str:
    """Rebuild the experimental-setting text from constraints (without the original question)."""
    c = rec.get("constraints", {}) or {}
    parts = [f"{c.get('cancer', '')}", f"{c.get('species', '')}",
             f"{c.get('cell_line', '') or c.get('tissue', '')} as {c.get('sample_type', 'sample')}",
             f"{c.get('treatment', '')} treatment", f"{c.get('method', '')} method"]
    parts = [p.strip() for p in parts if p and p.strip()]
    q = rec.get("question", "")
    head = q.split("Which")[0].split("Is ")[0].split("Between")[0]
    return "; ".join(parts) if parts else head.strip().rstrip(".")


def _pick(row: dict | None, bucket: str):
    if not row:
        return None
    if bucket == "pairwise":
        return str(row.get("winner") or "").upper() or None
    a = row.get("answer")
    return None if a is None else bool(a)


def _probe2_correct(rec: dict, gt: dict, parsed: dict) -> bool | None:
    bucket = rec["bucket"]
    fill = str((parsed or {}).get("fill", "")).strip()
    if not fill:
        return None
    cands = [c.upper() for c in rec.get("candidates", [])]
    if bucket == "pairwise":
        f = fill.upper()
        # allow phrasings like "SLC1A6 rather than SLC25A1": take the first candidate that hits
        f = next((c for c in cands if c in f), "")
        if not f:
            return None
        return f == str(gt.get("winner", "")).upper()
    m = fill.lower()
    if m in {"true", "false"}:
        return (m == "true") == bool(gt.get("answer"))
    # tolerate free-form model output: test the negation first ("not true" contains the substring
    # "true", so the order must not be reversed)
    if re.search(r"\b(false|incorrect|no|not|wrong)\b", m):
        return False == bool(gt.get("answer"))
    if re.search(r"\b(true|correct|yes|holds)\b", m):
        return True == bool(gt.get("answer"))
    return None


def _membership_predicate(rec: dict) -> str:
    """Take the core predicate of a membership question (the trailing 'Is SLC.. inhibit ...?'), and
    rewrite it as a claim under audit (changing the frame = disrupting verbatim memory, while the
    meaning must be preserved exactly)."""
    q = rec.get("question", "")
    m = re.search(r"(Is\s+\S+\s+.*)$", q, re.S)
    core = (m.group(1) if m else q).strip()
    if core.endswith("?"):
        core = core[:-1]
    if core.lower().startswith("is "):
        core = "It is true that " + core[3:] + "."
    return core


async def probe2_once(cfg: dict, rec: dict) -> dict:
    timeout = float(os.environ.get("SLC_PROBE2_TIMEOUT", "60"))
    from openai import AsyncOpenAI
    cands = [c.upper() for c in rec.get("candidates", [])]
    if rec["bucket"] == "pairwise":
        prompt = BLANK_TEMPLATE.format(
            cancer=rec.get("constraints", {}).get("cancer", "tumor"),
            setting=_setting(rec), cands=", ".join(cands))
    else:
        prompt = MEMBERSHIP_TEMPLATE.format(
            cancer=rec.get("constraints", {}).get("cancer", "tumor"),
            predicate=_membership_predicate(rec), gene=cands[0])
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=timeout, max_retries=0)
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(model=cfg["model"],
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0.2),
            timeout=timeout)
        data = extract_json(rsp.choices[0].message.content or "")
        return {"ok": bool(data), "raw": (rsp.choices[0].message.content or "")[:300],
                "fill": (data or {}).get("fill"), "parsed": data}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass


async def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    a_qids = list(manifest["qids"])
    inputs = {r["qid"]: r for r in json.loads(INPUT_PATH.read_text(encoding="utf-8"))}
    gt_all = json.loads(GT_PATH.read_text(encoding="utf-8"))
    pure: dict[str, dict] = {}
    for line in PURE_ROWS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            pure[r["qid"]] = r            # a later line overwrites an earlier one (same convention as the baseline pending handling)

    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        raise SystemExit("llm_settings is missing base_url/api_key, so probe2 cannot run")

    # resume: rows with the same template_version and a valid probe2 verdict (True/False) are reused
    TEMPLATE_VERSION = "2"      # template 2 = membership switched to the claim-under-audit frame (the template-1 blank fill is not convertible for membership)
    prev_labels: dict[str, dict] = {}
    prev_tv = None
    if LABEL_OUT.exists():
        try:
            old = json.loads(LABEL_OUT.read_text(encoding="utf-8"))
            prev_tv = str(old.get("template_version"))
            prev_labels = {l["qid"]: l for l in old.get("labels", [])}
        except Exception:  # noqa BLE001
            prev_tv = None

    # concurrency + incremental writes (so a whole round is never wasted): a .partial sidecar that
    # appends as soon as a question finishes, which makes the run resumable
    partial_path = LABEL_OUT.with_name(LABEL_OUT.stem + ".partial.jsonl")
    if partial_path.exists():
        for ln in partial_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                try:
                    d = json.loads(ln)
                    if d.get("qid"):
                        prev_labels[d["qid"]] = d          # the partial file supersedes LABEL_OUT (fresher)
                except Exception:  # noqa BLE001
                    pass

    conc = max(1, int(os.environ.get("SLC_PROBE2_CONC", "6")))
    sem = asyncio.Semaphore(conc)
    wrlock = asyncio.Lock()
    done = {"n": 0}

    async def _one(idx: int, qid: str):
        rec = inputs.get(qid)
        gt = gt_all.get(qid)
        if rec is None or gt is None:
            return idx, {"qid": qid, "class": "missing_input"}, "missing_input", False
        old_l = prev_labels.get(qid)
        if (prev_tv == TEMPLATE_VERSION and old_l and old_l.get("class") != "missing_input"
                and old_l.get("probe2", {}).get("correct") in (True, False)
                and old_l.get("probe1", {}).get("correct") in (True, False)):
            return idx, old_l, old_l["class"], True
        p1 = _pick(pure.get(qid), rec["bucket"])
        c1 = None if p1 is None else (
            p1 == str(gt.get("winner", "")).upper() if rec["bucket"] == "pairwise"
            else p1 == bool(gt.get("answer")))
        async with sem:
            r2 = await probe2_once(cfg, rec)
        c2 = _probe2_correct(rec, gt, r2.get("parsed") or {}) if r2.get("ok") else None
        if c1 is None and c2 is None:
            cls = "probe_error"
        elif c1 and c2:
            cls = "stable"
        elif c1 or c2:
            cls = "unstable"
        else:
            cls = "novel"
        lab = {
            "qid": qid, "bucket": rec["bucket"], "gt": gt,
            "probe1": {"answer": p1, "correct": c1},
            "probe2": {"ok": r2.get("ok"), "answer": (r2.get("parsed") or {}).get("fill"),
                       "correct": c2, "error": r2.get("error")},
            "class": cls,
        }
        async with wrlock:
            with open(partial_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(lab, ensure_ascii=False) + "\n")
            done["n"] += 1
            print(f"[{done['n']:>3}/{len(a_qids)}] {cls:<12} probe1={c1} probe2={c2} {qid[:60]}", flush=True)
        return idx, lab, cls, False

    results = await asyncio.gather(*(_one(i, q) for i, q in enumerate(a_qids, 1)),
                                   return_exceptions=True)
    labels = []
    counters = {"stable": 0, "novel": 0, "unstable": 0, "probe_error": 0, "missing_input": 0}
    n_reused = 0
    for r in results:
        if isinstance(r, Exception):
            print(f"[gather-exc] {type(r).__name__}: {str(r)[:150]}", flush=True)
            continue
        idx, lab, cls, reused = r
        labels.append(lab)
        counters[cls] = counters.get(cls, 0) + 1
        if reused:
            n_reused += 1

    b_qids = [l["qid"] for l in labels if l["class"] in ("novel", "unstable")]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LABEL_OUT.write_text(json.dumps({
        "generated_at": now, "dataset_a_n": len(a_qids), "counters": counters,
        "template_version": TEMPLATE_VERSION, "reused_rows": n_reused,
        "probe_model": cfg["model"], "note": "GT is read only inside this labelling tool for comparison; B := novel union unstable",
        "labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8")
    BQIDS_OUT.write_text(json.dumps({
        "generated_at": now, "definition": "pure-LLM double probe (original question text reused + blank-fill paraphrase) that is not correct on both",
        "counters": counters, "b_qids": b_qids}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nclass counts: {counters}\nDataset B: {len(b_qids)}/{len(a_qids)} -> {BQIDS_OUT}")


if __name__ == "__main__":
    asyncio.run(main())
