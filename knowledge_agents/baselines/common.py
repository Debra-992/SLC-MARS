# -*- coding: utf-8 -*-
"""Shared pieces for baseline runners: standard library only, so any standalone venv (autogen/mdagents) can import them straight from the path.

Fairness conventions:
  * Same question set: read outputs/traces/baselines_input.json, exported by export_inputs.py from the same sample
    manifest used for the headline comparison (n=489, seed5, candidate genes included).
  * Same instruction: every pure-reasoning baseline uses the same _BASELINE_PROMPT (verbatim identical to the stock MetaGPT baseline).
  * Same endpoint: the same model through the openai-compatible interface; credentials come from SLC_LLM_* environment
    variables (injected by secrets.local.ps1; the checked-in config2.yaml holds placeholders only).
  * Answer schema aligned with the pipeline's Answer: {qid,bucket,winner,answer,confidence,abstained,rationale}.
  * Honest failure: endpoint unreachable / parse failure -> abstained=true; a winner is never fabricated.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "outputs" / "traces"
INPUT_PATH = TRACES / "baselines_input.json"  # no ground truth here; readable by any runner
GT_PATH = TRACES / "baselines_gt.json"  # scoring inside merge_baselines.py only; runners must not read it


def load_inputs() -> list[dict]:
    return json.loads(INPUT_PATH.read_text(encoding="utf-8"))


def answers_path(system: str) -> Path:
    return TRACES / f"answers_{system}.jsonl"


def load_done(system: str) -> dict[str, dict]:
    """resume: qid -> row already on disk (a later write overwrites an earlier one)."""
    out: dict[str, dict] = {}
    p = answers_path(system)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    out[r["qid"]] = r
                except Exception:  # noqa BLE001
                    continue
    return out


def pending(system: str, inputs: list[dict]) -> list[dict]:
    """Skip only rows that succeeded; error rows (timeouts and other transient faults) rerun, same qid the later write wins."""
    done = load_done(system)
    return [r for r in inputs
            if not (r["qid"] in done and not (done[r["qid"]].get("error") or ""))]


def append_answer(system: str, rec: dict) -> None:
    with answers_path(system).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# Verbatim copy of knowledge_agents/baseline_metagpt._BASELINE_PROMPT (edit one, sync the other)
BASELINE_PROMPT = """You are a biomedical question-answering assistant. Answer from your own knowledge only, using no tools, no retrieval and no external data.
You must commit to one answer (this is a native LLM baseline, abstention is not allowed): pairwise picks exactly one candidate, membership answers true or false.

Question (bucket={bucket}): {question}
{candidates_line}
Output JSON only (abstained is always false):
- pairwise: {{"winner": "<one of the candidate genes>", "confidence": 0~1, "abstained": false, "rationale": "..."}}
- membership: {{"answer": true/false, "confidence": 0~1, "abstained": false, "rationale": "..."}}
"""


def candidates_line(rec: dict) -> str:
    cands = rec.get("candidates") or []
    if rec.get("bucket") == "pairwise":
        return f"candidate genes: {cands}"
    return f"target gene: {cands[:1]}"


def build_prompt(rec: dict, context: str = "") -> str:
    base = BASELINE_PROMPT.format(bucket=rec["bucket"], question=rec["question"],
                                  candidates_line=candidates_line(rec))
    return (context + "\n\n" + base) if context else base


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def normalize(rec: dict, data: dict | None, *, system: str, error: str = "") -> dict:
    """Raw LLM JSON -> standard answer row. A winner outside the candidate list, or a missing field, becomes an honest abstain."""
    bucket = rec["bucket"]
    cands_u = {c.upper() for c in (rec.get("candidates") or [])}
    out = {"qid": rec["qid"], "bucket": bucket, "system": system,
           "winner": None, "answer": None, "confidence": 0.0,
           "abstained": True, "rationale": "", "error": error}
    if data:
        abst = bool(data.get("abstained"))
        w = data.get("winner") if bucket == "pairwise" else None
        a = data.get("answer") if bucket == "membership" else None
        if w and str(w).upper() not in cands_u:
            w = None
            abst = True
        if w is None and bucket == "pairwise":
            abst = True
        if a is None and bucket == "membership":
            abst = True
        try:
            conf = float(data.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        out.update(winner=w, answer=bool(a) if (bucket == "membership" and a is not None) else None,
                   confidence=conf, abstained=abst, rationale=str(data.get("rationale", ""))[:800])
    return out


# ---------------------------------------------------------------- LLM endpoint (env first, openai-compatible)
# Credentials never live in the repo: config2.yaml keeps placeholders, secrets.local.ps1 injects the real values into SLC_LLM_* environment variables.
_LLM_UNSET = "REPLACE_VIA_SECRETS_LOCAL_PS1"


def llm_settings() -> dict:
    """Resolve the three LLM endpoint keys (shared by runners in standalone venvs).

    Priority: environment variables SLC_LLM_BASE_URL / SLC_LLM_API_KEY / SLC_LLM_MODEL
    (exported by `secrets.local.ps1`, a shell profile or a CI secret store)
    > then the llm block of config2.yaml. Placeholders that were never substituted count as unset (empty string),
    so the caller's "llm.base_url/api_key not found" branch raises a readable error
    instead of sending a literal placeholder off as an API key and getting an unexplained 401.
    """
    cfg = ROOT / "config2.yaml"
    txt = cfg.read_text(encoding="utf-8", errors="replace") if cfg.exists() else ""
    out = {"base_url": "", "api_key": "", "model": ""}
    in_llm = False
    for line in txt.splitlines():
        if re.match(r"^llm:\s*$", line):
            in_llm = True
            continue
        if in_llm:
            if line and not line.startswith((" ", "\t")):
                break
            m = re.match(r"\s+(base_url|api_key|model):\s*(\S.*)$", line)
            if m:
                out[m.group(1)] = m.group(2).strip().strip("\"'")
    for k in out:
        envv = os.environ.get("SLC_LLM_" + k.upper(), "")
        if envv:
            out[k] = envv
        elif out[k] == _LLM_UNSET:
            out[k] = ""
    return out


# ---------------------------------------------------------------- token metering (uniform across baselines)
def usage_triplet(rsp) -> tuple[int, int, int]:
    """Safely pull (prompt, completion, total) tokens from an openai response object; anything missing counts as 0."""
    u = getattr(rsp, "usage", None)
    if u is None:
        return (0, 0, 0)
    p = int(getattr(u, "prompt_tokens", 0) or 0)
    c = int(getattr(u, "completion_tokens", 0) or 0)
    t = int(getattr(u, "total_tokens", 0) or (p + c))
    return (p, c, t)


class TokenSink:
    """Process-wide shared accumulator (guarded by a lock). The autogen and mdagents pools each run a single worker,
    so resetting at the start of a question and snapshotting at the end gives exactly that question's usage; autogen may
    issue create calls from a child thread, so thread-local storage would hide those adds and undercount."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._v = [0, 0, 0, 0]   # prompt, completion, total, calls

    def reset(self) -> None:
        with self._lock:
            self._v = [0, 0, 0, 0]

    def add(self, p: int, c: int, t: int) -> None:
        with self._lock:
            self._v[0] += p; self._v[1] += c; self._v[2] += t; self._v[3] += 1

    def snapshot(self) -> dict:
        with self._lock:
            v = list(self._v)
        return {"prompt": v[0], "completion": v[1], "total": v[2], "calls": v[3]}


def install_openai_token_tracking(sink: TokenSink) -> bool:
    """Monkeypatch the synchronous openai Chat Completions.create and add every call's usage to the sink.
    Returns whether the patch installed (False if the library layout differs; callers then degrade). Return values and exception semantics are untouched."""
    try:
        from openai.resources.chat.completions import Completions
    except Exception:  # noqa BLE001
        return False
    real = Completions.create
    if getattr(real, "__tok_wrapped__", False):
        return True

    def create(self, *a, **kw):
        rsp = real(self, *a, **kw)
        try:
            if not kw.get("stream"):
                p, c, t = usage_triplet(rsp)
                sink.add(p, c, t)
        except Exception:  # noqa BLE001 -- metering must never interfere with answering
            pass
        return rsp

    create.__tok_wrapped__ = True
    Completions.create = create
    return True


def install_metagpt_token_tracking(sink: TokenSink) -> bool:
    """Mirror MetaGPT's CostManager.update_cost into the sink (called once per LLM request).
    MetaGPT's provider uses AsyncOpenAI with stream=True, so usage is hard to read at the openai layer, but the
    provider always calls update_cost to book the request, which makes that the sturdiest interception point. metagpt is
    imported lazily inside so venvs without it stay clean. Returns whether the patch installed."""
    try:
        from metagpt.utils.cost_manager import CostManager
    except Exception:  # noqa BLE001
        return False
    real = CostManager.update_cost
    if getattr(real, "__tok_wrapped_mg__", False):
        return True

    def update_cost(self, prompt_tokens, completion_tokens, model):
        try:
            p = int(prompt_tokens or 0); c = int(completion_tokens or 0)
            if p + c > 0:
                sink.add(p, c, p + c)
        except Exception:  # noqa BLE001 -- metering must never interfere with answering
            pass
        return real(self, prompt_tokens, completion_tokens, model)

    update_cost.__tok_wrapped_mg__ = True
    CostManager.update_cost = update_cost
    return True


def install_temperature_zero() -> bool:
    """With env SLC_FORCE_TEMP0=1, force temperature=0 on every openai chat.completions.create in this process
    (sync and async alike). Every pipeline LLM call goes through AsyncOpenAI.chat.completions.create
    (prior/debate/decision/L2) and MetaGPT's provider does too, so one central interception point pins down
    all sampling and keeps LLM nondeterminism out of the R2 on/off A/B pairing. Unset means no-op (zero side effects).
    Only the temperature kwarg changes; return values and exception semantics are untouched."""
    if os.environ.get("SLC_FORCE_TEMP0", "0").lower() not in {"1", "true", "on", "yes"}:
        return False
    try:
        from openai.resources.chat.completions import Completions, AsyncCompletions
    except Exception:  # noqa BLE001
        return False
    real_s = Completions.create
    if not getattr(real_s, "__temp0__", False):
        def create(self, *a, **kw):
            kw["temperature"] = 0
            return real_s(self, *a, **kw)
        create.__temp0__ = True
        Completions.create = create
    real_a = AsyncCompletions.create
    if not getattr(real_a, "__temp0__", False):
        async def acreate(self, *a, **kw):
            kw["temperature"] = 0
            return await real_a(self, *a, **kw)
        acreate.__temp0__ = True
        AsyncCompletions.create = acreate
    return True
