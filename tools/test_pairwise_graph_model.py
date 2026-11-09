# -*- coding: utf-8 -*-
"""Offline test: the pairwise graph-feature logistic wired into the Vega decision layer.
Covers: the model branch picking a winner by argmax, an honest fallback to the heuristic rules when
there is no graph evidence, the pure heuristic path when the env var is unset, and a fallback on a
broken artifact."""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing metagpt.actions pulls in chardet, whose mypyc-compiled extension is blocked by the OS
# application-control policy on some Windows machines. This test only exercises the heuristic scoring
# and never needs chardet, so a pure-Python stub is injected when the real one fails to load; a normal
# environment is left untouched.
try:  # noqa: SIM105
    import chardet  # noqa: F401
except Exception:  # noqa: BLE001
    import types

    _stub = types.ModuleType("chardet")
    _stub.detect = lambda b: {"encoding": "utf-8", "confidence": 1.0, "language": ""}  # type: ignore[attr-defined]
    _stub.__version__ = "0"  # type: ignore[attr-defined]
    sys.modules["chardet"] = _stub

from knowledge_agents.actions import decision_action as da  # noqa: E402
from knowledge_agents.graph_features import PAIR_COLS  # noqa: E402
from knowledge_agents.schema import (  # noqa: E402
    BackgroundKnowledge, ComputedKnowledge, KGEvidence, ToolStatus,
)

ad = da.AnswerDecision()
CK = ComputedKnowledge(question="t")
ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("PASS " if cond else "FAIL ") + name)
    ok += int(cond)
    fail += int(cond is False)


def kg_paths(rows):
    kg = BackgroundKnowledge(question="t", status=ToolStatus.OK)
    kg.evidence.append(KGEvidence(operation="gene_disease_paths", rows=rows))
    return kg


def row(gene, eff="PROMOTION"):
    return {"gene": gene, "disease": "breast cancer(BC)", "rel_AB": eff, "rel_BC": eff,
            "gene_disease_effect": eff, "pathway": "proliferation"}


def write_artifact(coef_over=None, only_col=None):
    """Build a controllable artifact: by default only the n_paths weight is +1 (the rest 0), so the
    winner is the gene with more paths."""
    coef = [0.0] * len(PAIR_COLS)
    if only_col:
        coef[PAIR_COLS.index(only_col)] = 1.0
    if coef_over:
        coef = coef_over
    art = {"task": "pairwise_graph_logistic", "features": list(PAIR_COLS),
           "mu": [0.0] * len(PAIR_COLS), "sd": [1.0] * len(PAIR_COLS),
           "coef": coef, "intercept": 0.0, "C": 0.01, "seed": 5, "train_frac": 0.6,
           "train_gene_rows": 10, "trained_qids": [], "holdout": {"n_questions": 0, "acc": 0.74},
           "no_signal_behavior": "fallback_rules"}
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(art, f, ensure_ascii=False)
    return path


# Clear the cache before every case (the module-level cache keys on path, so an env change already
# swaps the key, but clearing explicitly is safer)
def use_env(path):
    da._PAIRWISE_MODEL_CACHE.clear()
    if path:
        os.environ["SLC_PAIRWISE_MODEL"] = path
    else:
        os.environ.pop("SLC_PAIRWISE_MODEL", None)


try:
    # (1) model branch: A has 3 paths and B has 1 -> the gene favoured by the n_paths weight, A, wins
    p = write_artifact(only_col="n_paths")
    use_env(p)
    kg = kg_paths([row("GPA"), row("GPA"), row("GPA"), row("GPB")])
    a = ad._heuristic("q", kg, CK, "pairwise", ["GPB", "GPA"], "pairwise_positive", {"cancer": "breast"})
    check("model branch picks higher n_paths gene (A)", a.winner == "GPA" and a.abstained is False)
    check("model branch marks the graph-model rationale",
          "[pairwise graph-feature logistic model]" in a.rationale)
    check("model branch confidence = winner prob in [0,1]", 0.0 <= a.confidence <= 1.0)

    # (2) neither candidate has gene_disease_paths (empty evidence) -> fall back to the rules (the model must not invent a winner)
    p2 = write_artifact(only_col="n_paths")
    use_env(p2)
    kg_empty = BackgroundKnowledge(question="t", status=ToolStatus.OK)
    a2 = ad._heuristic("q", kg_empty, CK, "pairwise", ["GX", "GY"], "pairwise_positive", {"cancer": "breast"})
    check("no graph evidence -> fallback (rationale NOT the graph model)",
          "[pairwise graph-feature logistic model]" not in a2.rationale)

    # (3) env unset -> the pure heuristic path (the model is not consulted even with graph evidence)
    use_env(None)
    kg3 = kg_paths([row("GPA"), row("GPA"), row("GPA"), row("GPB")])
    a3 = ad._heuristic("q", kg3, CK, "pairwise", ["GPB", "GPA"], "pairwise_positive", {"cancer": "breast"})
    check("env unset -> pure heuristic rules (no graph-model marker)",
          "[pairwise graph-feature logistic model]" not in a3.rationale)

    # (4) broken artifact (missing / invalid JSON) -> fallback to the rules, no exception raised
    fd, bad = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write("{not valid json")
    use_env(bad)
    try:
        a4 = ad._heuristic("q", kg_paths([row("GPA")]), CK, "pairwise",
                           ["GPA", "GPB"], "pairwise_positive", {"cancer": "breast"})
        check("broken artifact -> fallback, no crash",
              "[pairwise graph-feature logistic model]" not in a4.rationale)
    except Exception as e:  # noqa BLE001
        check("broken artifact -> fallback, no crash", False)
        print("   raised:", type(e).__name__, e)
    finally:
        os.remove(bad)

    # (5) the real artifact (if present) loads and scores the real distribution (smoke, no hard-coded expectation)
    real = os.path.join("outputs", "traces", "pairwise_graph_model.json")
    if os.path.exists(real):
        use_env(real)
        kgr = kg_paths([row("SLC7A11"), row("SLC7A11"), row("SLC7A5")])
        ar = ad._heuristic("q", kgr, CK, "pairwise", ["SLC7A5", "SLC7A11"],
                           "pairwise_positive", {"cancer": "breast"})
        check("real artifact: decisive, in-candidates winner",
              ar.winner in ("SLC7A5", "SLC7A11") and ar.abstained is False)

    print(f"\ntotal PASS={ok} FAIL={fail}")
    sys.exit(1 if fail else 0)
finally:
    use_env(None)
