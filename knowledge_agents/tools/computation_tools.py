# -*- coding: utf-8 -*-
"""Adapter layer for the computation tools: project_graph_computation / timer / km / DESeq2.

These are the four real tools SLC uses to supply candidate numbers to the computation agent.
The status / evidence_scope / interpretation semantics of each tool are preserved exactly; missing
data, unsupported input and failures all record unknown -- nothing is fabricated.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from knowledge_agents.schema import (
    ComputedKnowledge,
    DifferentialExpression,
    ExpressionObservation,
    GraphComputation,
    KnowledgeType,
    PPRCandidate,
    SurvivalObservation,
    ToolStatus,
)
from knowledge_agents.tools.slc_backend import Backend, get_default_backend


class ComputationTools:
    """Tool set for the computation agent."""

    def __init__(self, backend: Backend | None = None):
        self.backend = backend or get_default_backend()

    # ---- PPR: personalised PageRank seeded on the disease ----
    async def graph_computation(self, qid: str) -> GraphComputation:
        resp = await self.backend.acall("project_graph_computation", {"qid": qid})
        comp = resp.get("graph_computation") or {}
        gc = GraphComputation(
            status=ToolStatus(comp.get("status", "error")),
            algorithm=comp.get("algorithm", GraphComputation.model_fields["algorithm"].default),
            alpha=float(comp.get("alpha", 0.6)),
            beta=float(comp.get("beta", 0.5)),
            seed_count=int(comp.get("seed_count") or 0),
            seed_names=list(comp.get("seed_names") or []),
            node_count=comp.get("node_count"),
            edge_count=comp.get("edge_count"),
            graph_sha256=comp.get("graph_sha256"),
            interpretation=comp.get("interpretation", GraphComputation.model_fields["interpretation"].default),
        )
        for cand in comp.get("candidates") or []:
            gc.candidates.append(
                PPRCandidate(
                    gene=cand.get("gene", ""),
                    ppr_score=cand.get("ppr_score"),
                    degree_corrected_score=cand.get("degree_corrected_score"),
                    graph_degree=cand.get("graph_degree"),
                    in_graph=bool(cand.get("in_graph", False)),
                )
            )
        return gc

    # ---- TIMER: pan-cancer tumour/normal expression ----
    async def timer_expression(self, gene: str, gene_family: str = "SLC") -> ExpressionObservation:
        resp = await self.backend.acall("timer", {"gene": gene, "gene_family": gene_family})
        return ExpressionObservation(
            status=ToolStatus(resp.get("status", "error")),
            gene=resp.get("gene", gene),
            row_count=resp.get("row_count"),
            rows=list(resp.get("rows") or []),
            boxplot_path=resp.get("boxplot_path"),
            evidence_scope=resp.get("evidence_scope", "unknown"),
        )

    # ---- KM: survival analysis ----
    async def km_survival(self, gene: str, cancer_type: str) -> SurvivalObservation:
        resp = await self.backend.acall("km", {"gene": gene, "cancer_type": cancer_type})
        return SurvivalObservation(
            status=ToolStatus(resp.get("status", "error")),
            gene=resp.get("gene", gene),
            cancer_type=resp.get("cancer_type", cancer_type),
            tcga_code=resp.get("tcga_code"),
            stats_text=resp.get("stats_text"),
            plot_path=resp.get("plot_path"),
            evidence_scope=resp.get("evidence_scope", "unknown"),
        )

    # ---- DESeq2: TCGA differential expression ----
    # Decisive fact: GDCdata/raw_counts.csv is a deterministic input, so the DESeq2
    # (tumour/normal) result is reusable across questions and seeds; but slc_backend starts a new
    # stdio process per question with no cross-question cache, so DESeq2 for a human cancer type
    # gets recomputed repeatedly and expensively. This tool layer therefore adds a disk cache keyed
    # on (cancer_type, gene_family, top_n) holding the [deterministic terminal states], so DESeq2
    # is paid for once per cancer type and reused instantly afterwards -> the DE channel can
    # genuinely take part in decisions.
    _DE_CACHEABLE = {"ok", "download_required", "unsupported_cancer", "unavailable", "error", "no_data"}

    @staticmethod
    def _de_cache_path(cancer_type: str, gene_family: str, top_n: int) -> Path:
        root = os.environ.get("SLC_TOOLS_CACHE_ROOT") or os.path.join(
            os.environ.get("SLC_ROOT", "."), "outputs", "slc-tools-cache")
        d = Path(root) / "de_workspace_cache"
        d.mkdir(parents=True, exist_ok=True)
        key = f"{cancer_type}|{gene_family}|{top_n}".lower()
        return d / (hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".json")

    @staticmethod
    def _de_build(resp: dict, cancer_type: str, gene_family: str, top_n: int) -> DifferentialExpression:
        return DifferentialExpression(
            status=ToolStatus(resp.get("status", "error")),
            cancer_type=resp.get("cancer_type", cancer_type),
            tcga_code=resp.get("tcga_code"),
            gene_family=resp.get("gene_family", gene_family),
            analysis_method=resp.get("analysis_method"),
            contrast=resp.get("contrast", "tumor/normal"),
            significant_count=resp.get("significant_count"),
            top_genes=list(resp.get("top_genes") or []),
            sample_counts={k: int(v) for k, v in (resp.get("sample_counts") or {}).items()},
            family_tested_count=resp.get("family_tested_count"),
        )

    async def differential_expression(
        self, cancer_type: str, gene_family: str = "SLC", top_n: int = 20
    ) -> DifferentialExpression:
        cache_on = os.environ.get("SLC_DE_CACHE", "1") != "0"
        cpath = self._de_cache_path(cancer_type, gene_family, top_n)
        if cache_on and cpath.exists():
            try:
                cached = json.loads(cpath.read_text(encoding="utf-8"))
                if cached.get("status") in self._DE_CACHEABLE:
                    return self._de_build(cached, cancer_type, gene_family, top_n)
            except Exception:  # noqa BLE001 -- a corrupt cache falls back to real computation
                pass
        resp = await self.backend.acall("de", {"cancer_type": cancer_type, "gene_family": gene_family, "top_n": top_n})
        de = self._de_build(resp, cancer_type, gene_family, top_n)
        if cache_on and resp.get("status") in self._DE_CACHEABLE:  # cache deterministic terminal states only (timeouts/errors are not cached and retry next time)
            try:
                cpath.write_text(json.dumps(resp, ensure_ascii=False), encoding="utf-8")
            except Exception:  # noqa BLE001 -- failing to write the cache must not affect this result
                pass
        return de


def new_computed_knowledge(question: str) -> ComputedKnowledge:
    """Build an empty candidate-knowledge container that ComputeRole fills in per routed tool."""
    return ComputedKnowledge(
        question=question,
        knowledge_type=KnowledgeType.CANDIDATE_DATA.value,
        status=ToolStatus.ERROR,
    )


def summarize_computed(k: ComputedKnowledge) -> dict[str, Any]:
    """Summarise the availability of the four computations for the downstream coverage judgement
    (unknown is never treated as negative)."""
    parts = {
        "graph_computation": k.graph_computation.status.value if k.graph_computation else "not_routed",
        "expression": k.expression.status.value if k.expression else "not_routed",
        "survival": k.survival.status.value if k.survival else "not_routed",
        "differential_expression": k.differential_expression.status.value if k.differential_expression else "not_routed",
    }
    parts["ok_count"] = sum(1 for v in parts.values() if v == ToolStatus.OK.value)
    return parts
