# -*- coding: utf-8 -*-
"""Core action of the computation agent: routes the four computation tools from the intent in the
question text and produces candidate numeric knowledge.

Follows the real SLC on-demand policy (on-demand-rules-v1):
  - PPR: unsigned Top-K ranking prior; runs for membership/pairwise/topk alike
  - TIMER/KM: run by default on the candidate genes for membership/pairwise/topk
  - DESeq2: "relevant means wanted" -- attempted whenever the bucket is supported and a
    cancer_type exists (the first call can be heavy; on timeout it is honestly skipped, not faked)
Anything not routed or unavailable stays unknown: numbers are never fabricated and unknown is
never read as negative.
"""
from __future__ import annotations

import asyncio
import os
import re

from metagpt.actions import Action

from knowledge_agents.schema import ComputedKnowledge, ToolStatus
from knowledge_agents.tools.computation_tools import (
    ComputationTools,
    new_computed_knowledge,
    summarize_computed,
)
from knowledge_agents.tools import paralog as paralog_mod

# Intent heuristics on the question text (the real pipeline triggers on routing rules, not on
# model function calling). Patterns are English because the released question pool is English.
_EXPR_RE = re.compile(r"express|TIMER|differential expression|log2fc|fold[- ]change", re.I)
_SURVIVAL_RE = re.compile(r"survival|prognosis|kaplan|HR\b|hazard", re.I)
_TOPK_RE = re.compile(r"\btop[- ]?k\b|rank|prioritis|candidate priorit", re.I)


class Computation(Action):
    """Computation action: produces ComputedKnowledge."""

    name: str = "Computation"

    def __init__(self, tools: ComputationTools | None = None, **kwargs):
        super().__init__(**kwargs)
        self.tools = tools or ComputationTools()

    @staticmethod
    def _infer_field(question: str, key: str) -> str | None:
        m = re.search(rf"{key}=([A-Za-z0-9_.\-]+)", question)
        return m.group(1) if m else None

    def route(self, question: str, task_bucket: str = "") -> dict[str, bool]:
        """Return whether each computation tool needs to be routed.

        TIMER/KM/PPR run on local caches or graph computation and are cheap to control, so all
        three buckets (membership/pairwise/topk) run them on the candidate genes by default --
        PPR in particular computes the [gene-to-disease network proximity], which membership
        also uses (it used to be routed only for topk, leaving membership with the weak KG signal).
        DESeq2 follows "relevant means wanted": as long as the bucket is supported and a
        cancer_type exists (differential expression is relevant to every candidate gene of that
        cancer type) it runs by itself, no longer only when the question explicitly asks for
        differential expression; results are cached per cancer type, so the cost stays bounded.
        """
        bucket = (task_bucket or "").lower()
        supported_bucket = bucket in {"membership", "pairwise", "topk"}
        want_expr = bool(_EXPR_RE.search(question)) or supported_bucket      # TIMER
        want_de = supported_bucket or bool(_EXPR_RE.search(question))          # DESeq2: relevant means wanted (includes supported buckets)
        want_survival = bool(_SURVIVAL_RE.search(question)) or supported_bucket  # KM
        want_ppr = supported_bucket or bool(_TOPK_RE.search(question))        # gene-disease proximity
        return {"ppr": want_ppr, "expression": want_expr, "de": want_de, "survival": want_survival}

    async def run(
        self,
        question: str,
        *,
        qid: str | None = None,
        gene: str | None = None,
        genes: list[str] | None = None,
        cancer_type: str | None = None,
        task_bucket: str = "",
    ) -> ComputedKnowledge:
        qid = qid or self._infer_field(question, "qid")
        cancer_type = cancer_type or self._infer_field(question, "cancer")

        # Candidate gene list: explicit genes first, then the single gene, then the question text
        gene_list = [g for g in (genes or []) if g]
        if gene and gene not in gene_list:
            gene_list = [gene, *gene_list]
        if not gene_list:
            inferred = self._infer_field(question, "gene")
            gene_list = [inferred] if inferred else []
        primary_gene = gene_list[0] if gene_list else None
        gene = gene or primary_gene

        ck = new_computed_knowledge(question)
        need = self.route(question, task_bucket)
        skipped: list[str] = []

        # PPR (needs the qid disease seeds)
        if need["ppr"]:
            if qid:
                ck.graph_computation = await self.tools.graph_computation(qid)
            else:
                skipped.append("ppr:no_qid")
        else:
            skipped.append("ppr:not_routed")

        # TIMER expression (per candidate gene, cheap because of the cache)
        if need["expression"]:
            if gene_list:
                for g in gene_list:
                    obs = await self.tools.timer_expression(g)
                    ck.expression_by_gene[g] = obs
                ck.expression = ck.expression_by_gene.get(primary_gene)
            else:
                skipped.append("timer:no_gene")
        else:
            skipped.append("timer:not_routed")

        # DESeq2 differential expression (cancer-type level, needs cancer_type; "relevant means wanted".
        # The tool layer now disk-caches it, so each cancer type is paid for once -- this wait ceiling
        # only backstops the *first real computation* and must leave room for DESeq2 plus reading the
        # counts, aligned with SLC_MCP_R_TIMEOUT)
        if need["de"]:
            if cancer_type:
                de_timeout = float(os.environ.get("SLC_DE_TIMEOUT", "900"))
                try:
                    ck.differential_expression = await asyncio.wait_for(
                        self.tools.differential_expression(
                            cancer_type, top_n=int(os.environ.get("SLC_DE_TOPN", "400"))),
                        timeout=de_timeout,
                    )
                except asyncio.TimeoutError:
                    skipped.append(f"deseq2:timeout>{de_timeout:.0f}s")
                except Exception as e:  # noqa BLE001 -- DE unavailable means an honest skip, never a blocked pipeline
                    skipped.append(f"deseq2:{type(e).__name__}")
            else:
                skipped.append("deseq2:no_cancer")
        else:
            skipped.append("deseq2:not_routed")

        # Paralog buffering: if this cancer type's DESeq2 panel is already available, estimate the
        # functional redundancy of each candidate gene from its [co-expressed same-subfamily
        # paralogs] (high redundancy -> the knockout is easily buffered -> weakens an essentiality call).
        # This is one-sided evidence: no panel means unknown, and "no paralogs" never manufactures a positive.
        if gene_list:
            panel = paralog_mod.panel_from_de(ck.differential_expression)
            if panel:
                red, partners = paralog_mod.redundancy_map(
                    gene_list, panel, k_sat=float(os.environ.get("SLC_PARALOG_KSAT", "4")))
                for g, r in red.items():
                    ck.paralog_redundancy[g] = r
                    ck.paralog_partners[g] = partners.get(g, [])
            else:
                skipped.append("paralog:no_expression_panel")

        # KM survival (per candidate gene, needs gene + cancer_type)
        if need["survival"]:
            if gene_list and cancer_type:
                for g in gene_list:
                    obs = await self.tools.km_survival(g, cancer_type)
                    ck.survival_by_gene[g] = obs
                ck.survival = ck.survival_by_gene.get(primary_gene)
            else:
                skipped.append("km:no_gene_or_cancer")
        else:
            skipped.append("km:not_routed")

        ck.skipped_tools = skipped
        stats = summarize_computed(ck)
        ck.status = ToolStatus.OK if stats["ok_count"] else ToolStatus.UNAVAILABLE
        ck.summary = (
            f"candidate computation done: {stats['ok_count']} tool(s) available"
            f" (ppr={stats['graph_computation']}, timer={stats['expression']}, "
            f"km={stats['survival']}, de={stats['differential_expression']}); "
            f"candidate genes={gene_list}; not routed/unknown: {skipped}. Unknown is not negative."
        )
        return ck
