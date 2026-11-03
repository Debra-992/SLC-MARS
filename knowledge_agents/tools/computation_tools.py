# -*- coding: utf-8 -*-
"""计算类工具适配层：对接 project_graph_computation / timer / km / DESeq2。

对应 SLC 计算智能体供给候选数值的 4 个真实工具。严格保留各工具的
status / evidence_scope / interpretation 语义，缺数据/不支持/失败均记未知，不伪造。
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
    """计算智能体的工具集。"""

    def __init__(self, backend: Backend | None = None):
        self.backend = backend or get_default_backend()

    # ---- PPR：疾病种子加权个性化 PageRank ----
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

    # ---- TIMER：泛癌肿瘤/正常表达 ----
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

    # ---- KM：生存分析 ----
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

    # ---- DESeq2：TCGA 差异表达 ----
    # 决定性事实：GDCdata/raw_counts.csv 是确定性输入，DESeq2(肿瘤/正常)结果可跨题/跨种子复用；
    # 而 slc_backend 每题新建 stdio 进程、无跨题缓存，故人类癌种的 DESeq2 会被反复重算且很贵。
    # 这里在工具层加一层磁盘缓存：按 (cancer_type, gene_family, top_n) 缓存【确定性终态】，
    # 使 DESeq2 每个癌种只付一次代价，之后即时复用 -> DE 通道得以真正参与。
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
            except Exception:  # noqa BLE001 —— 缓存损坏则回退真实计算
                pass
        resp = await self.backend.acall("de", {"cancer_type": cancer_type, "gene_family": gene_family, "top_n": top_n})
        de = self._de_build(resp, cancer_type, gene_family, top_n)
        if cache_on and resp.get("status") in self._DE_CACHEABLE:  # 只缓存确定性终态（超时/异常不缓存，下次重试）
            try:
                cpath.write_text(json.dumps(resp, ensure_ascii=False), encoding="utf-8")
            except Exception:  # noqa BLE001 —— 写缓存失败不影响本次结果
                pass
        return de


def new_computed_knowledge(question: str) -> ComputedKnowledge:
    """构造一条空的候选知识容器，供 ComputeRole 按需填充各计算子结构。"""
    return ComputedKnowledge(
        question=question,
        knowledge_type=KnowledgeType.CANDIDATE_DATA.value,
        status=ToolStatus.ERROR,
    )


def summarize_computed(k: ComputedKnowledge) -> dict[str, Any]:
    """汇总四类计算的可用性，供下游判断证据覆盖度（不把未知当阴性）。"""
    parts = {
        "graph_computation": k.graph_computation.status.value if k.graph_computation else "not_routed",
        "expression": k.expression.status.value if k.expression else "not_routed",
        "survival": k.survival.status.value if k.survival else "not_routed",
        "differential_expression": k.differential_expression.status.value if k.differential_expression else "not_routed",
    }
    parts["ok_count"] = sum(1 for v in parts.values() if v == ToolStatus.OK.value)
    return parts
