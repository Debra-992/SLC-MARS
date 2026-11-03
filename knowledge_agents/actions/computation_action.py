# -*- coding: utf-8 -*-
"""计算智能体核心动作：按题面语义路由 4 类计算工具，产出候选数值知识。

对齐 SLC 真实按需策略（on-demand-rules-v1）：
  - PPR：Top-K 无符号排序先验；membership/pairwise/topk 均开跑
  - TIMER/KM：membership/pairwise/topk 对候选基因默认开跑
  - DESeq2：秉持"有关就要"，受支持桶且有 cancer_type 即尝试（首次可能很重，超时则诚实跳过、不伪造）
未路由/不可用一律保持未知，绝不伪造数值、绝不当阴性。
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

# 题面意图启发式关键词（真实管线由路由规则触发，非模型 function calling）
_EXPR_RE = re.compile(r"express|TIMER|差异表达|表达差异|表达", re.I)
_SURVIVAL_RE = re.compile(r"survival|prognosis|预后|生存|HR\b|kaplan", re.I)
_TOPK_RE = re.compile(r"\btop[- ]?k\b|rank|排序|候选优先级", re.I)


class Computation(Action):
    """计算动作：产出 ComputedKnowledge。"""

    name: str = "Computation"

    def __init__(self, tools: ComputationTools | None = None, **kwargs):
        super().__init__(**kwargs)
        self.tools = tools or ComputationTools()

    @staticmethod
    def _infer_field(question: str, key: str) -> str | None:
        m = re.search(rf"{key}=([A-Za-z0-9_.\-]+)", question)
        return m.group(1) if m else None

    def route(self, question: str, task_bucket: str = "") -> dict[str, bool]:
        """返回各计算工具是否需要路由。

        TIMER/KM/PPR 基于本地缓存或图谱计算、代价可控，membership/pairwise/topk
        三类桶都默认对候选基因开跑——其中 PPR 正是“基因↔疾病 网络邻近度”的计算，
        membership 也用它（此前只对 topk 路由，导致 membership 只剩 KG 弱信号）。
        DESeq2 秉持“有关就要”：只要桶受支持且有 cancer_type（差异表达对该癌种所有候选基因相关）
        就自行开跑，不再只在题面显式要求“差异表达”时才路由；结果按癌种级缓存、代价可控。
        """
        bucket = (task_bucket or "").lower()
        supported_bucket = bucket in {"membership", "pairwise", "topk"}
        want_expr = bool(_EXPR_RE.search(question)) or supported_bucket      # TIMER
        want_de = supported_bucket or bool(_EXPR_RE.search(question))          # DESeq2：有关就要（含受支持桶）
        want_survival = bool(_SURVIVAL_RE.search(question)) or supported_bucket  # KM
        want_ppr = supported_bucket or bool(_TOPK_RE.search(question))        # 基因-疾病邻近度
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

        # 候选基因列表：显式 genes 优先，否则单 gene，再否则题面推断
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

        # PPR（需 qid 疾病种子）
        if need["ppr"]:
            if qid:
                ck.graph_computation = await self.tools.graph_computation(qid)
            else:
                skipped.append("ppr:no_qid")
        else:
            skipped.append("ppr:not_routed")

        # TIMER 表达（逐候选基因，缓存代价低）
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

        # DESeq2 差异表达（癌种级，需 cancer_type；"有关就要"。工具层已加磁盘缓存，每癌种只付一次
        # 代价，故这里的等待上限只兜底"首次真实计算"，需给足 DESeq2+读counts 的时间，与 SLC_MCP_R_TIMEOUT 对齐）
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
                except Exception as e:  # noqa BLE001 —— DE 不可用即诚实跳过，不阻断管线
                    skipped.append(f"deseq2:{type(e).__name__}")
            else:
                skipped.append("deseq2:no_cancer")
        else:
            skipped.append("deseq2:not_routed")

        # 旁系缓冲（paralog buffering）：若已拿到该癌种 DESeq2 面板，则据『同亚家族共表达旁系』
        # 估计候选基因功能冗余（冗余高 -> knockout 易被缓冲 -> 削弱必需性判断）。
        # 这是单边证据：拿不到面板即未知，绝不凭"无旁系"制造阳性。
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

        # KM 生存（逐候选基因，需 gene + cancer_type）
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
            f"计算候选完成：可用工具 {stats['ok_count']} 项"
            f"（ppr={stats['graph_computation']}, timer={stats['expression']}, "
            f"km={stats['survival']}, de={stats['differential_expression']}）；"
            f"候选基因={gene_list}；未路由/未知：{skipped}。未知不代表阴性。"
        )
        return ck
