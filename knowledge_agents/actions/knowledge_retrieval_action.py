# -*- coding: utf-8 -*-
"""知识检索智能体核心动作：图谱/文献主知识 + text2cypher 补充背景知识。

链路（对齐 SLC 真实策略）：
  1. project_knowledge(qid) 取 Neo4j 三路证据 + MongoDB 文献（主背景知识）
  2. 候选覆盖不足时，kg_schema -> LLM 生成只读 Cypher -> kg_cypher 执行（text2cypher）
  3. 仍缺则 kg_gene_neighbors 补直接邻居
只读守卫与"失败即未知"贯穿全程，绝不伪造。
"""
from __future__ import annotations

import json
import re

from metagpt.actions import Action

from knowledge_agents.schema import BackgroundKnowledge, KGEvidence, ToolStatus
from knowledge_agents.tools.retrieval_tools import RetrievalTools

CODE_BLOCK_RE = re.compile(r"```(?:cypher)?\s*(.*?)```", re.DOTALL)


class KnowledgeRetrieval(Action):
    """知识检索动作：产出 BackgroundKnowledge。"""

    name: str = "KnowledgeRetrieval"

    TEXT2CYPHER_PROMPT: str = """
本图谱是【时间因果图谱】。SLC 基因与疾病【从不直连】，也【不存在名为 relevant 的关系类型】；
它们的关联被具体化为事件节点 RelaEvent，形成 A-B-C 因果三元耦合：
  (g:SLCGene)-[:IS_SOURCE]->(e:RelaEvent)<-[:IS_MEDIATOR]-(p:Pathway)
  (g:SLCGene)-[:IS_SOURCE]->(e:RelaEvent)<-[:IS_TARGET]--(d:Disease)
RelaEvent 属性：rel_AB = SLC→Pathway 的方向(PROMOTION/SUPPRESSION)，
  rel_BC = Pathway→Disease 的方向(PROMOTION/SUPPRESSION)，paper_dois = 文献来源(数组)。
gene→disease 净方向 = 当 rel_AB 与 rel_BC 同向取 PROMOTION、反向取 SUPPRESSION。
真实标签：SLCGene(name,NCBIGeneID,ProteinFunction,ProteinAliases,Transport_type,Tissue_and_cellular_expression)、
  Pathway(name)、Disease(name,diseaseSort)、RelaEvent(rel_AB,rel_BC,paper_dois)、SLCFamily(name)、Protein(name)、Product(name,target)。

实时 Schema（仅作补充校验，非生物学证据）：
{schema}

候选基因：{genes}
题目约束：{context}

请生成【一条只读 Cypher】检索候选基因经 RelaEvent 关联的疾病/通路，并【务必返回 rel_AB、rel_BC 及
复合符号 CASE WHEN e.rel_AB=e.rel_BC THEN 'PROMOTION' ELSE 'SUPPRESSION' END AS gene_disease_effect】。
硬性要求：
- 只允许 MATCH/OPTIONAL MATCH/WITH/RETURN/UNWIND/EXPLAIN 开头
- 禁止 CREATE/MERGE/DELETE/SET/DROP 等写子句，禁止 CALL/APOC/GDS 存储过程，禁止多语句(分号)
- 属性/标签必须严格来自上面的真实 Schema；不得臆造关系类型（如 relevant）或直连 (gene)-[]-(disease)
只返回 ```cypher 代码块```，不要其他文字。
"""

    def __init__(self, tools: RetrievalTools | None = None, **kwargs):
        super().__init__(**kwargs)
        self.tools = tools or RetrievalTools()

    @staticmethod
    def _extract_cypher(rsp: str) -> str:
        match = CODE_BLOCK_RE.search(rsp)
        return (match.group(1) if match else rsp).strip().rstrip(";")

    @staticmethod
    def _infer_qid(question: str) -> str:
        m = re.search(r"qid=([A-Za-z0-9_.\-]+)", question)
        return m.group(1) if m else question.strip()

    async def run(self, question: str) -> BackgroundKnowledge:
        qid = self._infer_qid(question)

        # 1) 主背景知识：项目图谱 + 文献
        kg = await self.tools.project_knowledge(qid)
        if kg.status is not ToolStatus.OK:
            kg.summary = f"主知识不可用（{kg.status.value}），仅尝试 text2cypher 补充。"

        # 2) text2cypher 补充：仅当候选存在但图谱行覆盖不足时触发
        if kg.candidates and self._coverage(kg) < 1:
            schema = await self.tools.kg_schema()
            if schema.get("labels"):
                prompt = self.TEXT2CYPHER_PROMPT.format(
                    schema=json.dumps(schema, ensure_ascii=False)[:2000],
                    genes=", ".join(kg.candidates),
                    context=json.dumps(kg.requested_context, ensure_ascii=False),
                )
                cypher = self._extract_cypher(await self._aask(prompt))
                result = await self.tools.kg_cypher(cypher)
                kg.cypher = result.get("query", cypher)
                kg.cypher_rows = result.get("rows", [])
                kg.schema_used = schema
                kg.evidence.append(
                    KGEvidence(
                        operation="text2cypher",
                        rows=kg.cypher_rows,
                        cypher=kg.cypher,
                        status=ToolStatus(result.get("status", "error")),
                    )
                )

        # 3) 时间因果补强：候选基因仍缺『带复合符号』的疾病路径时，用 kg_gene_neighbors 看穿 RelaEvent 补方向
        await self._signed_neighbor_backfill(kg)

        kg.summary = self._finalize(kg)
        return kg

    async def _signed_neighbor_backfill(self, kg: BackgroundKnowledge) -> None:
        """对没有 gene_disease_paths 符号行的候选基因，调用已『看穿 RelaEvent』的 kg_neighbors，
        把 RELATES_DISEASE 的 rel_AB/rel_BC/effect 归一为 gene_disease_paths 兼容行补进去，
        使纯邻近题也能享受时间因果方向（同源事件，仅在无覆盖时补，避免重复计数）。"""
        covered = {
            str(r.get("gene", "")).upper()
            for ev in kg.evidence if ev.operation == "gene_disease_paths"
            for r in (ev.rows or []) if isinstance(r, dict) and r.get("gene_disease_effect")
        }
        missing = [c for c in (kg.candidates or []) if str(c).upper() not in covered]
        extra: list[dict] = []
        for gene in missing:
            nb = await self.tools.kg_gene_neighbors(gene)
            if str(nb.get("status", "ok")) not in ("ok", "None", ""):
                continue
            for r in nb.get("neighbors") or []:
                if not isinstance(r, dict) or r.get("relationship") != "RELATES_DISEASE":
                    continue
                eff = r.get("effect")
                if eff not in ("PROMOTION", "SUPPRESSION"):
                    continue
                extra.append({
                    "gene": gene, "disease": r.get("disease") or r.get("neighbor"),
                    "rel_AB": r.get("rel_AB"), "rel_BC": r.get("rel_BC"),
                    "gene_disease_effect": eff, "pathway": None, "source": "kg_neighbors",
                })
        if extra:
            kg.evidence.append(KGEvidence(operation="gene_disease_paths", rows=extra, status=ToolStatus.OK))

    @staticmethod
    def _coverage(kg: BackgroundKnowledge) -> float:
        """候选是否已被图谱证据覆盖的粗略比例（有返回行的证据即视为有覆盖）。"""
        if not kg.candidates:
            return 1.0
        hit = {row.get("gene") for ev in kg.evidence for row in ev.rows if isinstance(row, dict)}
        hit |= {row.get("gene") for row in kg.cypher_rows if isinstance(row, dict)}
        covered = sum(1 for g in kg.candidates if g in hit)
        return covered / len(kg.candidates)

    @staticmethod
    def _finalize(kg: BackgroundKnowledge) -> str:
        total_rows = sum(len(ev.rows) for ev in kg.evidence)
        if kg.status is ToolStatus.OK or total_rows:
            return f"检索到 {len(kg.evidence)} 类证据、{total_rows} 行背景知识（来源独立性={kg.source_independence}）。"
        return f"未取得背景知识：status={kg.status.value}, error={kg.error}（保持未知，不作阴性）。"
