# -*- coding: utf-8 -*-
"""Core action of the knowledge-retrieval agent: graph/literature primary knowledge plus
text2cypher supplementary background knowledge.

Path (following the real SLC strategy):
  1. project_knowledge(qid) pulls the three Neo4j evidence channels + MongoDB literature (the
     main background knowledge)
  2. when candidate coverage is short: kg_schema -> LLM writes a read-only Cypher -> kg_cypher
     executes it (text2cypher)
  3. if still missing, kg_gene_neighbors fills in the direct neighbours
The read-only guard and "a failure is unknown" hold throughout; nothing is ever fabricated.
"""
from __future__ import annotations

import json
import re

from metagpt.actions import Action

from knowledge_agents.schema import BackgroundKnowledge, KGEvidence, ToolStatus
from knowledge_agents.tools.retrieval_tools import RetrievalTools

CODE_BLOCK_RE = re.compile(r"```(?:cypher)?\s*(.*?)```", re.DOTALL)


class KnowledgeRetrieval(Action):
    """Knowledge-retrieval action: produces BackgroundKnowledge."""

    name: str = "KnowledgeRetrieval"

    TEXT2CYPHER_PROMPT: str = """
This graph is a [temporal causal graph]. SLC genes and diseases are [never directly connected],
and [no relationship type named relevant exists]; their association is reified into event nodes
RelaEvent, forming an A-B-C causal triple:
  (g:SLCGene)-[:IS_SOURCE]->(e:RelaEvent)<-[:IS_MEDIATOR]-(p:Pathway)
  (g:SLCGene)-[:IS_SOURCE]->(e:RelaEvent)<-[:IS_TARGET]--(d:Disease)
RelaEvent properties: rel_AB = the direction SLC->Pathway (PROMOTION/SUPPRESSION),
  rel_BC = the direction Pathway->Disease (PROMOTION/SUPPRESSION), paper_dois = literature provenance (array).
gene->disease net direction = PROMOTION when rel_AB and rel_BC agree, SUPPRESSION when they oppose.
Real labels: SLCGene(name,NCBIGeneID,ProteinFunction,ProteinAliases,Transport_type,Tissue_and_cellular_expression),
  Pathway(name), Disease(name,diseaseSort), RelaEvent(rel_AB,rel_BC,paper_dois), SLCFamily(name), Protein(name), Product(name,target).

Live schema (supplementary validation only, never biological evidence):
{schema}

Candidate genes: {genes}
Question constraints: {context}

Generate [one read-only Cypher] that retrieves the diseases/pathways linked to the candidate genes
through RelaEvent, and [always return rel_AB, rel_BC plus the signed composite
CASE WHEN e.rel_AB=e.rel_BC THEN 'PROMOTION' ELSE 'SUPPRESSION' END AS gene_disease_effect].
Hard requirements:
- the query may only start with MATCH/OPTIONAL MATCH/WITH/RETURN/UNWIND/EXPLAIN
- no write clauses such as CREATE/MERGE/DELETE/SET/DROP, no CALL/APOC/GDS procedures, no multi-statement text (semicolons)
- labels/properties must come strictly from the real schema above; never invent a relationship type (such as relevant) or a direct (gene)-[]-(disease) edge
Return only the ```cypher code block```, nothing else.
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

        # 1) main background knowledge: the project graph plus literature
        kg = await self.tools.project_knowledge(qid)
        if kg.status is not ToolStatus.OK:
            kg.summary = f"primary knowledge unavailable ({kg.status.value}); trying text2cypher as a supplement only."

        # 2) text2cypher supplement: triggered only when candidates exist but graph row coverage is short
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

        # 3) temporal-causal reinforcement: for candidate genes still lacking a disease path [with a
        # signed effect], let kg_gene_neighbors see through RelaEvent and supply the direction
        await self._signed_neighbor_backfill(kg)

        kg.summary = self._finalize(kg)
        return kg

    async def _signed_neighbor_backfill(self, kg: BackgroundKnowledge) -> None:
        """For candidate genes with no signed gene_disease_paths row, call the kg_neighbors that already
        [sees through RelaEvent] and normalise the RELATES_DISEASE rel_AB/rel_BC/effect into
        gene_disease_paths-compatible rows, so pure proximity questions also get the temporal causal
        direction (same underlying events, filled in only where coverage is missing, so nothing is
        double-counted)."""
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
        """Rough share of candidates already covered by graph evidence (any evidence that returned rows
        counts as coverage)."""
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
            return f"retrieved {len(kg.evidence)} evidence channel(s), {total_rows} rows of background knowledge (source independence={kg.source_independence})."
        return f"no background knowledge obtained: status={kg.status.value}, error={kg.error} (kept unknown, not read as negative)."
