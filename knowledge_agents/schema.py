# -*- coding: utf-8 -*-
"""Unified data contract for the two knowledge-supply agents.

Mirrors the return structures and the evidence-integrity semantics of the eight real SLC
multi_agent tools:
  - retrieval class -> BackgroundKnowledge (Neo4j graph evidence + MongoDB literature + text2cypher)
  - computation class -> ComputedKnowledge (PPR / TIMER / KM / DESeq2 candidate numbers)

Central invariant: failure / unsupported / no data == UNKNOWN. Numbers are never fabricated and
unknown is never treated as a negative.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    """Mirrors the real status semantics of tool_runtime.safe_error."""

    OK = "ok"
    UNAVAILABLE = "unavailable"          # a dependency/browser/cache/annotation is missing
    DOWNLOAD_REQUIRED = "download_required"  # raw counts absent and downloading disabled
    UNSUPPORTED_CANCER = "unsupported_cancer"
    NO_SEED = "no_seed"                  # PPR found no matching disease seed
    REJECTED = "rejected"                # qid outside the whitelist / invalid Cypher
    ERROR = "error"                      # execution failure, never substituted with another result

    @property
    def is_knowledge(self) -> bool:
        """Only ok means knowledge/data was actually retrieved."""
        return self is ToolStatus.OK


class KnowledgeType(str, Enum):
    BACKGROUND_KG = "background_kg"     # produced by the knowledge-retrieval agent
    CANDIDATE_DATA = "candidate_data"   # produced by the computation agent
    ANSWER = "answer"                    # produced by the decision agent (the final answer)


# Message-source markers used by downstream watching/filtering (one per Action class)
SUPPLY_KG_CAUSE = "KnowledgeRetrieval"
SUPPLY_COMPUTE_CAUSE = "Computation"
SUPPLY_ANSWER_CAUSE = "AnswerDecision"


class KnowledgePayload(BaseModel):
    """Base class of every supplementary knowledge item; downstream parses it as a payload."""

    knowledge_type: str
    question: str = Field(description="the original question/task that triggered this knowledge supply")
    status: ToolStatus = ToolStatus.ERROR
    summary: str = Field(default="", description="natural-language digest of this knowledge item")
    source: Optional[str] = Field(default=None, description="where the data came from (provenance)")
    error: Optional[str] = None

    def to_message_content(self) -> str:
        """Serialise into a MetaGPT Message.content (a JSON string)."""
        return self.model_dump_json(indent=2)


# ---------------------------------------------------------------- retrieval class
class KGEvidence(BaseModel):
    """Merged observations from the three project_knowledge Neo4j queries plus the Mongo literature."""

    operation: str = Field(description="gene_profile / gene_neighbors / gene_disease_paths / literature")
    rows: list[dict[str, Any]] = Field(default_factory=list)
    cypher: Optional[str] = None
    status: ToolStatus = ToolStatus.OK


class BackgroundKnowledge(KnowledgePayload):
    """Background knowledge produced by the knowledge-retrieval agent."""

    knowledge_type: str = Field(default=KnowledgeType.BACKGROUND_KG.value)
    candidates: list[str] = Field(default_factory=list, description="candidate genes the question is about")
    requested_context: dict[str, Any] = Field(default_factory=dict, description="question constraints (cancer type and friends)")
    evidence: list[KGEvidence] = Field(default_factory=list, description="per-item Neo4j/Mongo evidence receipts")
    # text2cypher part
    cypher: str = Field(default="", description="the read-only Cypher the LLM wrote and that actually ran")
    cypher_rows: list[dict[str, Any]] = Field(default_factory=list)
    schema_used: Optional[dict[str, Any]] = Field(default=None, description="kg_schema context (never biological evidence)")
    # evidence-integrity annotations, copied straight from the real tools rather than hidden
    public_web_used: bool = False
    source_independence: str = Field(default="NOT_ESTABLISHED")


# ---------------------------------------------------------------- computation class
class PPRCandidate(BaseModel):
    gene: str
    ppr_score: Optional[float] = None
    degree_corrected_score: Optional[float] = None
    graph_degree: Optional[int] = None
    in_graph: bool = False


class GraphComputation(BaseModel):
    """project_graph_computation: disease-seeded weighted PPR."""

    status: ToolStatus = ToolStatus.ERROR
    algorithm: str = "weighted_disease_seeded_degree_corrected_ppr"
    alpha: float = 0.6
    beta: float = 0.5
    seed_count: int = 0
    seed_names: list[str] = Field(default_factory=list)
    node_count: Optional[int] = None
    edge_count: Optional[int] = None
    graph_sha256: Optional[str] = None
    candidates: list[PPRCandidate] = Field(default_factory=list)
    interpretation: str = (
        "Network proximity only; not signed inhibition/promotion, not CRISPR measurement, "
        "missing/zero score is not negative outcome"
    )


class ExpressionObservation(BaseModel):
    """timer_pancancer_expression."""

    status: ToolStatus = ToolStatus.ERROR
    gene: str
    row_count: Optional[int] = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    boxplot_path: Optional[str] = None
    evidence_scope: str = Field(default="unknown", description="table / plot_only / unknown")


class SurvivalObservation(BaseModel):
    """km_survival."""

    status: ToolStatus = ToolStatus.ERROR
    gene: str
    cancer_type: str
    tcga_code: Optional[str] = None
    stats_text: Optional[str] = None
    plot_path: Optional[str] = None
    evidence_scope: str = Field(default="unknown", description="plot_and_stats / plot_only / unknown")


class DifferentialExpression(BaseModel):
    """slc_differential_expression: TCGA + DESeq2."""

    status: ToolStatus = ToolStatus.ERROR
    cancer_type: str
    tcga_code: Optional[str] = None
    gene_family: str = "SLC"
    analysis_method: Optional[str] = None
    contrast: str = "tumor/normal"
    significant_count: Optional[int] = None
    top_genes: list[dict[str, Any]] = Field(default_factory=list)
    sample_counts: dict[str, int] = Field(default_factory=dict)
    family_tested_count: Optional[int] = None


class ComputedKnowledge(KnowledgePayload):
    """Candidate numeric knowledge produced by the computation agent, with the four computation tools
    routed on demand.

    pairwise/topk questions carry several candidate genes, so besides the single-valued
    `expression`/`survival` fields (kept for backward compatibility, holding the primary gene)
    there are per-gene observations in `expression_by_gene`/`survival_by_gene`.
    """

    knowledge_type: str = Field(default=KnowledgeType.CANDIDATE_DATA.value)
    graph_computation: Optional[GraphComputation] = None
    expression: Optional[ExpressionObservation] = None
    survival: Optional[SurvivalObservation] = None
    differential_expression: Optional[DifferentialExpression] = None
    expression_by_gene: dict[str, ExpressionObservation] = Field(default_factory=dict)
    survival_by_gene: dict[str, SurvivalObservation] = Field(default_factory=dict)
    # Paralog buffering: functional redundancy estimated from the co-expression share of the
    # same-subfamily paralogues in the SLC panel assayed for this cancer type.
    # redundancy in [0,1] (larger = more redundant -> the knockout is more likely buffered -> less
    # likely essential); populated only when that cancer type's expression panel (DESeq2 top_genes)
    # is available, and left unwritten otherwise (unknown, never guessed).
    paralog_redundancy: dict[str, float] = Field(default_factory=dict)
    paralog_partners: dict[str, list[str]] = Field(default_factory=dict)
    # Evidence integrity: every tool that did not return stays unknown, nothing is fabricated
    skipped_tools: list[str] = Field(default_factory=list, description="tools not routed and the reason")


# ---------------------------------------------------------------- decision class
class Answer(KnowledgePayload):
    """Final answer produced by the decision agent, scoped to pairwise / membership.

    Evidence-integrity invariant: when the evidence is insufficient, abstained=True and
    answer/winner are None -- "unknown" is never turned into a negative (False) and a winner is
    never picked arbitrarily.
    """

    knowledge_type: str = Field(default=KnowledgeType.ANSWER.value)
    bucket: str = Field(description="pairwise | membership")
    task_type: str = ""
    answer: Optional[bool] = Field(default=None, description="membership truth value")
    winner: Optional[str] = Field(default=None, description="winning gene for pairwise")
    candidates: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    abstained: bool = Field(default=False, description="abstained because the evidence was insufficient")
    gene_scores: dict[str, float] = Field(default_factory=dict)
    basis: list[str] = Field(default_factory=list, description="evidence sources this verdict used")
    rationale: str = ""
    analogy_cancers: list[str] = Field(default_factory=list, description="the [neighbouring cancer types] the LLM referred to when transferring across cancer types")
    decision_mode: str = Field(default="heuristic", description="heuristic | llm")
    used_llm: bool = False
    evidence_integrity: str = "unknown!=negative"
    conformal_threshold: Optional[float] = Field(default=None, description="confidence threshold used for conformal abstention (= 1 - non-conformity score); None means not applied")


def parse_knowledge(content: str) -> Optional[KnowledgePayload]:
    """Re-parse a knowledge object out of Message.content; the single entry point for downstream."""
    import json

    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    ktype = data.get("knowledge_type")
    if ktype == KnowledgeType.BACKGROUND_KG.value:
        return BackgroundKnowledge.model_validate(data)
    if ktype == KnowledgeType.CANDIDATE_DATA.value:
        return ComputedKnowledge.model_validate(data)
    if ktype == KnowledgeType.ANSWER.value:
        return Answer.model_validate(data)
    return None
