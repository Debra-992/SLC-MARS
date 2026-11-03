# -*- coding: utf-8 -*-
"""知识供给双智能体的统一数据契约。

对齐 SLC multi_agent 真实 8 工具的返回结构与证据完整性语义：
  - 知识检索类 -> BackgroundKnowledge（Neo4j 图谱证据 + MongoDB 文献 + text2cypher）
  - 计算类     -> ComputedKnowledge（PPR / TIMER / KM / DESeq2 候选数值）

核心不变量：失败 / 不支持 / 无数据 == 未知（UNKNOWN），绝不伪造数值、绝不当作阴性。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    """镜像 tool_runtime.safe_error 的真实状态语义。"""

    OK = "ok"
    UNAVAILABLE = "unavailable"          # 依赖/浏览器/缓存/注解缺失
    DOWNLOAD_REQUIRED = "download_required"  # 缺原始计数且禁用下载
    UNSUPPORTED_CANCER = "unsupported_cancer"
    NO_SEED = "no_seed"                  # PPR 无匹配疾病种子
    REJECTED = "rejected"                # 非白名单 qid / 非法 Cypher
    ERROR = "error"                      # 执行异常，绝不替换结果

    @property
    def is_knowledge(self) -> bool:
        """仅 ok 才算真正取到了知识/数据。"""
        return self is ToolStatus.OK


class KnowledgeType(str, Enum):
    BACKGROUND_KG = "background_kg"     # 知识检索智能体产出
    CANDIDATE_DATA = "candidate_data"   # 计算智能体产出
    ANSWER = "answer"                    # 决策智能体产出（最终答案）


# 供下游 watch / 过滤使用的消息来源标记（与 Action 类名一一对应）
SUPPLY_KG_CAUSE = "KnowledgeRetrieval"
SUPPLY_COMPUTE_CAUSE = "Computation"
SUPPLY_ANSWER_CAUSE = "AnswerDecision"


class KnowledgePayload(BaseModel):
    """所有补充知识的基类：下游统一按 payload 解析。"""

    knowledge_type: str
    question: str = Field(description="触发本次知识供给的原始问题/任务")
    status: ToolStatus = ToolStatus.ERROR
    summary: str = Field(default="", description="对本条知识的自然语言摘要")
    source: Optional[str] = Field(default=None, description="取数来源（provenance）")
    error: Optional[str] = None

    def to_message_content(self) -> str:
        """序列化为 MetaGPT Message.content（JSON 字符串）。"""
        return self.model_dump_json(indent=2)


# ---------------------------------------------------------------- 检索类
class KGEvidence(BaseModel):
    """project_knowledge 三路 Neo4j 查询 + Mongo 文献的合并观测。"""

    operation: str = Field(description="gene_profile / gene_neighbors / gene_disease_paths / literature")
    rows: list[dict[str, Any]] = Field(default_factory=list)
    cypher: Optional[str] = None
    status: ToolStatus = ToolStatus.OK


class BackgroundKnowledge(KnowledgePayload):
    """知识检索智能体产出的背景知识。"""

    knowledge_type: str = Field(default=KnowledgeType.BACKGROUND_KG.value)
    candidates: list[str] = Field(default_factory=list, description="题目涉及的候选基因")
    requested_context: dict[str, Any] = Field(default_factory=dict, description="题目约束（cancer 等）")
    evidence: list[KGEvidence] = Field(default_factory=list, description="Neo4j/Mongo 逐项证据回执")
    # text2cypher 部分
    cypher: str = Field(default="", description="LLM 生成并实际执行的只读 Cypher")
    cypher_rows: list[dict[str, Any]] = Field(default_factory=list)
    schema_used: Optional[dict[str, Any]] = Field(default=None, description="kg_schema 上下文（非生物学证据）")
    # 证据完整性标注（照搬真实工具，不掩盖）
    public_web_used: bool = False
    source_independence: str = Field(default="NOT_ESTABLISHED")


# ---------------------------------------------------------------- 计算类
class PPRCandidate(BaseModel):
    gene: str
    ppr_score: Optional[float] = None
    degree_corrected_score: Optional[float] = None
    graph_degree: Optional[int] = None
    in_graph: bool = False


class GraphComputation(BaseModel):
    """project_graph_computation：疾病种子加权 PPR。"""

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
    """timer_pancancer_expression。"""

    status: ToolStatus = ToolStatus.ERROR
    gene: str
    row_count: Optional[int] = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    boxplot_path: Optional[str] = None
    evidence_scope: str = Field(default="unknown", description="table / plot_only / unknown")


class SurvivalObservation(BaseModel):
    """km_survival。"""

    status: ToolStatus = ToolStatus.ERROR
    gene: str
    cancer_type: str
    tcga_code: Optional[str] = None
    stats_text: Optional[str] = None
    plot_path: Optional[str] = None
    evidence_scope: str = Field(default="unknown", description="plot_and_stats / plot_only / unknown")


class DifferentialExpression(BaseModel):
    """slc_differential_expression：TCGA + DESeq2。"""

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
    """计算智能体产出的候选数值知识（按需路由四类计算工具）。

    pairwise/topk 有多个候选基因，故除单值 `expression`/`survival`（向后兼容，
    存主基因）外，额外提供 `expression_by_gene`/`survival_by_gene` 逐基因观测。
    """

    knowledge_type: str = Field(default=KnowledgeType.CANDIDATE_DATA.value)
    graph_computation: Optional[GraphComputation] = None
    expression: Optional[ExpressionObservation] = None
    survival: Optional[SurvivalObservation] = None
    differential_expression: Optional[DifferentialExpression] = None
    expression_by_gene: dict[str, ExpressionObservation] = Field(default_factory=dict)
    survival_by_gene: dict[str, SurvivalObservation] = Field(default_factory=dict)
    # 旁系缓冲（paralog buffering）：以该癌种已测 SLC 面板中『同亚家族旁系』的共表达比例
    # 估计功能冗余。redundancy∈[0,1]（越大越冗余->Knockout 越可能被缓冲->越不像必需基因）；
    # 仅在能拿到该癌种表达面板（DESeq2 top_genes）时有值，拿不到则不写入（未知，不臆造）。
    paralog_redundancy: dict[str, float] = Field(default_factory=dict)
    paralog_partners: dict[str, list[str]] = Field(default_factory=dict)
    # 证据完整性：所有未取到的工具保持未知，不伪造
    skipped_tools: list[str] = Field(default_factory=list, description="未路由的工具及原因")


# ---------------------------------------------------------------- 决策类
class Answer(KnowledgePayload):
    """决策智能体产出的最终答案（限 pairwise / membership）。

    证据完整性不变量：证据不足时 abstained=True 且 answer/winner=None，
    绝不把"未知"当成阴性(False)或随意挑一个 winner。
    """

    knowledge_type: str = Field(default=KnowledgeType.ANSWER.value)
    bucket: str = Field(description="pairwise | membership")
    task_type: str = ""
    answer: Optional[bool] = Field(default=None, description="membership 真值")
    winner: Optional[str] = Field(default=None, description="pairwise 胜者基因")
    candidates: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    abstained: bool = Field(default=False, description="证据不足而弃答")
    gene_scores: dict[str, float] = Field(default_factory=dict)
    basis: list[str] = Field(default_factory=list, description="本次判定用到的证据来源")
    rationale: str = ""
    analogy_cancers: list[str] = Field(default_factory=list, description="LLM 跨癌种类推时参考的『临近癌种』")
    decision_mode: str = Field(default="heuristic", description="heuristic | llm")
    used_llm: bool = False
    evidence_integrity: str = "unknown!=negative"
    conformal_threshold: Optional[float] = Field(default=None, description="保形弃答采用的置信阈值(=1-非一致性分数)；None=未施加")


def parse_knowledge(content: str) -> Optional[KnowledgePayload]:
    """从 Message.content 反解析知识对象，供下游统一入口使用。"""
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
