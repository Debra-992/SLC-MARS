# -*- coding: utf-8 -*-
"""知识检索智能体：从 SLC 知识图谱/文献经 text2cypher 供给背景知识。"""
from __future__ import annotations

from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message

from knowledge_agents.actions.knowledge_retrieval_action import KnowledgeRetrieval
from knowledge_agents.tools.retrieval_tools import RetrievalTools


class KnowledgeQueryRole(Role):
    """纯背景知识供给侧：问题/qid -> project_knowledge + text2cypher -> BackgroundKnowledge。

    对应 SLC 五角色管线的 Knowledge 段。BY_ORDER 单动作直出，符合"只供知识"定位。
    """

    name: str = "Kira"
    profile: str = "KnowledgeGraphQuerier"
    goal: str = "从项目 Neo4j 图谱与 MongoDB 文献、并经 text2cypher 检索背景知识，供下游分析引用"

    def __init__(self, tools: RetrievalTools | None = None, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([KnowledgeRetrieval])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._tools = tools or RetrievalTools()

    async def _act(self) -> Message:
        todo: KnowledgeRetrieval = self.rc.todo
        todo.tools = self._tools  # 显式装配，便于替换/离线兜底后端
        question = self.get_memories(k=1)[0].content
        logger.info(f"{self._setting}: knowledge retrieval for '{question[:60]}'")

        knowledge = await todo.run(question)
        msg = Message(
            content=knowledge.to_message_content(),
            role=self.profile,
            cause_by=KnowledgeRetrieval,
            sent_from=self.name,
        )
        self.rc.memory.add(msg)
        return msg
