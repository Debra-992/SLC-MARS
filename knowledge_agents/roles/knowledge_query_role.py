# -*- coding: utf-8 -*-
"""Knowledge-retrieval agent: supplies background knowledge from the SLC knowledge graph and
the literature corpus through text2cypher."""
from __future__ import annotations

from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message

from knowledge_agents.actions.knowledge_retrieval_action import KnowledgeRetrieval
from knowledge_agents.tools.retrieval_tools import RetrievalTools


class KnowledgeQueryRole(Role):
    """Pure background-knowledge supplier: question/qid -> project_knowledge + text2cypher ->
    BackgroundKnowledge.

    Maps onto the Knowledge stage of the five-role SLC pipeline. BY_ORDER runs a single action
    straight through, which matches the role's remit of supplying knowledge only.
    """

    name: str = "Kira"
    profile: str = "KnowledgeGraphQuerier"
    goal: str = "Retrieve background knowledge from the project Neo4j graph and MongoDB literature, including via text2cypher, for the downstream agents to cite"

    def __init__(self, tools: RetrievalTools | None = None, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([KnowledgeRetrieval])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._tools = tools or RetrievalTools()

    async def _act(self) -> Message:
        todo: KnowledgeRetrieval = self.rc.todo
        todo.tools = self._tools  # wired explicitly so the backend can be swapped or run offline
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
