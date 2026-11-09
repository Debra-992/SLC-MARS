# -*- coding: utf-8 -*-
"""Computation agent: calls PPR/TIMER/KM/DESeq2 on demand to supply candidate numeric knowledge."""
from __future__ import annotations

from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message

from knowledge_agents.actions.computation_action import Computation
from knowledge_agents.tools.computation_tools import ComputationTools


class ComputeRole(Role):
    """Pure candidate-data supplier: task -> routes the four computation tools -> ComputedKnowledge.

    Maps onto the Tool stage of the five-role SLC pipeline. Accepts explicit qid/gene/cancer/
    task_bucket injection; when absent, the Action infers them from the question text by regex.
    """

    name: str = "Cael"
    profile: str = "ComputationAgent"
    goal: str = "Compute statistics with the graph-proximity/expression/survival/differential-expression tools and supply them as candidate knowledge downstream"

    def __init__(self, tools: ComputationTools | None = None, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([Computation])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._tools = tools or ComputationTools()
        self._hint: dict = {}

    def set_task(self, *, qid: str | None = None, gene: str | None = None, cancer: str | None = None, task_bucket: str = "", genes: list[str] | None = None):
        """Inject the computation routing parameters explicitly instead of relying on question regex.
        genes carries multiple candidates for the pairwise/topk buckets."""
        self._hint = {"qid": qid, "gene": gene, "genes": genes, "cancer_type": cancer, "task_bucket": task_bucket}

    async def _act(self) -> Message:
        todo: Computation = self.rc.todo
        todo.tools = self._tools
        task = self.get_memories(k=1)[0].content
        logger.info(f"{self._setting}: computation for '{task[:60]}'")

        knowledge = await todo.run(task, **self._hint)
        msg = Message(
            content=knowledge.to_message_content(),
            role=self.profile,
            cause_by=Computation,
            sent_from=self.name,
        )
        self.rc.memory.add(msg)
        return msg
