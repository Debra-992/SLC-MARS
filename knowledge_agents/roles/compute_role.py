# -*- coding: utf-8 -*-
"""计算智能体：按需调用 PPR/TIMER/KM/DESeq2 供给候选数值知识。"""
from __future__ import annotations

from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message

from knowledge_agents.actions.computation_action import Computation
from knowledge_agents.tools.computation_tools import ComputationTools


class ComputeRole(Role):
    """纯候选数据供给侧：任务 -> 路由 4 类计算工具 -> ComputedKnowledge。

    对应 SLC 五角色管线的 Tool 段。支持显式注入 qid/gene/cancer/task_bucket，
    未注入时由 Action 从题面正则推断。
    """

    name: str = "Cael"
    profile: str = "ComputationAgent"
    goal: str = "调用图谱邻近度/表达/生存/差异表达工具计算统计数据，作为候选知识供给下游"

    def __init__(self, tools: ComputationTools | None = None, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([Computation])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._tools = tools or ComputationTools()
        self._hint: dict = {}

    def set_task(self, *, qid: str | None = None, gene: str | None = None, cancer: str | None = None, task_bucket: str = "", genes: list[str] | None = None):
        """显式注入计算路由参数，避免依赖题面正则。genes 支持 pairwise/topk 多候选。"""
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
