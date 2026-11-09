# -*- coding: utf-8 -*-
"""Decision agent: consumes Kira (background knowledge) + Cael (computed knowledge) and emits
the final Answer.

Maps onto the Analyst/verdict stage of the SLC pipeline, scoped to pairwise / membership.
It never re-fetches data, it only synthesises the two knowledge channels already produced; if
the evidence is insufficient it abstains rather than inventing a ground truth.
"""
from __future__ import annotations

from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message

from knowledge_agents.actions.decision_action import AnswerDecision
from knowledge_agents.schema import BackgroundKnowledge, ComputedKnowledge


class DecisionRole(Role):
    name: str = "Vega"
    profile: str = "AnswerDecisionMaker"
    goal: str = "Synthesise the graph/literature background knowledge with the PPR/TIMER/KM/DESeq2 computed evidence into a pairwise/membership answer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([AnswerDecision])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._inputs: dict = {}

    def set_inputs(
        self,
        *,
        kg: BackgroundKnowledge,
        ck: ComputedKnowledge,
        bucket: str,
        candidates: list[str],
        question: str,
        task_type: str = "",
        context: dict | None = None,
        prior: dict | None = None,
        prior2: dict | None = None,
    ):
        self._inputs = {
            "kg": kg, "ck": ck, "bucket": bucket, "candidates": candidates,
            "question": question, "task_type": task_type, "context": context or {},
            "prior": prior, "prior2": prior2,
        }

    async def _act(self) -> Message:
        todo: AnswerDecision = self.rc.todo
        inp = self._inputs
        logger.info(f"{self._setting}: decide bucket={inp['bucket']} candidates={inp['candidates']}")

        answer = await todo.run(
            inp["question"],
            kg=inp["kg"], ck=inp["ck"], bucket=inp["bucket"],
            candidates=inp["candidates"], task_type=inp["task_type"], context=inp["context"],
            prior=inp.get("prior"), prior2=inp.get("prior2"),
        )
        msg = Message(
            content=answer.to_message_content(),
            role=self.profile,
            cause_by=AnswerDecision,
            sent_from=self.name,
        )
        self.rc.memory.add(msg)
        return msg
