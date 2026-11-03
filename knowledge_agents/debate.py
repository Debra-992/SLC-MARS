# -*- coding: utf-8 -*-
"""v11 专家辩论层（借 MDAgents 的多专家+多数裁决思想，做成管线内的轻量版）。

与 MDAgents 的差异（刻意为之）：
  * 只仲裁歧义题（证据通道与 prior 冲突 / 或需要补覆盖时），不是每题都开辩；
  * 3 位专家【并行】各出 1 票（MDAgents 是 5 专家×5 轮的串行辩论，成本数量级）；
  * 裁决用确定性置信加权多数（免 moderator 第二次 LLM 调用，可复现、可审计）。
关键公平性：专家看到管线的证据摘要 + prior 观点，被要求"只依据提供的证据投票，
证据不足投 abstain"——先验只能作为被审视的一方观点，不再是无约束的裸记忆答题。

env：SLC_DEBATE=1 启用；SLC_DEBATE_TIMEOUT 单专家超时（默认 90s）。
失败诚实：任何专家故障=缺票；有效票不足或分歧过平 -> 返回 None（不强行仲裁）。
"""
from __future__ import annotations

import asyncio
import os

from knowledge_agents.baselines.common import extract_json, llm_settings

_ON = {"1", "true", "on", "yes"}

EXPERTS = [
    ("基因功能与旁系遗传专家",
     "你专精 SLC 基因家族功能、旁系同源性冗余与 CRISPR loss-of-function 必需性筛选逻辑。"),
    ("肿瘤临床与转化专家",
     "你专精肿瘤分型、代谢重编程的癌种特异性依赖与耐药机制的临床证据解读。"),
    ("统计基因组学专家",
     "你专精网络邻近度(PPR)、表达(TIMER/DESeq2)、生存(KM)与因果方向证据的量级解读，"
     "只认证据强度、不认名气。"),
]

# 置信加权多数的裁决裕度：top1 与 top2 的加权差需>=此值才算"辩论有结论"
DEBATE_MARGIN = float(os.environ.get("SLC_DEBATE_MARGIN", "0.15"))


def debate_enabled() -> bool:
    return os.environ.get("SLC_DEBATE", "0").lower() in _ON


def _vote_prompt(persona: str, question: str, bucket: str, candidates: list[str],
                 digest: str, views: str) -> str:
    task = ("请从候选中投票更可能的 winner" if bucket == "pairwise"
            else "请投票该命题 answer=true/false")
    fmt = ('{"winner": "<候选之一>", "confidence": 0~1, "abstained": false, "rationale": "一句话"}'
           if bucket == "pairwise" else
           '{"answer": true/false, "confidence": 0~1, "abstained": false, "rationale": "一句话"}')
    return f"""{persona}
你在一个多学科专家组(MDT)中，就下列 SLC 基因 CRISPR 必需性问题独立投票：{task}。

铁律：只依据【管线证据】与【待审观点】投票。禁止凭文献记忆补充事实——证据不足就投 abstain。
问题(桶={bucket}) 候选={candidates}
管线证据（图谱/统计通道打分与判定理由）:
{digest}

待审观点（prior=纯LLM记忆先验的立场，仅供批判参考，不是证据）:
{views}

只输出 JSON：{fmt}"""


async def _one_expert(cfg: dict, prompt: str) -> dict | None:
    from openai import AsyncOpenAI
    timeout = float(os.environ.get("SLC_DEBATE_TIMEOUT", "90"))
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=timeout, max_retries=0)
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(model=cfg["model"],
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0.3),
            timeout=timeout)
        return extract_json(rsp.choices[0].message.content or "")
    except Exception:  # noqa BLE001 —— 专家故障=缺票
        return None
    finally:
        try:
            await client.close()
        except Exception:  # noqa BLE001
            pass


async def run_debate(question: str, bucket: str, candidates: list[str],
                     digest: str, views: str) -> dict | None:
    """3 专家并行投票 -> 确定性置信加权多数。返回 {winner|answer, support, margin,
    tally, rationale} 或 None（没开出有效结论）。"""
    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        return None
    cands = [c.upper() for c in (candidates or [])]
    prompts = [_vote_prompt(p, question, bucket, cands, digest, views) for _, p in EXPERTS]
    votes = await asyncio.gather(*(_one_expert(cfg, pr) for pr in prompts))

    # 置信加权计票（abstain/越界票不计）
    tally: dict[str, float] = {}
    n_valid = 0
    for v in votes:
        if not v or v.get("abstained"):
            continue
        try:
            conf = max(0.0, min(1.0, float(v.get("confidence", 0) or 0)))
        except (TypeError, ValueError):
            conf = 0.0
        if bucket == "pairwise":
            w = str(v.get("winner") or "").upper()
            if w not in cands:
                continue
            tally[w] = tally.get(w, 0.0) + conf + 0.2   # +0.2 基础票：置信塌缩时仍数人头
        else:
            a = v.get("answer")
            if a is None:
                continue
            key = "true" if bool(a) else "false"
            tally[key] = tally.get(key, 0.0) + conf + 0.2
        n_valid += 1
    if n_valid < 2 or not tally:
        return None
    ranked = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    top_key, top_w = ranked[0]
    margin = top_w - (ranked[1][1] if len(ranked) > 1 else 0.0)
    if margin < DEBATE_MARGIN:
        return None                                   # 分歧过平 -> 不强行仲裁
    support = (f"{n_valid}/{len(EXPERTS)} 位专家有效投票，"
               f"top1 加权={top_w:.2f}(差={margin:.2f})")
    out = {"support": support, "margin": round(margin, 3),
           "tally": {k: round(w, 2) for k, w in ranked},
           "rationale": f"[专家辩论·v11] {bucket} 加权多数="}
    if bucket == "pairwise":
        out["winner"] = top_key
    else:
        out["answer"] = (top_key == "true")
    out["rationale"] += top_key
    return out
