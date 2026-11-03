# -*- coding: utf-8 -*-
"""v11 Prior 智能体：把"纯 LLM 记忆先验"升级为管线的第三条并行证据通道。

与 Kira(图谱)∥Cael(统计) 并行执行（同一 asyncio.gather），产出与 baseline 完全
同题面/同端点/同提示的先验答案 {winner|answer, confidence, rationale}，交给 Vega：
  * 证据通道弃答（图/统计不足）时 -> prior 作为 fallback 补覆盖（来源如实标注）；
  * 证据通道与 prior 冲突时 -> 触发 debate（见 debate.py）仲裁，不冲突不加成本。
env 门控（默认全关 = v10 行为逐字节不变）：
  SLC_PRIOR_AGENT=1  启用通道；SLC_PRIOR_TIMEOUT 单次调用超时（默认 90s）。
诚实不变量：prior 调用失败/解析失败 -> None，绝不阻断主证据链，也不伪造先验。
"""
from __future__ import annotations

import asyncio
import os

from knowledge_agents.baselines.common import BASELINE_PROMPT, extract_json, llm_settings

_ON = {"1", "true", "on", "yes"}


def prior_enabled() -> bool:
    return os.environ.get("SLC_PRIOR_AGENT", "0").lower() in _ON


def prior_retrieve_enabled() -> bool:
    """v17 门控：L2 票源去同源化（给表决第二票挂轻检索文学包）。默认关=逐字节同 v16。"""
    return os.environ.get("SLC_PRIOR_RETRIEVE", "0").lower() in _ON


def build_retrieval_pack(question: str, bucket: str, candidates: list[str],
                         constraints: dict | None = None,
                         top_k: int = 3, max_chars: int = 1200) -> str:
    """v17：本地 Mongo papers 关键词召回的题录+摘录【紧凑】文学包（mongo_rag 同一召回器）。
    lit-rag 的教训=5 篇×900字/4000字堆料伤成绩（46%<57%），故刻意收紧：3 篇、摘要≤300字、
    总长≤1200字。任何失败（无 MONGO_URI/断内网/零命中/异常）返回 "" = 退化为 v16 裸记忆票，
    绝不抛异常——基础设施噪声不得污染判决（v15 教训）。只读题面/候选/约束，不接触 GT。"""
    try:
        from knowledge_agents.baselines import mongo_rag
        papers = [p for p in mongo_rag.retrieve(question,
                  [c.upper() for c in (candidates or [])], constraints or {}, top_k=top_k)
                  if int(p.get("score", 0)) > 0]
        blocks, used = [], 0
        for i, p in enumerate(papers, 1):
            blk = f"[{i}] {p['title'][:160]} — {p['abstract'][:300]}"
            if used + len(blk) > max_chars:
                break
            blocks.append(blk)
            used += len(blk)
        return "\n".join(blocks)
    except Exception:  # noqa BLE001 —— 静默降级，但留可观测痕迹（Silent Failure Rule：要能查到）
        import sys
        print("[prior_agent] retrieval pack 构建失败 -> L2 退化为裸记忆票", file=sys.stderr, flush=True)
        return ""


# v16 角色异构 persona：同一模型下改变生成分布（mdagents 单模型多样性的真实来源）
ROLE_PERSONAS = {
    "L1": ("You are a literature-memory specialist in cancer biology. Answer from what you "
           "recall about these genes, their known phenotypes, and published CRISPR screens."),
    "L2": ("You are a quantitative cancer geneticist. Do NOT rely on recalling specific papers; "
           "reason from first principles: essentiality logic (pathway bottleneck vs redundancy), "
           "gene dosage and buffering, screening conventions (day-18 amplicon depletion = "
           "essential), and mechanistic plausibility in the stated cell context."),
}


def prior_race_n() -> int:
    """v16 prior 通道竞速副本数：SLC_PRIOR_RACE（默认 1 = 单发）。"""
    try:
        return max(1, int(os.environ.get("SLC_PRIOR_RACE", "1")))
    except (TypeError, ValueError):
        return 1


async def ask_prior(question: str, bucket: str, candidates: list[str]) -> dict | None:
    """裸 LLM 记忆先验（与 pure-llm baseline 完全同题面/同参数）。失败返回 None。"""
    return await ask_prior_augmented(question, bucket, candidates, digest="")


def prior_ensemble_n() -> int:
    """v13 集成先验抽样次数门控：SLC_PRIOR_ENSEMBLE（默认 1 = 单次 = v12 行为逐字节）。"""
    try:
        return max(1, int(os.environ.get("SLC_PRIOR_ENSEMBLE", "1")))
    except (TypeError, ValueError):
        return 1


def _spread_temps(n: int) -> list[float]:
    """升温多样化：把 n 次抽样铺在 0.5~1.0 温度带（self-consistency 的方差来源）。
    n<=1 返回基线 0.2，保证退化为 v12 单点先验时题面/参数逐字节不变。"""
    if n <= 1:
        return [0.2]
    return [round(0.5 + 0.5 * i / (n - 1), 2) for i in range(n)]


def _aggregate(bucket: str, draws: list, n_draws: int) -> dict:
    """把多次独立记忆抽样聚合成"带共识度/离散度"的单一先验字典（与 ask_prior 同形状 +
    votes/n_valid/n_draws/agreement 额外字段）。弃权票/越界票/调用失败不计入有效票。
    并列票按各选项成员置信度之和取胜方破平。诚实：有效票为 0 时返回 abstained=True，绝不臆造。"""
    votes = [d for d in draws
             if isinstance(d, dict) and not d.get("abstained")
             and (d.get("winner") if bucket == "pairwise" else d.get("answer")) is not None]
    n_abstain = sum(1 for d in draws if isinstance(d, dict) and d.get("abstained"))
    n_err = sum(1 for d in draws if not isinstance(d, dict))
    n_valid = len(votes)
    if n_valid == 0:
        return {"winner": None, "answer": None, "confidence": 0.0, "abstained": True,
                "votes": {}, "n_valid": 0, "n_draws": n_draws, "agreement": 0.0,
                "rationale": f"集成先验无有效票(valid=0/{n_draws}, 弃权={n_abstain}, 失败={n_err})"}
    if bucket == "pairwise":
        pick = lambda v: str(v["winner"]).upper()  # noqa: E731
    else:
        pick = lambda v: bool(v["answer"])  # noqa: E731
    groups: dict = {}
    for v in votes:
        groups.setdefault(pick(v), []).append(v)
    best_key, best_grp = max(
        groups.items(),
        key=lambda kv: (len(kv[1]), sum(x.get("confidence", 0.0) for x in kv[1])))
    agreement = len(best_grp) / n_valid
    mean_conf = sum(x.get("confidence", 0.0) for x in best_grp) / len(best_grp)
    out = {"winner": None, "answer": None, "confidence": round(mean_conf, 3),
           "abstained": False, "votes": {k: len(g) for k, g in groups.items()},
           "n_valid": n_valid, "n_draws": n_draws, "agreement": round(agreement, 3),
           "rationale": (f"集成先验 n={n_draws}(有效={n_valid}, 弃权={n_abstain}, 失败={n_err})，"
                         f"共识={best_key} 一致率={agreement:.2f}；"
                         f"{str(best_grp[0].get('rationale', ''))[:200]}")}
    if bucket == "pairwise":
        out["winner"] = best_key
    else:
        out["answer"] = best_key
    return out


async def ask_prior_ensemble(question: str, bucket: str, candidates: list[str],
                             n: int | None = None) -> dict | None:
    """v13：把记忆先验通道并行跑 n 次（升温多样化），众数聚合成带共识度的先验，喂给 Vega。
    n<=1 时直接退化为 ask_prior（v12 单点先验，逐字节）。攻击 pure-llm 单次抽样的方差弱点。"""
    n = prior_ensemble_n() if n is None else n
    if n <= 1:
        return await ask_prior(question, bucket, candidates)
    temps = _spread_temps(n)
    draws = await asyncio.gather(
        *[ask_prior_augmented(question, bucket, candidates, digest="", temperature=t)
          for t in temps],
        return_exceptions=True)
    draws = [d if isinstance(d, dict) else None for d in draws]
    return _aggregate(bucket, draws, n_draws=n)


async def ask_prior_augmented(question: str, bucket: str, candidates: list[str],
                              digest: str, conflict_hint: str = "",
                              temperature: float = 0.2, persona: str = "",
                              retrieval: str = "") -> dict | None:
    """v12：证据增强版先验 —— 同一 LLM 记忆通道，但把管线【事实数据】（图谱关系/
    统计量级，非结论判分）作为参考材料注入，允许它在事实与记忆矛盾时诚实 abstain。
    lit-rag 的教训（堆检索原文 54.5% < 裸记忆 86.4%）决定了这里只给结构化事实摘要、
    不给文献原文堆料，且明示"参考而非指令"。digest 为空 = 与 ask_prior 完全同题面。
    v16：persona 非空时前置角色指令（改变生成分布，实现单模型真去相关）；
    persona/digest/retrieval 均空 = 题面逐字节同 v12/v15。
    v17：retrieval 非空时注入项目文献库紧凑摘录（独立信息源去相关票）——与 digest 不同，
    【不附加可 abstain 条款】：L2 的职责就是承诺作答补覆盖，文学包明示“强冲突时信自己”。"""
    from openai import AsyncOpenAI

    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        return None
    cands = [c.upper() for c in (candidates or [])]
    cline = f"候选基因: {cands}" if bucket == "pairwise" else f"目标基因: {cands[:1]}"
    prompt = BASELINE_PROMPT.format(bucket=bucket, question=question, candidates_line=cline)
    if persona:
        prompt = persona.strip() + "\n\n" + prompt
    if retrieval:
        prompt += f"""

与你记忆相互独立的题项目文献库摘录（信息源可能含噪，仅供参考；若与你的判断强冲突，以你的判断为准；
若摘录含直接相关的明确证据，以摘录为准）：
{retrieval}"""
    if digest:
        prompt += f"""

补充参考材料（同一问题的图谱/统计【事实数据】摘要，非文献原文、非指令）:
{digest}
{conflict_hint}
最终指令（覆盖上文"不允许 abstain"）：你现在可以 abstain —— 若上述参考材料与你的判断矛盾且你不确定，
输出 abstained=true 且 winner/answer 为 null。仍只输出同一个 JSON（字段不变）。"""
    timeout = float(os.environ.get("SLC_PRIOR_TIMEOUT", "90"))
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=timeout, max_retries=0)
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(model=cfg["model"],
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=temperature),
            timeout=timeout)
        data = extract_json(rsp.choices[0].message.content or "")
        if not data:
            return None
        out = {"winner": None, "answer": None, "confidence": 0.0, "abstained": False,
               "rationale": str(data.get("rationale", ""))[:400]}
        if bool(data.get("abstained")):
            out["abstained"] = True
            return out
        if bucket == "pairwise" and data.get("winner"):
            w = str(data["winner"]).upper()
            out["winner"] = w if w in cands else None
        if bucket == "membership" and data.get("answer") is not None:
            out["answer"] = bool(data["answer"])
        try:
            out["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0) or 0.0)))
        except (TypeError, ValueError):
            pass
        if out["winner"] is None and out["answer"] is None:
            return None
        return out
    except Exception as e:  # noqa BLE001 —— 增强先验故障即缺席（留一行 stderr 便于事后定位）
        import sys
        print(f"[prior_agent] ask_prior_augmented 缺席: {type(e).__name__}: {str(e)[:160]}",
              file=sys.stderr, flush=True)
        return None
    finally:
        try:
            await client.close()
        except Exception:  # noqa BLE001
            pass


async def adjudicate_multi(question: str, bucket: str, candidates: list[str],
                           evidence: str, views: list, tally: str = "") -> dict | None:
    """v15 三通道表决：最终智能体同时看到【事实审料】与多路通道论点（证据通道 ∥ 纯LLM×2，
    纯LLM 通道零证据、独立采样），先逐一权衡辩论各方理由、再表决输出唯一结论。
    views = [(通道名, 通道观点), ...]。失败/弃权返回 None（不强行裁决，诚实缺席）。"""
    from openai import AsyncOpenAI

    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        return None
    cands = [c.upper() for c in (candidates or [])]
    task = ("Which candidate gene is more likely the winner? " if bucket == "pairwise"
            else "Is the claim true or false? ")
    chan_block = "\n\n".join(f"Channel {name}:\n{view}" for name, view in views)
    prompt = f"""You are the FINAL AGENT resolving a CRISPR essentiality question (bucket={bucket}) by DEBATING several independent channels and then VOTING one verdict.

Question: {question}
Candidates: {cands}

Factual observations (graph/statistics computed from databases; observations only, not conclusions):
{evidence}

Independent channels and their arguments:
{chan_block}

Current vote tally: {tally or 'none'}

{task}Debate internally: weigh each channel's argument against the factual observations and against the other channels (database evidence is association-level, not a CRISPR readout; a zero-evidence LLM vote may be mistaken or outdated, but two independent LLM votes agreeing is a real signal). Then cast your single verdict. If nothing is convincing, abstain.

Output JSON only: {{"winner": "<candidate or null>", "answer": true/false/null, "confidence": 0~1, "abstained": false, "rationale": "<=50 words"}}"""
    timeout = float(os.environ.get("SLC_JUDGE_TIMEOUT", "120"))
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=timeout, max_retries=0)
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(model=cfg["model"],
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0.2),
            timeout=timeout)
        data = extract_json(rsp.choices[0].message.content or "")
        if not data or bool(data.get("abstained")):
            return None
        out = {"winner": None, "answer": None, "confidence": 0.0,
               "rationale": str(data.get("rationale", ""))[:400]}
        if bucket == "pairwise" and data.get("winner"):
            w = str(data["winner"]).upper()
            out["winner"] = w if w in cands else None
        if bucket == "membership" and data.get("answer") is not None:
            out["answer"] = bool(data["answer"])
        try:
            out["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0) or 0.0)))
        except (TypeError, ValueError):
            pass
        if out["winner"] is None and out["answer"] is None:
            return None
        return out
    except Exception as e:  # noqa BLE001
        import sys
        print(f"[prior_agent] 法官缺席: {type(e).__name__}: {str(e)[:160]}",
              file=sys.stderr, flush=True)
        return None
    finally:
        try:
            await client.close()
        except Exception:  # noqa BLE001
            pass


async def ask_prior_raced(question: str, bucket: str, candidates: list[str], *,
                          persona: str = "", temperature: float = 0.2,
                          n: int | None = None, retrieval: str = "") -> dict | None:
    """v16 竞速：同一请求并行发 n 份，取第一个有效作答（治端点长尾超时）。
    n<=1 = 单发（逐字节同 ask_prior_augmented）。全部失败返回首个非 None（可能是弃权型，
    如实传递缺席信号）；多余任务取消，不留孤儿调用。v17：retrieval 透传给每份副本。"""
    n = prior_race_n() if n is None else n
    if n <= 1:
        return await ask_prior_augmented(question, bucket, candidates, digest="",
                                         conflict_hint="", temperature=temperature,
                                         persona=persona, retrieval=retrieval)
    pend = {asyncio.ensure_future(ask_prior_augmented(question, bucket, candidates,
                                                      digest="", conflict_hint="",
                                                      temperature=temperature,
                                                      persona=persona,
                                                      retrieval=retrieval))
            for _ in range(n)}
    fallback: dict | None = None
    try:
        while pend:
            done, pend = await asyncio.wait(pend, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                try:
                    r = t.result()
                except Exception:  # noqa BLE001
                    r = None
                if isinstance(r, dict) and not r.get("abstained"):
                    return r
                if isinstance(r, dict) and fallback is None:
                    fallback = r
        return fallback
    finally:
        for t in pend:
            t.cancel()


async def critique_prior(question: str, bucket: str, candidates: list[str],
                         persona: str, own_view: str, others_view: str) -> dict | None:
    """v16 交叉批评：通道保持自己的角色 persona，先攻最弱论点（他方的或自己的），
    再给修订后终票——只有当对方推理真驳倒自己才改票。输出同 ask_prior 形状；
    失败返回 None（调用方保留该通道原票，不强行改票）。"""
    from openai import AsyncOpenAI

    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        return None
    cands = [c.upper() for c in (candidates or [])]
    prompt = f"""{persona.strip()}

You are in a reasoned disagreement about a CRISPR essentiality question (bucket={bucket}).

Question: {question}
Candidates: {cands}

Your channel's earlier answer: {own_view}
Other channels' answers and arguments:
{others_view}

Step 1: briefly attack the weakest argument (theirs or your own). Step 2: give your REVISED final verdict — keep or change your earlier answer, but only switch if the other side's reasoning genuinely defeats yours.

Output JSON only: {{"winner": "<candidate or null>", "answer": true/false/null, "confidence": 0~1, "abstained": false, "rationale": "<=40 words"}}"""
    timeout = float(os.environ.get("SLC_CRITIQUE_TIMEOUT", "180"))
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=timeout, max_retries=0)
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(model=cfg["model"],
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0.4),
            timeout=timeout)
        data = extract_json(rsp.choices[0].message.content or "")
        if not data or bool(data.get("abstained")):
            return None
        out = {"winner": None, "answer": None, "confidence": 0.0,
               "rationale": str(data.get("rationale", ""))[:400]}
        if bucket == "pairwise" and data.get("winner"):
            w = str(data["winner"]).upper()
            out["winner"] = w if w in cands else None
        if bucket == "membership" and data.get("answer") is not None:
            out["answer"] = bool(data["answer"])
        try:
            out["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0) or 0.0)))
        except (TypeError, ValueError):
            pass
        if out["winner"] is None and out["answer"] is None:
            return None
        return out
    except Exception as e:  # noqa BLE001
        import sys
        print(f"[prior_agent] 批评缺席: {type(e).__name__}: {str(e)[:160]}",
              file=sys.stderr, flush=True)
        return None
    finally:
        try:
            await client.close()
        except Exception:  # noqa BLE001
            pass


async def adjudicate(question: str, bucket: str, candidates: list[str],
                     evidence: str, prior_view: str) -> dict | None:
    """v12 冲突裁决：LLM 法官同时看到【证据事实】与【先验论点】，产出最终判断。
    与辩论层的区别：单法官、看事实全文、允许 abstain。失败返回 None（不强行裁决）。"""
    from openai import AsyncOpenAI

    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        return None
    cands = [c.upper() for c in (candidates or [])]
    task = ("Which candidate gene is more likely the winner? " if bucket == "pairwise"
            else "Is the claim true or false? ")
    prompt = f"""You must resolve a disagreement between two analysis channels about a CRISPR essentiality question (bucket={bucket}).

Question: {question}
Candidates: {cands}

Channel A (graph/statistical evidence computed from databases):
{evidence}

Channel B (LLM literature-memory prior):
{prior_view}

{task}Weigh the reliability of each channel (database evidence is association-level, not a CRISPR readout; memory prior may be mistaken or outdated). If neither is convincing, abstain.

Output JSON only: {{"winner": "<candidate or null>", "answer": true/false/null, "confidence": 0~1, "abstained": false, "rationale": "<=40 words"}}"""
    timeout = float(os.environ.get("SLC_JUDGE_TIMEOUT", "120"))
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=timeout, max_retries=0)
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(model=cfg["model"],
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0.2),
            timeout=timeout)
        data = extract_json(rsp.choices[0].message.content or "")
        if not data or bool(data.get("abstained")):
            return None
        out = {"winner": None, "answer": None, "confidence": 0.0,
               "rationale": str(data.get("rationale", ""))[:400]}
        if bucket == "pairwise" and data.get("winner"):
            w = str(data["winner"]).upper()
            out["winner"] = w if w in cands else None
        if bucket == "membership" and data.get("answer") is not None:
            out["answer"] = bool(data["answer"])
        try:
            out["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0) or 0.0)))
        except (TypeError, ValueError):
            pass
        if out["winner"] is None and out["answer"] is None:
            return None
        return out
    except Exception as e:  # noqa BLE001
        import sys
        print(f"[prior_agent] 法官缺席: {type(e).__name__}: {str(e)[:160]}",
              file=sys.stderr, flush=True)
        return None
    finally:
        try:
            await client.close()
        except Exception:  # noqa BLE001
            pass
