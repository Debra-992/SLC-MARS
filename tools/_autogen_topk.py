# -*- coding: utf-8 -*-
"""AutoGen 等价基线：多轮对话排序（模拟 AutoGen assistant↔user_proxy 模式）。
与 AutoGen 等价：assistant 生成初始排序 → critic 复审修正 → 最终排序。
使用与 pure_llm/MetaGPT 同一 LLM 端点，关闭思考模式。

对应 AutoGen 模式：AssistantAgent(回答) ↔ UserProxyAgent(回显/终止)
此处简化为两轮 self-critique（等价于 max_turns=2 的 AutoGen 行为）。

Usage: .venv\Scripts\python.exe tools\_autogen_topk.py
"""
import asyncio, json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from _topk_eval import PACK_PATH, llm_settings, ndcg_at_k, hit_at_k

TRACES = ROOT / "outputs" / "traces"

TOPK_PROMPT = """你是生物医学专家。下面是一道 CRISPR 筛选排序题，需从候选基因中选出最符合题意的 top-5，并按优先级排序。

实验条件与问题:
{question}

候选 SLC 基因列表（已随机打乱，无序）:
{genes}

请评估各基因在上述实验条件下抑制/促进肿瘤生长的可能性（基因功能、在该癌种中的已知角色、代谢/转运通路关联），然后输出一个 JSON：
{{
  "ranking": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
  "rationale": "简要说明排序逻辑（1-2句）"
}}

仅输出 JSON，不要额外解释。ranking 必须是候选列表中的基因名（大小写一致），5 个全部不同。"""

CRITIQUE_PROMPT = """你是一位资深审稿人。请审阅以下基因排序结果的合理性。

原始问题:
{question}

候选基因池: {genes}

第一轮排序结果: {ranking}

请判断该排序是否合理，如果不合理请给出修正后的排序。输出 JSON：
{{
  "verdict": "agree" 或 "revise",
  "ranking": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
  "rationale": "审阅意见（1-2句）"
}}

仅输出 JSON。ranking 必须是候选列表中的基因名，5 个全部不同。"""


def build_prompt(question: str, genes: list[str], seed: int = 42) -> str:
    import random
    shown = [g.upper() for g in genes]
    random.Random(seed).shuffle(shown)
    return TOPK_PROMPT.format(question=question, genes="\n".join(f"  - {g}" for g in shown))


def parse_ranking(text: str, valid: set[str]) -> list[str] | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            ranking = obj.get("ranking", [])
            if isinstance(ranking, list):
                cleaned = []
                seen = set()
                for g in ranking:
                    gu = str(g).upper()
                    if gu in valid and gu not in seen:
                        cleaned.append(gu)
                        seen.add(gu)
                if len(cleaned) >= 1:
                    return cleaned
        except json.JSONDecodeError:
            pass
    slc_re = re.compile(r"\b(SLC[A-Z0-9]+)\b", re.I)
    found = slc_re.findall(text)
    cleaned = []
    seen = set()
    for g in found:
        gu = g.upper()
        if gu in valid and gu not in seen:
            cleaned.append(gu)
            seen.add(gu)
    return cleaned if len(cleaned) >= 1 else None


async def run_autogen():
    from openai import AsyncOpenAI

    cfg = llm_settings()
    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    model = cfg["model"]
    client = AsyncOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        timeout=120,
        max_retries=0,
    )
    print(f"[AutoGen] topK baseline (2-turn self-critique), {len(packs)} 题, model={model}", flush=True)

    results = []
    t0 = time.time()
    total_tok = 0

    SYS_ASSISTANT = "你是生物医学专家，请直接输出 JSON 格式的 SLC 基因排序结果。"
    SYS_CRITIC = "你是资深审稿人，请审阅基因排序结果，可同意或修正。"

    for i, pack in enumerate(packs):
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        valid = set(pool)
        question = pack.get("question", "")
        prompt = build_prompt(question, pool)

        try:
            # Turn 1: assistant generates initial ranking
            rsp1 = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYS_ASSISTANT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                extra_body={"enable_thinking": False},
                timeout=120,
            )
            rsp1_str = rsp1.choices[0].message.content or ""
            ranking1 = parse_ranking(rsp1_str, valid)
            usage1 = rsp1.usage

            # Turn 2: critic reviews and potentially revises
            if ranking1 and len(ranking1) >= 3:
                critique_prompt = CRITIQUE_PROMPT.format(
                    question=question,
                    genes=", ".join(sorted(valid)),
                    ranking=", ".join(ranking1[:5]),
                )
                rsp2 = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYS_CRITIC},
                        {"role": "user", "content": critique_prompt},
                    ],
                    temperature=0.1,
                    extra_body={"enable_thinking": False},
                    timeout=120,
                )
                rsp2_str = rsp2.choices[0].message.content or ""
                ranking2 = parse_ranking(rsp2_str, valid)
                usage2 = rsp2.usage
                # Use critic's revision if available, otherwise keep original
                final_ranking = ranking2 if ranking2 else ranking1
            else:
                final_ranking = ranking1
                usage2 = type('obj', (object,), {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})()

            tok = {
                "prompt": usage1.prompt_tokens + usage2.prompt_tokens,
                "completion": usage1.completion_tokens + usage2.completion_tokens,
                "total": usage1.total_tokens + usage2.total_tokens,
                "calls": 2,
            }
        except Exception as e:
            final_ranking = None
            tok = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
            print(f"  [{i+1:2d}] {qid[-24:]:<24s} AutoGen ERR: {type(e).__name__}: {str(e)[:60]}", flush=True)

        pred = final_ranking[:5] if final_ranking else []
        ndcg = ndcg_at_k(pred, gt) if pred else 0.0
        hit = hit_at_k(pred, gt) if pred else 0.0
        results.append(dict(qid=qid, pred=pred, ndcg5=ndcg, hit5=hit, tokens=tok,
                        error="" if final_ranking else "no_parse"))
        total_tok += tok["total"]
        print(f"  [{i+1:2d}] {qid[-24:]:<24s} NDCG={ndcg:.3f} hit={hit:.0f} tok={tok['total']} "
              f"pred={pred[:3] if pred else 'NONE'}", flush=True)

    ok = [r for r in results if not r["error"]]
    ndcg = sum(r["ndcg5"] for r in ok) / len(ok) if ok else 0
    hit = sum(r["hit5"] for r in ok) / len(ok) if ok else 0
    out = dict(system="autogen_topk", n=len(results), n_ok=len(ok),
               n_err=len(results) - len(ok), results=results,
               total_tokens=total_tok)
    json.dump(out, open(TRACES / "topk_eval_autogen.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[AutoGen] NDCG@5={ndcg:.4f}  hit@5={hit:.3f}  tok={total_tok}  "
          f"elapsed={time.time()-t0:.0f}s  -> topk_eval_autogen.json", flush=True)
    return ndcg, hit


if __name__ == "__main__":
    asyncio.run(run_autogen())