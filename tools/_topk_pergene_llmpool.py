# -*- coding: utf-8 -*-
"""Per-gene (Kira) ranker on the LLM self-generated candidate pool.

Mirrors Phase 1 of tools/_topk_llm_enrich.py verbatim (same SCORE_PROMPT,
same build_score_prompt / parse_scores, temperature=0.2, enable_thinking off,
same tie-break ranking), but WITHOUT the MCP backend and WITHOUT _topk_head /
sklearn. It scores each pool gene 1-5 with a single LLM call per question and
ranks by (-score, gene name). Reads the pack from _topk_eval.PACK_PATH, which
is env-gated via SLC_TOPK_PACK; writes topk_eval_v20pergene{SLC_TOPK_SUFFIX}.json.

Usage:
  . tools\\slc_mars_env.ps1
  $env:NO_PROXY='*'; $env:PYTHONIOENCODING='utf-8'
  $env:SLC_TOPK_PACK='...\\outputs\\traces\\topk_eval_pack_llmpool.json'
  $env:SLC_TOPK_SUFFIX='_llmpool'
  .venv\\Scripts\\python.exe tools\\_topk_pergene_llmpool.py
"""
from __future__ import annotations
import asyncio, json, os, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))
sys.path.insert(0, str(ROOT / "tools"))

from _topk_eval import llm_settings, usage_triplet, ndcg_at_k, hit_at_k, PACK_PATH

TRACES = ROOT / "outputs" / "traces"
SUF = os.environ.get("SLC_TOPK_SUFFIX", "")

# ---- Per-gene scoring prompt (verbatim from _topk_llm_enrich.py) ----
SCORE_PROMPT = """你是癌症生物学专家。请对以下每个 SLC 基因在 {cancer} 中的功能相关性打分（1-5 分整数）。

评分标准:
  1 = 无已知关联或无关
  2 = 弱相关（仅泛癌代谢通路中的一般成员）
  3 = 中等相关（有文献报道但非核心角色）
  4 = 较强相关（在该癌种有明确功能研究，涉及关键通路）
  5 = 强相关（该癌种的已知驱动依赖/药物靶点/CRISPR 必需基因）

仅输出 JSON，不要额外解释:
{{
  "scores": {{"GENE1": 4, "GENE2": 3, ...}},
  "rationale_brief": "一句话总结评分逻辑"
}}"""


def build_score_prompt(cancer: str, genes: list[str]) -> str:
    gene_list = "\n".join(f"  - {g}" for g in sorted(genes))
    return SCORE_PROMPT.format(cancer=cancer, genes=gene_list) + f"\n基因列表:\n{gene_list}"


def parse_scores(text: str, valid_genes: set[str]) -> dict[str, int]:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        scores_raw = obj.get("scores", {})
        if not isinstance(scores_raw, dict):
            return {}
        scores = {}
        for g, s in scores_raw.items():
            g_upper = str(g).upper()
            if g_upper in valid_genes:
                try:
                    v = int(s)
                    scores[g_upper] = max(1, min(5, v))
                except (ValueError, TypeError):
                    pass
        return scores
    except json.JSONDecodeError:
        return {}


async def run():
    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        raise SystemExit("config2.yaml llm 配置缺失")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=300, max_retries=0)
    model = cfg["model"]
    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    print(f"[per-gene{SUF}] {len(packs)} 题, model={model}, pack={PACK_PATH.name}", flush=True)

    results = []
    t0 = time.time()
    total_tok = 0
    for i, pack in enumerate(packs):
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        cancer = (pack.get("constraints") or {}).get("cancer", "")
        valid = set(pool)
        prompt = build_score_prompt(cancer, pool)
        try:
            rsp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    extra_body={"enable_thinking": False},
                ), timeout=120)
            text = rsp.choices[0].message.content or ""
            _p, _c, t = usage_triplet(rsp)
            total_tok += t
            scores = parse_scores(text, valid)
            err = "" if scores else "no_parse"
        except Exception as e:
            scores = {}
            t = 0
            err = f"{type(e).__name__}"
            print(f"  [{i+1:2d}] {qid[-24:]:<24s} ERR {err}: {str(e)[:60]}", flush=True)

        ranked = sorted(pool, key=lambda g: (-scores.get(g, 0), g))
        pred = ranked[:5]
        ndcg = ndcg_at_k(pred, gt)
        hit = hit_at_k(pred, gt)
        results.append(dict(qid=qid, pred=pred, ndcg5=ndcg, hit5=hit,
                            tokens={"total": t, "calls": 1 if not err else 0},
                            error=err))
        print(f"  [{i+1:2d}] {qid[-24:]:<24s} scored={len(scores)}/{len(valid)} "
              f"NDCG={ndcg:.3f} hit={hit:.0f} tok={t} pred={pred[:3]}", flush=True)

    ok = [r for r in results if not r["error"]]
    ndcg = sum(r["ndcg5"] for r in ok) / len(ok) if ok else 0
    hit = sum(r["hit5"] for r in ok) / len(ok) if ok else 0
    out = dict(system=f"v20pergene{SUF}", n=len(results), n_ok=len(ok),
               n_err=len(results) - len(ok), results=results, total_tokens=total_tok)
    out_path = TRACES / f"topk_eval_v20pergene{SUF}.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[per-gene{SUF}] NDCG@5={ndcg:.4f}  hit@5={hit:.3f}  tok={total_tok}  "
          f"elapsed={time.time()-t0:.0f}s  -> {out_path.name}", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
