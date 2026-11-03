# -*- coding: utf-8 -*-
"""完整 LLM 全池排序 + v20graph_ppr RRF 融合。

与 top-5 近似版的区别：让 LLM 输出 pool 里所有 N 个基因的完整排序，
不再丢弃 #6~#N 的排名信息——LLM hit@5=0.848 的召回优势才能被 RRF 完整利用。
"""
from __future__ import annotations
import asyncio, json, math, os, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))
sys.path.insert(0, str(ROOT / "tools"))

from _topk_head import (fetch_path_rows, cap_per_gene, metrics_of,
                        fused_ranked, load_model as load_pairwise, prob_of,
                        CAP as ORIG_CAP, PACK_PATH, TRACES)
from _topk_eval import llm_settings, usage_triplet   # 复用 LLM 配置与 token 数
from knowledge_agents.graph_features import gene_graph_feat

TRACES_DIR = TRACES
K_RRF = 60


async def async_fetch_path_rows(be, pool: list[str], cancer: str) -> list[dict]:
    """async 版 fetch_path_rows —— 用 await be.acall() 避免嵌套事件循环。"""
    glist = ",".join("'" + g.replace("'", "") + "'" for g in pool)
    safe_cancer = cancer.replace("'", "")
    cy = (
        "MATCH (g:SLCGene)-[:IS_SOURCE]->(e:RelaEvent)<-[:IS_TARGET]-(d:Disease) "
        f"WHERE g.name IN [{glist}] "
        "OPTIONAL MATCH (p:Pathway)-[:IS_MEDIATOR]->(e) "
        "RETURN g.name AS gene, p.name AS pathway, e.rel_AB AS rel_AB, e.rel_BC AS rel_BC, "
        "CASE WHEN e.rel_AB = e.rel_BC THEN 'PROMOTION' ELSE 'SUPPRESSION' END AS gene_disease_effect, "
        "d.name AS disease, d.diseaseSort AS disease_sort "
        f"ORDER BY CASE WHEN toLower(d.name) CONTAINS toLower('{safe_cancer}') THEN 0 ELSE 1 END, d.name, g.name"
    )
    res = await be.acall("kg_cypher", {"query": cy, "limit": 1000})
    if res.get("status") != "ok":
        return []
    return res.get("rows") or []


FULLRANK_PROMPT = """你是生物医学专家。下面是一道 CRISPR 筛选排序题，要求你把候选池中全部基因按在该实验条件下成为必需基因的可能性从高到低做完整排序。

实验条件与问题:
{question}

候选 SLC 基因列表（共 {n_genes} 个，已随机打乱）:
{genes}

请仔细评估各基因在上述实验条件下抑制/促进肿瘤生长的可能性（基因功能、在该癌种中的已知角色、代谢/转运通路关联），然后输出一个 JSON，把所有候选基因按优先级从高到低全部列出：
{{
  "ranking": [{gene_list_placeholder}],
  "rationale": "简要说明排序逻辑（1-2句）"
}}

仅输出 JSON，不要额外解释。ranking 数组必须包含全部 {n_genes} 个基因名（大小写一致），不能多不能少。"""


def build_fullrank_prompt(question: str, genes: list[str], seed: int = 42) -> str:
    """乱序呈现全部候选基因，要求 LLM 完整排序。"""
    shown = [g.upper() for g in genes]
    import random
    random.Random(seed).shuffle(shown)
    placeholder = ", ".join(f'"{g}"' for g in shown)
    return FULLRANK_PROMPT.format(
        question=question,
        n_genes=len(shown),
        genes="\n".join(f"  - {g}" for g in shown),
        gene_list_placeholder=placeholder,
    )


def parse_full_ranking(text: str, valid_genes: set[str]) -> list[str] | None:
    """与 _topk_eval.parse_ranking 同逻辑，但不限长度——必须返回 pool 全部基因。"""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            ranking = obj.get("ranking", [])
            if isinstance(ranking, list):
                cleaned = []
                seen = set()
                for g in ranking:
                    g_upper = str(g).upper()
                    if g_upper in valid_genes and g_upper not in seen:
                        cleaned.append(g_upper)
                        seen.add(g_upper)
                if len(cleaned) == len(valid_genes):
                    return cleaned
                # 不全：补足缺失基因
                if len(cleaned) >= 1:
                    missing = [g for g in valid_genes if g not in seen]
                    return cleaned + sorted(missing)
        except json.JSONDecodeError:
            pass
    # fallback: 逐行扫描
    slc_re = re.compile(r"\b(SLC[A-Z0-9]+)\b", re.I)
    found = slc_re.findall(text)
    cleaned = []
    seen = set()
    for g in found:
        g_upper = g.upper()
        if g_upper in valid_genes and g_upper not in seen:
            cleaned.append(g_upper)
            seen.add(g_upper)
    if len(cleaned) == len(valid_genes):
        return cleaned
    if len(cleaned) >= 1:
        missing = [g for g in valid_genes if g not in seen]
        return cleaned + sorted(missing)
    return None


def rrf_rank(rank_a: int, rank_b: int) -> float:
    return 1.0 / (K_RRF + rank_a) + 1.0 / (K_RRF + rank_b)


async def run():
    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        raise SystemExit("config2.yaml 未找到 llm 配置")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=300 + 20, max_retries=0)
    model = cfg["model"]
    print(f"LLM: {cfg['base_url'][:40]}...  model={model}")

    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    m_graph = load_pairwise()

    # 取 v20graph_ppr 所需的图数据
    from knowledge_agents.tools.slc_backend import get_default_backend
    be = get_default_backend()
    if type(be).__name__ == "UnavailableBackend":
        raise SystemExit("MCP 后端不可用")

    rows = []
    t0 = time.time()
    for i, pack in enumerate(packs):
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        valid = set(pool)
        cancer = (pack.get("constraints") or {}).get("cancer", "")
        question = pack.get("question", "")
        ppr_named = len({g.upper() for g, _s in (pack.get("ppr_candidates") or [])
                         if g.upper() in valid})
        ppr_map = {g.upper(): (s or 0.0) for g, s in (pack.get("ppr_candidates") or [])}

        # ① 图谱特征
        raw = await async_fetch_path_rows(be, pool, cancer)
        capped = cap_per_gene(raw, ORIG_CAP)
        if not capped:
            rows.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                             pred=None, error="no_gene_disease_paths",
                             ndcg5=0.0, kendall_tau=0.0, precision5=0.0, hit5=0.0, mrr=0.0,
                             tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0},
                             ppr=None, seeds=[], n_seed_ok=1, per_seed=[], rationale=""))
            print(f"[{i+1:2d}] {qid[-24:]:<24} ERR no_paths")
            continue

        feats = {g: gene_graph_feat(g, capped, cancer) for g in pool}
        probs = {g: prob_of(feats[g], m_graph) for g in pool}
        vg_ranked = fused_ranked(pool, probs, ppr_map)
        vg_r = {g: idx for idx, g in enumerate(vg_ranked)}

        # ② LLM 完整排序
        prompt = build_fullrank_prompt(question, pool)
        q_t0 = time.time()
        llm_r = {}
        llm_tokens = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
        llm_error = ""
        try:
            extra = {"enable_thinking": False}
            rsp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    extra_body=extra,
                ), timeout=300)
            text = rsp.choices[0].message.content or ""
            p, c, t = usage_triplet(rsp)
            llm_tokens = {"prompt": p, "completion": c, "total": t, "calls": 1}
            full_rank = parse_full_ranking(text, valid)
            if full_rank and len(full_rank) == len(pool):
                llm_r = {g: idx for idx, g in enumerate(full_rank)}
            else:
                llm_error = f"parse_short({len(full_rank or [])}/{len(pool)})" if full_rank else "parse_failed"
                # fallback: 用下划线顺序
                for idx, g in enumerate(sorted(pool)):
                    llm_r[g] = idx
        except Exception as e:
            llm_error = f"{type(e).__name__}: {str(e)[:80]}"
            for idx, g in enumerate(sorted(pool)):
                llm_r[g] = idx

        # ③ RRF 融合
        fused = sorted(pool, key=lambda g: (-rrf_rank(vg_r[g], llm_r.get(g, len(pool))), g))
        pred = fused[:5]
        gm = metrics_of(pred, gt)

        q_elapsed = time.time() - q_t0
        overlap = len(set(vg_ranked[:5]) & set(pred))
        print(f"[{i+1:2d}] {qid[-24:]:<24} size={len(pool):>2d} "
              f"parse={'ok' if not llm_error else llm_error[:20]:<20s} "
              f"llm_s={q_elapsed:.0f} tok={llm_tokens['total']:>5d} "
              f"overlap@5={overlap} | "
              f"NDCG={gm['ndcg5']:.3f} hit@5={gm['hit5']:.0f} τ={gm['kendall_tau']:+.3f}")

        rows.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                         pred=pred, ppr=None, seeds=[], n_seed_ok=1, per_seed=[],
                         rationale=f"[llm_tokens={llm_tokens['total']}]",
                         tokens=llm_tokens, error="" if not llm_error else f"llm_{llm_error}",
                         **gm))

    ok = [r for r in rows if not r["error"]]
    out = TRACES_DIR / "topk_eval_v20rrf_llm_full.json"
    json.dump(dict(system="v20rrf_llm_full", n=len(rows), n_ok=len(ok),
                   n_err=len(rows) - len(ok), cap=ORIG_CAP, k_rrf=K_RRF,
                   results=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ndcg = sum(r["ndcg5"] for r in ok) / len(ok)
    hit = sum(r["hit5"] for r in ok) / len(ok)
    tau = sum(r["kendall_tau"] for r in ok) / len(ok)
    total_tok = sum(r["tokens"]["total"] for r in rows)
    print(f"\n写入 {out.name}  | total_tok={total_tok}  elapsed={time.time()-t0:.0f}s "
          f"NDCG@5={ndcg:.4f}  hit@5={hit:.3f}  τ={tau:+.4f}")


if __name__ == "__main__":
    asyncio.run(run())