# -*- coding: utf-8 -*-
"""LLM 知识驱动候选池构建 + 全臂评测（独立旁路，不碰现有 harness）。

流程：
  1. 对 11 题逐癌种调 LLM 提名 top-15 相关 SLC 基因
  2. pool = LLM_nominated ∪ GT（去重）
  3. 过滤 ppr_candidates 到新 pool 内
  4. 写 topk_eval_pack_llmpool.json
  5. 用 SLC_TOPK_PACK 环境变量指向新区，分别跑 pure_llm / pipeline / v20graph_ppr / RRF 融合

用法:
  . tools\slc_mars_env.ps1
  $env:NO_PROXY='*'; $env:PYTHONIOENCODING='utf-8'
  .venv\Scripts\python.exe tools\_topk_pool_llm.py          # 构建池 + 跑全部臂
  .venv\Scripts\python.exe tools\_topk_pool_llm.py --build-only  # 只构建池
  .venv\Scripts\python.exe tools\_topk_pool_llm.py --eval-only   # 只在已有池上跑评测
"""

from __future__ import annotations
import argparse, asyncio, json, math, os, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))
sys.path.insert(0, str(ROOT / "tools"))

from _topk_eval import (llm_settings, usage_triplet, ndcg_at_k, kendall_tau,
                        precision_at_k, hit_at_k, mrr, ppr_ranking,
                        PACK_PATH as ORIG_PACK_PATH)

PACK_LLMPOOL = ROOT / "outputs" / "traces" / "topk_eval_pack_llmpool.json"
TRACES = ROOT / "outputs" / "traces"
K_NOMINATE = 15  # 每题 LLM 提名基因数

# ---- LLM 提名 prompt ----
NOMINATE_PROMPT = """你是癌症生物学专家。针对 {cancer} 癌种，列出在肿瘤代谢重编程、营养转运、药物抵抗、氧化还原平衡等方面有已知功能角色的 SLC 转运蛋白家族基因。

请输出恰好 {k} 个最相关的 SLC 基因，按相关性从高到低排列。如果对该癌种了解不充分，请基于泛癌代谢转运通路的常识推断。

仅输出 JSON，不要额外解释：
{{
  "genes": ["SLC1A5", "SLC7A11", ...]
}}"""


# ---- 构建 LLM 池 ----
async def nominate_one(client, model: str, cancer: str, k: int = K_NOMINATE) -> list[str]:
    """LLM 为单个癌种提名 k 个 SLC 基因。"""
    prompt = NOMINATE_PROMPT.format(cancer=cancer, k=k)
    extra = {"enable_thinking": False}
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                extra_body=extra,
            ), timeout=60)
        text = rsp.choices[0].message.content or ""
    except Exception as e:
        print(f"  LLM 提名 {cancer} 失败: {type(e).__name__}: {e}")
        return []

    # 解析 JSON
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        print(f"  LLM 提名 {cancer} 无 JSON 输出")
        return []
    try:
        obj = json.loads(m.group(0))
        genes = obj.get("genes", [])
    except json.JSONDecodeError:
        print(f"  LLM 提名 {cancer} JSON 解析失败")
        return []

    # 清洗：只保留 SLC 基因名，去重，截断到 k
    slc_re = re.compile(r"^(SLC[A-Z0-9]+)$", re.I)
    cleaned = []
    seen = set()
    for g in genes:
        g_upper = str(g).strip().upper()
        if slc_re.match(g_upper) and g_upper not in seen:
            cleaned.append(g_upper)
            seen.add(g_upper)
        if len(cleaned) >= k:
            break
    return cleaned


async def build_pools():
    """对每题构建 LLM 知识候选池。癌症相同则复用提名结果。"""
    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        raise SystemExit("config2.yaml 未找到 llm 配置")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=120, max_retries=1)
    model = cfg["model"]
    print(f"LLM 提名: {cfg['base_url'][:40]}...  model={model}")

    packs = json.loads(ORIG_PACK_PATH.read_text(encoding="utf-8"))["packs"]

    # 收集唯一癌种
    cancer_set = set()
    for pack in packs:
        c = (pack.get("constraints") or {}).get("cancer", "").strip()
        if c:
            cancer_set.add(c)

    # 逐癌种调 LLM 提名
    cancer_genes: dict[str, list[str]] = {}
    for cancer in sorted(cancer_set):
        genes = await nominate_one(client, model, cancer)
        cancer_genes[cancer] = genes
        print(f"  {cancer:<12s} → {len(genes)} genes: {', '.join(genes[:5])}..."
              if len(genes) > 5 else f"  {cancer:<12s} → {len(genes)} genes: {genes}")

    # 构建新 pack
    new_packs = []
    for pack in packs:
        cancer = (pack.get("constraints") or {}).get("cancer", "").strip()
        gt = [g.upper() for g in pack["gt_ranking"]]
        nominated = cancer_genes.get(cancer, [])

        # pool = LLM_nominated ∪ GT
        pool = list(dict.fromkeys(nominated + gt))  # 保持 LLM 提名序在前，GT 补充在后

        # 过滤 ppr_candidates 只保留在新区内的
        old_ppr = pack.get("ppr_candidates") or []
        pool_set = set(pool)
        new_ppr = [[g, s] for g, s in old_ppr if g.upper() in pool_set]

        new_pack = {
            "qid": pack["qid"],
            "question": pack["question"],
            "constraints": pack.get("constraints"),
            "task_type": pack.get("task_type"),
            "gt_ranking": gt,
            "ppr_candidates": new_ppr,
            "ppr_candidate_count": len(new_ppr),
            "eval_pool": pool,
            "pool_source": "llm_nominated",
        }
        new_packs.append(new_pack)
        print(f"  {pack['qid'][-24:]:<24s} pool={len(old_ppr):>2d}→{len(pool):>2d}  "
              f"ppr_kept={len(new_ppr)}/{len(old_ppr)}  "
              f"GT_in={len(set(gt)&pool_set)}")

    out = {"packs": new_packs, "pool_source": "llm_nominated", "k_nominate": K_NOMINATE}
    json.dump(out, open(PACK_LLMPOOL, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n写入 {PACK_LLMPOOL.name}  ({len(new_packs)} 题)")
    return new_packs


# ---- 评测（在新池上跑各臂） ----
def run_pure_llm_and_pipeline(pack_path: Path, label: str):
    """在新池上跑 pure_llm + pipeline 臂。直接调 _topk_eval.main 逻辑。"""
    import subprocess
    py = sys.executable
    env = os.environ.copy()
    env["SLC_TOPK_PACK"] = str(pack_path)
    env["NO_PROXY"] = "*"
    env["PYTHONIOENCODING"] = "utf-8"

    for hint, sys_name in (("none", "pure_llm_llmpool"), ("ppr_order", "pipeline_llmpool")):
        out_path = TRACES / f"topk_eval_{sys_name}.json"
        print(f"\n--- {sys_name} (hint={hint}) ---")
        cmd = [py, str(ROOT / "tools" / "_topk_eval.py"),
               "--hint", hint, "--system", sys_name]
        # _topk_eval writes to traces/topk_eval_{system}.json
        # so we need to either modify it or rename after
        r = subprocess.run(cmd, env=env, cwd=str(ROOT),
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"ERR: {r.stderr[:500]}")
        else:
            out_lines = [l for l in r.stdout.splitlines() if l.strip()]
            for l in out_lines[-10:]:
                print(f"  {l}")


def run_graph_arms(pack_path: Path):
    """在新池上跑 v20graph + v20graph_ppr 臂。"""
    import subprocess
    py = sys.executable
    env = os.environ.copy()
    env["SLC_TOPK_PACK"] = str(pack_path)
    env["NO_PROXY"] = "*"
    env["PYTHONIOENCODING"] = "utf-8"

    # _topk_head.py writes to TRACES / topk_eval_v20graph.json and topk_eval_v20graph_ppr.json
    # We need to rename them to avoid overwriting originals
    print(f"\n--- v20graph + v20graph_ppr (LLM pool) ---")
    cmd = [py, str(ROOT / "tools" / "_topk_head.py")]
    r = subprocess.run(cmd, env=env, cwd=str(ROOT),
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"ERR: {r.stderr[:500]}")
    else:
        out_lines = [l for l in r.stdout.splitlines() if l.strip()]
        for l in out_lines[-20:]:
            print(f"  {l}")

    # Rename output files
    for name in ("v20graph", "v20graph_ppr"):
        src = TRACES / f"topk_eval_{name}.json"
        dst = TRACES / f"topk_eval_{name}_llmpool.json"
        if src.exists():
            src.rename(dst)
            print(f"  重命名 {src.name} → {dst.name}")


def run_rrf_fusion():
    """在新池结果上做 v20graph_ppr + pure_llm RRF 融合（近似版：unknown rank=pool_size）。"""
    packs = json.loads(PACK_LLMPOOL.read_text(encoding="utf-8"))["packs"]
    vg_data = json.loads((TRACES / "topk_eval_v20graph_ppr_llmpool.json").read_text(encoding="utf-8"))
    llm_data = json.loads((TRACES / "topk_eval_pure_llm_llmpool.json").read_text(encoding="utf-8"))

    vg_map = {r["qid"]: r for r in vg_data["results"]}
    llm_map = {r["qid"]: r for r in llm_data["results"]}

    def rrf_score(rank_a, rank_b, k=60):
        return 1.0 / (k + rank_a) + 1.0 / (k + rank_b)

    rows = []
    for pack in packs:
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        vg_r = vg_map.get(qid, {})
        llm_r = llm_map.get(qid, {})

        vg_pred = vg_r.get("pred") or []
        llm_pred = llm_r.get("pred") or []

        # v20graph_ppr 完整排序（从 pred 反推不出完整序，用 fused_ranked 重建）
        # 简化：用 pred 顺序
        vg_rank = {}
        for idx, g in enumerate(vg_pred):
            vg_rank[g] = idx
        for g in pool:
            if g not in vg_rank:
                vg_rank[g] = len(pool)

        llm_rank = {}
        for idx, g in enumerate(llm_pred):
            llm_rank[g] = idx
        for g in pool:
            if g not in llm_rank:
                llm_rank[g] = len(pool)

        fused = sorted(pool, key=lambda g: (-rrf_score(vg_rank[g], llm_rank[g]), g))
        pred = fused[:5]
        ndcg = ndcg_at_k(pred, gt, k=5)
        hit = hit_at_k(pred, gt, k=5)
        tau = kendall_tau(pred, gt)
        rows.append(dict(qid=qid, pool_size=len(pool), gt=gt,
                         pred=pred, ndcg5=ndcg, hit5=hit, kendall_tau=tau,
                         precision5=precision_at_k(pred, gt, k=5),
                         mrr=mrr(pred, gt),
                         error=""))

    ok = [r for r in rows if not r["error"]]
    out_path = TRACES / "topk_eval_v20rrf_llm_llmpool.json"
    json.dump(dict(system="v20rrf_llm_llmpool", n=len(rows), n_ok=len(ok),
                   n_err=len(rows)-len(ok), results=rows),
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ndcg = sum(r["ndcg5"] for r in ok) / len(ok) if ok else 0
    hit = sum(r["hit5"] for r in ok) / len(ok) if ok else 0
    tau = sum(r["kendall_tau"] for r in ok) / len(ok) if ok else 0
    print(f"\nRRF 融合 (LLM pool): NDCG@5={ndcg:.4f}  hit@5={hit:.3f}  tau={tau:+.4f}  → {out_path.name}")
    return ndcg, hit, tau


def compare():
    """对比 PPR 池 vs LLM 池各臂 NDCG@5。"""
    import glob as _glob

    def _load(path):
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        ok = [r for r in d["results"] if not r["error"]]
        return sum(r["ndcg5"] for r in ok) / len(ok) if ok else 0, \
               sum(r["hit5"] for r in ok) / len(ok) if ok else 0

    print("\n" + "=" * 80)
    print(f"{'Arm':<30s} {'NDCG@5':>8s} {'hit@5':>8s}  {'池':>10s}")
    print("-" * 60)

    rows = []
    # PPR 池（原始）
    for sys_name in ("pure_llm", "pipeline", "v20graph", "v20graph_ppr", "v20rrf_llm", "v20full", "v20rrf_llm_full"):
        p = TRACES / f"topk_eval_{sys_name}.json"
        r = _load(p)
        if r:
            ndcg, hit = r
            rows.append((sys_name, ndcg, hit, "PPR"))

    # LLM 池
    for sys_name in ("pure_llm_llmpool", "pipeline_llmpool", "v20graph_llmpool",
                     "v20graph_ppr_llmpool", "v20rrf_llm_llmpool"):
        p = TRACES / f"topk_eval_{sys_name}.json"
        r = _load(p)
        if r:
            ndcg, hit = r
            rows.append((sys_name, ndcg, hit, "LLM"))

    # 按 NDCG 降序
    rows.sort(key=lambda x: -x[1])
    for name, ndcg, hit, pool in rows:
        print(f"{name:<30s} {ndcg:>8.4f} {hit:>8.3f}  {pool:>10s}")


# ---- 入口 ----
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if not args.eval_only:
        await build_pools()

    if not args.build_only:
        # 设置环境变量让子脚本读新区
        os.environ["SLC_TOPK_PACK"] = str(PACK_LLMPOOL)

        run_pure_llm_and_pipeline(PACK_LLMPOOL, "llmpool")
        run_graph_arms(PACK_LLMPOOL)
        run_rrf_fusion()
        compare()


if __name__ == "__main__":
    asyncio.run(main())