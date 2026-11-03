# -*- coding: utf-8 -*-
"""topk LLM 重排独立评测 harness（不碰 compare_eval 框架，零回归风险）。

评测逻辑：
  1. 加载预计算的 topk_eval_pack.json（PPR top-15 ∪ GT 候选池）
  2. 对每题：固定种子 shuffle 候选池 → 组装 prompt → LLM 重排 → 解析有序基因列表
  3. 计算 NDCG@5 / Kendall τ / Precision@5（三指标）
  4. 对比 PPR 基线（纯图谱邻近度排序）
  5. 输出逐题明细 + 汇总表

用法:
  . tools\slc_mars_env.ps1
  .venv\Scripts\python.exe tools\_topk_eval.py                        # 跑全 11 题
  .venv\Scripts\python.exe tools\_topk_eval.py --dry-run              # 只验证逻辑，不调 LLM
  .venv\Scripts\python.exe tools\_topk_eval.py --system pure_llm      # 指定系统名（默认 pipeline）
  .venv\Scripts\python.exe tools\_topk_eval.py --hint none            # 候选按随机序给 LLM（纯LLM，剥离图谱顺序）

三臂对照（--hint 决定候选池以什么顺序呈现给 LLM）：
  ppr_order（默认）= 图谱 PPR 先给排序，LLM 只负责重排 → "管线·PPR+LLM重排"
  none             = 只给乱序候选池，LLM 裸排         → "纯LLM基线"
  纯 PPR 图谱基线无需调用 LLM，由 pack 里的 ppr_candidates 直接算出，同表列出。

采样形态（2026-10-30 实测，务必保持关思考，否则整批 100% 超时）：
  qwen3.8-flash 在 openai 兼容模式下【默认思考开】，topk 这种"逐个权衡 18~20 个基因"的题
  会一口气生成 1.8 万+ reasoning token、单题 271s，远超超时；显式
  extra_body={"enable_thinking": False} 后降到 2.8s / 160 token。
  可用 SLC_TOPK_THINKING=1 回退到思考开（配 SLC_TOPK_LLM_TIMEOUT 放宽）做 A/B。
"""
from __future__ import annotations
import argparse, asyncio, json, math, os, random, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))

# ---- 可独立运行（复用 common.py 的 llm_settings） ----
try:
    from common import llm_settings, usage_triplet
except ImportError:
    # fallback: 若不在 baselines 子目录加载，直接解析（env 优先，同 common.llm_settings 语义）
    def llm_settings():
        cfg = ROOT / "config2.yaml"
        txt = cfg.read_text(encoding="utf-8", errors="replace") if cfg.exists() else ""
        out = {"base_url": "", "api_key": "", "model": ""}
        in_llm = False
        for line in txt.splitlines():
            if re.match(r"^llm:\s*$", line):
                in_llm = True; continue
            if in_llm and re.match(r"^\w", line):
                in_llm = False
            if in_llm:
                for k in out:
                    if f"{k}:" in line:
                        out[k] = line.split(":", 1)[1].strip().strip('"').strip("'")
        for k in out:
            envv = os.environ.get("SLC_LLM_" + k.upper(), "")
            if envv:
                out[k] = envv
            elif out[k] == "REPLACE_VIA_SECRETS_LOCAL_PS1":
                out[k] = ""
        return out

    def usage_triplet(rsp):
        u = getattr(rsp, 'usage', None)
        return (u.prompt_tokens, u.completion_tokens, u.total_tokens) if u else (0, 0, 0)


PACK_PATH = Path(os.environ.get("SLC_TOPK_PACK", ROOT / "outputs" / "traces" / "topk_eval_pack.json"))

# ---- 指标 ----
def ndcg_at_k(pred: list[str], gt: list[str], k: int = 5) -> float:
    gt_rank = {g: i for i, g in enumerate(gt)}
    rel = {g: (1.0 / math.log2(gt_rank[g] + 2)) for g in gt}
    dcg = sum(rel.get(g, 0) / math.log2(i + 2) for i, g in enumerate(pred[:k]))
    ideal_scores = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal_scores))
    return dcg / idcg if idcg > 0 else 0.0


def kendall_tau(pred: list[str], gt: list[str]) -> float:
    """Kendall τ：两序列【共有基因】上的成序一致度 ∈ [-1,1]。

    只在 pred ∩ gt 上配对（τ-a 口径）。不能用"pred 里没出现的 gt 基因算作排在最末"
    的口径（τ-b），因为本任务 pred 只给 top-5、gt 有 5 个，未选中的 gt 基因会被
    人为判成"次序正确"，凭空抬高一致度。共有基因不足 2 个 → 序关系不可测，记 0。
    """
    common = [g for g in pred if g in gt]
    if len(common) < 2:
        return 0.0
    gt_rank = {g: i for i, g in enumerate(gt)}
    pred_rank = {g: i for i, g in enumerate(pred)}
    concordant = discordant = 0
    for i, a in enumerate(common):
        for b in common[i + 1:]:
            if (pred_rank[a] < pred_rank[b]) == (gt_rank[a] < gt_rank[b]):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


def precision_at_k(pred: list[str], gt: list[str], k: int = 5) -> float:
    return len(set(pred[:k]) & set(gt[:k])) / k


def hit_at_k(pred: list[str], gt: list[str], k: int = 5) -> float:
    """top-k 里是否至少捞到一个真值（0/1），比 PR@5 在小 n 上更稳。"""
    return 1.0 if set(pred[:k]) & set(gt[:k]) else 0.0


def mrr(pred: list[str], gt: list[str]) -> float:
    """第一个真值命中的倒数排名；完全没命中记 0。"""
    gtset = set(gt)
    for i, g in enumerate(pred):
        if g in gtset:
            return 1.0 / (i + 1)
    return 0.0


def ppr_ranking(pack: dict, seed: int = 0) -> list[str] | None:
    """纯 PPR 图谱基线的有序列表：PPR 候选在前，池内其余基因随机垫底。

    候选池 pool = "PPR top-15 ∪ GT top-5"，GT 基因被追加在 pool 尾部（下标 15~19）。
    尾部那批"图谱没提名"的基因顺序【不参与评测口径】，所以这里把尾部固定 seed shuffle，
    既避免 pool 原始顺序（=GT 序）直接泄题，也不影响图谱真正提名的前 15 位。

    返回 None = 该题图谱零提名（PPR top-15 与候选池无交集），基线不可测，
    必须从 PPR 汇总里剔除；若兜底返回整池则会把 GT 顺序当成 PPR 排序（假 1.000）。
    """
    pool = [g.upper() for g in pack["eval_pool"]]
    poolset = set(pool)
    head = [g.upper() for g, _s in pack.get("ppr_candidates") or [] if g.upper() in poolset]
    if not head:
        return None                                   # 图谱完全没提名 → 不可测
    head_set = set(head)
    tail = [g for g in pool if g not in head_set]
    random.Random(seed).shuffle(tail)
    return head + tail


# ---- Prompt ----
# 两臂【共用同一主体】，只差候选池的呈现顺序，保证 pipeline vs pure_llm 是唯一变量。
TOPK_PROMPT = """你是生物医学专家。下面是一道 CRISPR 筛选排序题，需从候选基因中选出最符合题意的 top-5，并按优先级排序。

实验条件与问题:
{question}

候选 SLC 基因列表（{order_note}）:
{genes}

请评估各基因在上述实验条件下抑制/促进肿瘤生长的可能性（基因功能、在该癌种中的已知角色、代谢/转运通路关联），然后输出一个 JSON：
{{
  "ranking": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
  "rationale": "简要说明排序逻辑（1-2句）"
}}

仅输出 JSON，不要额外解释。ranking 必须是候选列表中的基因名（大小写一致），5 个全部不同。"""

_ORDER_NOTE = {"ppr_order": "按图谱 PageRank 邻近度由高到低给出",
               "none": "已随机打乱，无序"}


def build_topk_prompt(question: str, genes: list[str], seed: int = 42,
                      hint: str = "ppr_order",
                      ppr_cands: list | None = None) -> tuple[str, list[str]]:
    """返回 (prompt, 呈现给模型的基因序)。

    hint="ppr_order" → 图谱提名段按 PPR 降序呈现，图谱没提名的基因随机插到后面
                       （不能沿用 pool 原序——那批正是 GT，尾部固定等于泄题）；
    hint="none"      → 用 seed 固定 shuffle，整池乱序（LLM 裸排，无任何图谱顺序）。
    两条臂拿到的基因【集合完全一致】，只差顺序，故 Δ 可归因到"有没有图谱先验"。
    """
    if hint == "ppr_order":
        pool = [g.upper() for g in genes]
        poolset = set(pool)
        named = {g.upper() for g, _s in (ppr_cands or []) if g.upper() in poolset}
        head = [g for g in pool if g in named]        # 保持传入的 PPR 降序
        tail = [g for g in pool if g not in named]
        random.Random(seed).shuffle(tail)             # 未提名段随机化，去掉尾序泄漏
        shown = head + tail
    else:
        shown = [g.upper() for g in genes]
        random.Random(seed).shuffle(shown)
    return TOPK_PROMPT.format(
        question=question,
        order_note=_ORDER_NOTE[hint],
        genes="\n".join(f"  - {g}" for g in shown),
    ), shown


def parse_ranking(text: str, valid_genes: set[str]) -> list[str] | None:
    """从 LLM 输出中提取有序基因列表。先试 JSON，失败则逐行扫描。"""
    # 1) JSON 提取
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            ranking = obj.get("ranking", [])
            if isinstance(ranking, list) and len(ranking) >= 1:
                # 仅保留 valid_genes 中的基因（去噪）
                cleaned = []
                seen = set()
                for g in ranking:
                    g_upper = g.upper()
                    if g_upper in valid_genes and g_upper not in seen:
                        cleaned.append(g_upper)
                        seen.add(g_upper)
                if cleaned:
                    return cleaned
        except json.JSONDecodeError:
            pass
    # 2) 逐行扫描 SLC 基因名
    slc_re = re.compile(r"\b(SLC[A-Z0-9]+)\b", re.I)
    found = slc_re.findall(text)
    cleaned = []
    seen = set()
    for g in found:
        g_upper = g.upper()
        if g_upper in valid_genes and g_upper not in seen:
            cleaned.append(g_upper)
            seen.add(g_upper)
    return cleaned if len(cleaned) >= 1 else None


# ---- 主流程 ----
def _thinking_off_body() -> dict | None:
    """端点 qwen3.8-flash 兼容模式默认【思考开】，topk 大池题会生成上万 reasoning token 直接超时。
    默认显式关闭；SLC_TOPK_THINKING=1 可回退思考开（需同步放宽 SLC_TOPK_LLM_TIMEOUT）。"""
    if os.environ.get("SLC_TOPK_THINKING", "0") == "1":
        return None
    return {"enable_thinking": False}


async def _try_one_call(client, model: str, prompt: str, valid: set, timeout: float) -> tuple:
    """单次 LLM 调用，返回 (pred_list | None, rationale, tokens, error)."""
    try:
        kw = dict(model=model,
                  messages=[{"role": "user", "content": prompt}],
                  temperature=0.2)
        extra = _thinking_off_body()
        if extra is not None:
            kw["extra_body"] = extra
        rsp = await asyncio.wait_for(client.chat.completions.create(**kw), timeout=timeout)
        text = rsp.choices[0].message.content or ""
        p, c, t = usage_triplet(rsp)
        tokens = {"prompt": p, "completion": c, "total": t, "calls": 1}
        pred = parse_ranking(text, valid)
        rationale = text[:200]
        error = "" if pred else "parse_failed"
        return pred, rationale, tokens, error
    except Exception as e:
        return None, "", {"prompt": 0, "completion": 0, "total": 0, "calls": 1}, f"{type(e).__name__}: {e}"[:200]


async def eval_one(client, model: str, pack: dict, seeds: list[int], dry_run: bool,
                   hint: str = "ppr_order", max_retries: int = 4) -> dict:
    qid = pack["qid"]
    pool = pack["eval_pool"]
    gt = pack["gt_ranking"]
    valid = {g.upper() for g in pool}
    ppr_cands = pack.get("ppr_candidates") or []

    # 纯 PPR 图谱基线（同题集同候选池，零 LLM 调用）
    gt_upper = [g.upper() for g in gt]
    ppr_pred = ppr_ranking(pack)
    ppr_metrics = None if ppr_pred is None else dict(
        ndcg5=ndcg_at_k(ppr_pred, gt_upper, k=5),
        kendall_tau=kendall_tau(ppr_pred, gt_upper),
        precision5=precision_at_k(ppr_pred, gt_upper, k=5),
        hit5=hit_at_k(ppr_pred, gt_upper, k=5),
        mrr=mrr(ppr_pred, gt_upper),
    )

    # LLM 臂：对每个 seed 各跑一次，逐 seed 记录（pool 顺序对 LLM 有锚定效应，多 seed 才能看方差）
    per_seed = []
    for seed in seeds:
        prompt, shown = build_topk_prompt(pack["question"], pool, seed=seed, hint=hint,
                                          ppr_cands=ppr_cands)
        if dry_run:
            pred = [g.upper() for g in shown[:5]]
            rationale, error = "[dry-run]", ""
            tokens = {"prompt": 0, "completion": 0, "total": 0, "calls": 1}
        else:
            pred, rationale, error = None, "", ""
            tokens = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
            timeout = float(os.environ.get("SLC_TOPK_LLM_TIMEOUT", "300"))
            for attempt in range(max_retries):
                p, rat, tok, err = await _try_one_call(client, model, prompt, valid, timeout)
                tokens["prompt"] += tok["prompt"]
                tokens["completion"] += tok["completion"]
                tokens["total"] += tok["total"]
                tokens["calls"] += tok["calls"]
                if p is not None:
                    pred, rationale, error = p, rat, ""
                    break
                error = err
                if attempt < max_retries - 1:
                    await asyncio.sleep(5.0)
        per_seed.append(dict(
            seed=seed, pred=pred, rationale=rationale, error=error, tokens=tokens,
            ndcg5=ndcg_at_k(pred, gt_upper, k=5) if pred else 0.0,
            kendall_tau=kendall_tau(pred, gt_upper) if pred else 0.0,
            precision5=precision_at_k(pred, gt_upper, k=5) if pred else 0.0,
            hit5=hit_at_k(pred, gt_upper, k=5) if pred else 0.0,
            mrr=mrr(pred, gt_upper) if pred else 0.0,
        ))

    ok_seeds = [s for s in per_seed if not s["error"]]
    n_ok = len(ok_seeds)

    def _mean(key: str) -> float:
        return sum(s[key] for s in ok_seeds) / n_ok if ok_seeds else 0.0

    # 题级代表值 = 成功 seed 的均值；全失败则 0
    ppr_named = len({g.upper() for g, _s in ppr_cands if g.upper() in valid})
    return dict(
        qid=qid, pool_size=len(pool), gt=gt_upper, ppr_named=ppr_named,
        pred=(ok_seeds[0]["pred"] if ok_seeds else None),   # 首个成功 seed 供逐题查验
        ndcg5=_mean("ndcg5"), kendall_tau=_mean("kendall_tau"),
        precision5=_mean("precision5"), hit5=_mean("hit5"), mrr=_mean("mrr"),
        ppr=ppr_metrics,                                    # None = 图谱零提名，基线不可测
        seeds=[s["seed"] for s in per_seed], n_seed_ok=n_ok, per_seed=per_seed,
        rationale=(ok_seeds[0]["rationale"] if ok_seeds else ""),
        tokens={k: sum(s["tokens"][k] for s in per_seed) for k in ("prompt", "completion", "total", "calls")},
        error="" if n_ok else (per_seed[0]["error"] or "all_seeds_failed"),
    )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="不调 LLM，验证逻辑")
    ap.add_argument("--system", default="pipeline", help="系统名（用于输出命名）")
    ap.add_argument("--hint", choices=("ppr_order", "none"), default="ppr_order",
                    help="候选池呈现顺序：ppr_order=图谱先给排序(LLM重排) / none=乱序(LLM裸排)")
    ap.add_argument("--seeds", default="42,7,13",
                    help="逗号分隔的 shuffle 种子（ppr_order 下仅作备查，none 下决定乱序）")
    ap.add_argument("--concurrency", type=int, default=2, help="并发 LLM 调用数（防端点被打爆超时）")
    args = ap.parse_args()

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]

    pack_data = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    packs = pack_data["packs"]
    think = "ON" if os.environ.get("SLC_TOPK_THINKING", "0") == "1" else "OFF"
    print(f"topk 评测 harness | system={args.system} | hint={args.hint} | seeds={seeds} | "
          f"thinking={think} | 题数={len(packs)} | mode={'DRY_RUN' if args.dry_run else 'LIVE_LLM'}")
    print(f"{'='*90}")

    if args.dry_run:
        client = model = None
    else:
        cfg = llm_settings()
        if not cfg["base_url"] or not cfg["api_key"]:
            raise SystemExit("config2.yaml 未找到 llm.base_url/api_key")
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                             timeout=float(os.environ.get("SLC_TOPK_LLM_TIMEOUT", "300")) + 20,
                             max_retries=0)
        model = cfg["model"]
        print(f"端点: {cfg['base_url'][:40]}...  model={model}")

    results = []
    t0 = time.time()
    # 限并发跑（信号量），避免一次性全打爆端点导致整批 TimeoutError；保持题序
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def _guarded(pack):
        async with sem:
            return await eval_one(client, model, pack, seeds=seeds, dry_run=args.dry_run,
                                  hint=args.hint)

    raw_results = await asyncio.gather(*[_guarded(pack) for pack in packs])

    for i, res in enumerate(raw_results):
        results.append(res)
        qid_short = res["qid"][-10:]
        status = "ERR" if res["error"] else "OK"
        ppr_s = f"{res['ppr']['ndcg5']:.3f}" if res["ppr"] else " N/A "
        line = (f"[{i+1:2d}/{len(packs)}] {status:3s} {qid_short:<12} "
                f"NDCG={res['ndcg5']:.3f}  τ={res['kendall_tau']:+.2f}  "
                f"PR@5={res['precision5']:.1f}  hit@5={res['hit5']:.0f}  "
                f"| PPR_NDCG={ppr_s} named={res['ppr_named']:>2d} "
                f"seed_ok={res['n_seed_ok']}/{len(seeds)}")
        if res["error"]:
            line += f"  ERR={res['error'][:60]}"
        print(line)
        sys.stdout.flush()

    elapsed = time.time() - t0

    # 汇总
    ok = [r for r in results if not r["error"]]
    err = [r for r in results if r["error"]]
    ppr_ok = [r for r in ok if r["ppr"]]           # 图谱有提名、基线可测的题

    print(f"\n{'='*90}")
    print(f"汇总 (system={args.system}, hint={args.hint}, n={len(results)}, "
          f"ok={len(ok)}, err={len(err)}, 耗时 {elapsed:.0f}s)")
    if ok:
        print(f"  NDCG@5    mean={sum(r['ndcg5'] for r in ok)/len(ok):.3f}  "
              f"median={sorted(r['ndcg5'] for r in ok)[len(ok)//2]:.3f}")
        print(f"  Kendall τ mean={sum(r['kendall_tau'] for r in ok)/len(ok):+.3f}")
        print(f"  PR@5      mean={sum(r['precision5'] for r in ok)/len(ok):.2f}  "
              f"hit@5={sum(r['hit5'] for r in ok)/len(ok):.2f}  "
              f"MRR={sum(r['mrr'] for r in ok)/len(ok):.3f}")
        total_tok = sum(r["tokens"].get("total", 0) for r in results)
        print(f"  token: total={total_tok}")
        # 同题集上 LLM vs 纯 PPR 图谱基线的差（只算基线可测的题，避免假 1.000 灌水位）
        if ppr_ok:
            ppr_mean = sum(r["ppr"]["ndcg5"] for r in ppr_ok) / len(ppr_ok)
            llm_mean = sum(r["ndcg5"] for r in ppr_ok) / len(ppr_ok)
            print(f"  纯PPR图谱基线 NDCG@5={ppr_mean:.3f} (n={len(ppr_ok)} 可测题)  →  "
                  f"LLM Δ={llm_mean-ppr_mean:+.3f}")
        else:
            print("  纯PPR图谱基线：本批无可测题（图谱对候选池零提名）")

    # 写结果
    out_path = ROOT / "outputs" / "traces" / f"topk_eval_{args.system}.json"
    json.dump(dict(system=args.system, hint=args.hint, seeds=seeds,
                   thinking=think, dry_run=args.dry_run,
                   n=len(results), n_ok=len(ok), n_err=len(err),
                   elapsed_s=elapsed, results=results),
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out_path}")


if __name__ == "__main__":
    asyncio.run(main())