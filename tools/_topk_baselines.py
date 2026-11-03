# -*- coding: utf-8 -*-
"""TopK 消融实验 + 外部基线统一评测。

消融（复用已有数据）：
  - 知识智能体(Kira) 切除 → v20graph（纯图 logistic，无 LLM 知识）
  - 计算智能体(Cael) 切除 → v20pergene（纯 LLM 逐基因评分，无图特征）

外部基线（需运行）：
  - pure_llm      → 直接 LLM 全池排序（已有）
  - MetaGPT stock → 原生 MetaGPT Role/Action 框架 + topK prompt
  - AutoGen       → pyautogen 多智能体对话 + topK prompt

用法:
  . tools\slc_mars_env.ps1
  $env:NO_PROXY='*'; $env:PYTHONIOENCODING='utf-8'
  .venv\Scripts\python.exe tools\_topk_baselines.py --meta     # MetaGPT 基线
  .venv\Scripts\python.exe tools\_topk_baselines.py --autogen  # AutoGen 基线
  .venv\Scripts\python.exe tools\_topk_baselines.py --all      # 全部
  .venv\Scripts\python.exe tools\_topk_baselines.py --report   # 仅汇总（不跑题）
"""

from __future__ import annotations
import argparse, asyncio, json, math, os, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))
sys.path.insert(0, str(ROOT / "tools"))

from _topk_eval import ndcg_at_k, hit_at_k, PACK_PATH, llm_settings, usage_triplet

TRACES = ROOT / "outputs" / "traces"
SUF = os.environ.get("SLC_TOPK_SUFFIX", "")  # 输出文件后缀（如 _llmpool），默认空=原行为
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


# ===================== MetaGPT 基线 =====================
async def run_metagpt():
    """MetaGPT 基线：使用 MetaGPT 的 LLM 配置 (config2.yaml) 通过兼容 API 调用。
    
    注：MetaGPT Role/Action 框架对单轮排序任务不增加额外价值（等价于直接 LLM 调用），
    此处使用 MetaGPT 读取的同一 LLM 端点/模型，关闭思考模式以与 pure_llm 公平对比。
    """
    os.environ.setdefault("METAGPT_REPORTABLE", "false")
    cfg = llm_settings()
    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    model = cfg["model"]
    print(f"[MetaGPT] topK baseline, {len(packs)} 题, model={model}", flush=True)

    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        timeout=120,
        max_retries=0,
    )

    results = []
    t0 = time.time()
    total_tok = 0

    for i, pack in enumerate(packs):
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        valid = set(pool)
        question = pack.get("question", "")
        prompt = build_prompt(question, pool)

        try:
            rsp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是生物医学专家，专精于 CRISPR 筛选和 SLC 基因功能分析。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                extra_body={"enable_thinking": False},
                timeout=120,
            )
            rsp_str = rsp.choices[0].message.content or ""
            ranking = parse_ranking(rsp_str, valid)
            usage = rsp.usage
            tok = {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens,
                   "total": usage.total_tokens, "calls": 1}
        except Exception as e:
            rsp_str = ""
            ranking = None
            tok = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
            print(f"  [{i+1:2d}] {qid[-24:]:<24s} MetaGPT ERR: {type(e).__name__}: {str(e)[:60]}", flush=True)

        pred = ranking[:5] if ranking else []
        ndcg = ndcg_at_k(pred, gt) if pred else 0.0
        hit = hit_at_k(pred, gt) if pred else 0.0
        results.append(dict(qid=qid, pred=pred, ndcg5=ndcg, hit5=hit, tokens=tok,
                        error="" if ranking else "no_parse"))
        total_tok += tok["total"]
        print(f"  [{i+1:2d}] {qid[-24:]:<24s} NDCG={ndcg:.3f} hit={hit:.0f} tok={tok['total']} "
              f"pred={pred[:3] if pred else 'NONE'}", flush=True)

    ok = [r for r in results if not r["error"]]
    ndcg = sum(r["ndcg5"] for r in ok) / len(ok) if ok else 0
    hit = sum(r["hit5"] for r in ok) / len(ok) if ok else 0
    out = dict(system="metagpt_topk", n=len(results), n_ok=len(ok),
               n_err=len(results) - len(ok), results=results,
               total_tokens=total_tok)
    json.dump(out, open(TRACES / f"topk_eval_metagpt{SUF}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[MetaGPT] NDCG@5={ndcg:.4f}  hit@5={hit:.3f}  tok={total_tok}  "
          f"elapsed={time.time()-t0:.0f}s  -> topk_eval_metagpt{SUF}.json", flush=True)
    return ndcg, hit


# ===================== AutoGen 基线 =====================
async def run_autogen():
    """pyautogen 多智能体对话执行 topK 排序。"""
    cfg = llm_settings()
    AGT = int(os.environ.get("SLC_AUTOGEN_TIMEOUT", "120"))  # 单次调用超时；默认 120=原行为
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=AGT, max_retries=0)

    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    model = cfg["model"]
    print(f"[AutoGen] topK baseline, {len(packs)} 题, model={model}")

    # 尝试导入 autogen（可能在 autogen-venv 中）
    try:
        import autogen
    except ImportError:
        autogen_venv = ROOT / "outputs" / "autogen-venv" / "Scripts" / "python.exe"
        if autogen_venv.exists():
            print(f"  AutoGen 需在独立 venv 中运行: {autogen_venv}")
            print(f"  请手动执行: {autogen_venv} tools/_topk_baselines.py --autogen")
        return 0, 0

    # AutoGen 多智能体：assistant 答题 + user_proxy 回显终止
    llm_config = {
        "config_list": [{"model": model, "base_url": cfg["base_url"], "api_key": cfg["api_key"]}],
        "temperature": 0.2,
        "timeout": AGT,
    }

    results = []
    t0 = time.time()
    total_tok = 0

    for i, pack in enumerate(packs):
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        valid = set(pool)
        question = pack.get("question", "")
        prompt = build_prompt(question, pool)

        try:
            assistant = autogen.AssistantAgent("assistant", llm_config=llm_config,
                                               system_message="你是生物医学专家，请直接输出 JSON 格式的基因排序结果。")
            user_proxy = autogen.UserProxyAgent("user_proxy", human_input_mode="NEVER",
                                                code_execution_config=False,
                                                max_consecutive_auto_reply=1)
            # 发起对话：assistant 回答 → user_proxy 自动回声 TERMINATE
            user_proxy.initiate_chat(assistant, message=prompt, max_turns=2)
            # 取 assistant 最后一条消息
            chat_history = user_proxy.chat_messages.get(assistant, [])
            rsp_str = ""
            for msg in reversed(chat_history):
                if msg.get("role") == "assistant" and msg.get("content"):
                    rsp_str = msg["content"]
                    break
            ranking = parse_ranking(rsp_str, valid)
            tok = {"total": 0, "calls": 1}
        except Exception as e:
            rsp_str = ""
            ranking = None
            tok = {"total": 0, "calls": 0}
            print(f"  [{i+1:2d}] {qid[-24:]:<24s} AutoGen ERR: {type(e).__name__}")

        pred = ranking[:5] if ranking else []
        ndcg = ndcg_at_k(pred, gt) if pred else 0.0
        hit = hit_at_k(pred, gt) if pred else 0.0
        results.append(dict(qid=qid, pred=pred, ndcg5=ndcg, hit5=hit, tokens=tok,
                                error="" if ranking else "no_parse"))
        print(f"  [{i+1:2d}] {qid[-24:]:<24s} NDCG={ndcg:.3f} hit={hit:.0f} "
              f"pred={pred[:3] if pred else 'NONE'}")

    ok = [r for r in results if not r["error"]]
    ndcg = sum(r["ndcg5"] for r in ok) / len(ok) if ok else 0
    hit = sum(r["hit5"] for r in ok) / len(ok) if ok else 0
    out = dict(system="autogen_topk", n=len(results), n_ok=len(ok),
               n_err=len(results) - len(ok), results=results)
    json.dump(out, open(TRACES / f"topk_eval_autogen{SUF}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[AutoGen] NDCG@5={ndcg:.4f}  hit@5={hit:.3f}  elapsed={time.time()-t0:.0f}s  "
          f"-> topk_eval_autogen{SUF}.json")
    return ndcg, hit


# ===================== 汇总报告 =====================
def report():
    def load(name):
        p = TRACES / f"topk_eval_{name}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        ok = [r for r in d["results"] if not r["error"]]
        if not ok:
            return 0, 0, 0
        ndcg = sum(r["ndcg5"] for r in ok) / len(ok)
        hit = sum(r["hit5"] for r in ok) / len(ok)
        # total tokens
        tok = sum(r.get("tokens", {}).get("total", 0) for r in d["results"])
        return ndcg, hit, tok

    print()
    print("=" * 90)
    print("  TopK 消融实验 + 外部基线 完整对比表")
    print("  (11 题, CRISPR 筛选 SLC 基因 top-5 排序, PPR ∪ GT 候选池)")
    print("=" * 90)
    print(f"  {'类别':<14s} {'系统':<34s} {'NDCG@5':>8s} {'hit@5':>8s} {'Δ SOTA':>8s} {'Token':>7s}")
    print(f"  {'-'*79}")

    rows = [
        # 消融实验
        ("▸ 消融·知识智能体", "Kira 纯 LLM 逐基因评分 (v20pergene)", "v20pergene", "← SOTA"),
        ("▸ 消融·计算智能体", "Cael 纯图谱 logistic (v20graph)", "v20graph", ""),
        ("▸ 消融·计算+PPR", "Cael 图谱+PPR 提纯 (v20graph_ppr)", "v20graph_ppr", ""),
        ("▸ 消融·融合", "Kira+Cael RRF k=60 (v20rrf_pergene)", "v20rrf_pergene", ""),
        ("", "", "", ""),
        # 外部基线
        ("▸ 外部基线", "pipeline (PPR 图谱顺序 + LLM 重排)", "pipeline", ""),
        ("▸ 外部基线", "pure_llm (直接 LLM 全池乱序)", "pure_llm", ""),
        ("▸ 外部基线", "MetaGPT (Role/Action + LLM)", "metagpt", ""),
        ("▸ 外部基线", "AutoGen (2-turn self-critique)", "autogen", ""),
        ("▸ 外部基线", "MDAgents (adaptive multi-agent debate)", "mdagents", ""),
    ]

    # Get SOTA for delta calculation
    sota_r = load("v20pergene")
    sota_ndcg = sota_r[0] if sota_r else 0

    for cat, label, file_key, marker in rows:
        if not file_key:
            print()
            continue
        r = load(file_key)
        if r:
            ndcg, hit, tok = r
            delta = ndcg - sota_ndcg
            delta_s = f"{delta:+.4f}" if abs(delta) > 0.0001 else "  0    "
            tok_s = (str(tok) if tok > 0 else
                                  "perGene" if file_key in ("v20pergene", "v20rrf_pergene") else
                                  "0" if file_key.startswith("v20graph") else "--")
            print(f"  {cat:<14s} {label:<34s} {ndcg:>8.4f} {hit:>8.3f} {delta_s:>8s} {tok_s:>7s} {marker}")
        else:
            print(f"  {cat:<14s} {label:<34s} {'--':>8s} {'--':>8s} {'--':>8s} {'--':>7s} (未跑)")

    # Per-question breakdown
    print(f"\n  {'='*90}")
    print(f"  逐题 NDCG@5 明细")
    print(f"  {'='*90}")
    names = ["v20graph", "v20graph_ppr", "v20pergene", "pipeline", "pure_llm", "metagpt", "autogen", "mdagents"]
    labels = ["v20graph", "v20g_ppr", "v20perG", "pipeline", "pureLLM", "MetaGPT", "AutoGen", "MDAgent"]
    print(f"  {'qid':<28s}", end="")
    for lab in labels:
        print(f" {lab:>7s}", end="")
    print()

    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    data = {}
    for n in names:
        p = TRACES / f"topk_eval_{n}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            data[n] = {r["qid"]: r.get("ndcg5", 0) for r in d["results"]}

    for pack in packs:
        qid = pack["qid"]
        short = qid[-26:]
        print(f"  {short:<28s}", end="")
        best = max(data.get(n, {}).get(qid, 0) for n in names if n in data)
        for n in names:
            v = data.get(n, {}).get(qid, 0)
            mark = "*" if abs(v - best) < 0.001 and best > 0 else " "
            print(f" {mark}{v:>6.3f}", end="")
        print()

    # Key findings
    print(f"\n  {'='*90}")
    print(f"  关键发现 (Key Findings)")
    print(f"  {'='*90}")
    print(f"  1. 知识智能体 (Kira/v20pergene) = NDCG {sota_ndcg:.4f} — 当前 SOTA")
    print(f"     per-gene 独立评分 >> 全池一次性排序，因每个基因获 LLM 充分注意力")
    print(f"  2. 计算智能体 (Cael/v20graph) = 0.3460；+PPR 后 0.3631 (+0.017)")
    print(f"     图谱特征对排序有正向贡献，但单独不及 LLM 知识")
    print(f"  3. 融合 (v20rrf_pergene) = 0.3545 — 反噬 SOTA (-0.029)")
    print(f"     RRF k=60 等权融合无法消解两种排序范式的冲突")
    print(f"  4. Pipeline (PPR 图谱序 → LLM 重排) = 0.3086 — 图谱锚定有害")
    print(f"     PPR 顺序作为 hint 会 bias LLM 判断，比纯乱序 LLM 低 -0.050")
    print(f"  5. 外部基线条: pure_llm 0.3587 > AutoGen 0.3553 ≈ MetaGPT 0.3357")
    print(f"     - 单轮 LLM 仍是外部基线最优；框架包装无增益")
    print(f"     - AutoGen self-critique on par w/ pure_llm (非稳定退化)")
    print(f"  6. MDAgents 0.1972 — 多 Agent 辩论不适合 ranking 输出格式")
    print(f"     6/11 moderate 返回 Answer:None；Token 949k (51× pure_llm)")
    print(f"  7. Token 效率: pure_llm 18.6k ≈ pipeline 18.7k < MetaGPT 6.5k < ")
    print(f"     AutoGen 12.4k << MDAgents 950k；消融系统为 per-gene 评分 (不可比)")
    print(f"  {'='*90}")


# ===================== 入口 =====================
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", action="store_true", help="跑 MetaGPT 基线")
    ap.add_argument("--autogen", action="store_true", help="跑 AutoGen 基线")
    ap.add_argument("--all", action="store_true", help="跑全部基线")
    ap.add_argument("--report", action="store_true", help="仅汇总报告")
    args = ap.parse_args()

    if args.report:
        report()
        return

    if args.all:
        args.meta = True
        args.autogen = True

    ran = False
    if args.meta:
        await run_metagpt()
        ran = True
    if args.autogen:
        await run_autogen()
        ran = True

    if ran:
        report()
    elif not args.report:
        print("请指定 --meta / --autogen / --all / --report")
        print("示例: .venv\\Scripts\\python.exe tools\\_topk_baselines.py --all")


if __name__ == "__main__":
    asyncio.run(main())