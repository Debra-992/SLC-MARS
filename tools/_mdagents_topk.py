# -*- coding: utf-8 -*-
"""MDAgents topK baseline — 自适应多智能体协作排序。
复刻 knowledge_agents/baselines/run_mdagents.py 的补丁/路由逻辑，
题面改为 topK 排序 prompt，解析 ranking JSON 计算 NDCG@5/hit@5。

用法:
  . tools\slc_mars_env.ps1
  $env:NO_PROXY='*'; $env:PYTHONIOENCODING='utf-8'
  .\outputs\mdagents-venv\Scripts\python.exe tools\_mdagents_topk.py [--limit N]
"""
from __future__ import annotations

import argparse, concurrent.futures, json, os, re, sys, time, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MDA = str(ROOT / "outputs" / "MDAgents")
TOOLS = str(ROOT / "tools")
REPO_STR = str(ROOT)
sys.path.insert(0, REPO_STR)
sys.path.insert(0, TOOLS)          # 让 from _topk_eval import PACK_PATH 可用
sys.path.insert(0, MDA)
# MDAgents 源码往控制台 cprint 医生 emoji —— Windows GBK 标准输出会 UnicodeEncodeError
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---- 端点/模型 patch（须在 import utils 之前）----
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))
from common import llm_settings, usage_triplet  # noqa: E402

CFG = llm_settings()

PER_Q_TIMEOUT = float(os.environ.get("SLC_MDAGENTS_TIMEOUT", "1800"))


class TokenSink:
    """线程安全的 token 累加器。"""
    def __init__(self):
        self._lock = __import__("threading").Lock()
        self.prompt = self.completion = self.total = self.calls = 0
    def reset(self):
        with self._lock:
            self.prompt = self.completion = self.total = self.calls = 0
    def add(self, p, c, t):
        with self._lock:
            self.prompt += p; self.completion += c; self.total += t; self.calls += 1
    def snapshot(self):
        with self._lock:
            return {"prompt": self.prompt, "completion": self.completion,
                    "total": self.total, "calls": self.calls}


_sink = TokenSink()


def _patch_openai() -> None:
    import openai
    real = openai.OpenAI

    class QwenClient(real):
        def __init__(self, *a, **kw):
            kw["api_key"] = CFG["api_key"]
            kw["base_url"] = CFG["base_url"]
            kw.setdefault("max_retries", 1)
            super().__init__(*a, **kw)
            real_create = self.chat.completions.create

            def create(*ca, **ckw):
                ckw["model"] = CFG["model"]
                ckw.setdefault("timeout", 300)
                # qwen3.8-flash 默认思考开 → 单次调用 270s；显式关闭
                ckw.setdefault("extra_body", {"enable_thinking": False})
                rsp = real_create(*ca, **ckw)
                try:
                    if not ckw.get("stream"):
                        p, c, t = usage_triplet(rsp)
                        _sink.add(p, c, t)
                except Exception:
                    pass
                return rsp

            self.chat.completions.create = create

    openai.OpenAI = QwenClient


def _stub_gemini() -> None:
    if "google.generativeai" in sys.modules:
        return
    g = types.ModuleType("google")
    ga = types.ModuleType("google.generativeai")
    ga.configure = lambda **kw: None
    ga.GenerativeModel = object
    g.generativeai = ga
    sys.modules["google"] = g
    sys.modules["google.generativeai"] = ga


os.environ.setdefault("openai_api_key", CFG["api_key"])
_patch_openai()
_stub_gemini()

from utils import (  # noqa: E402  (MDAgents 官方逻辑)
    determine_difficulty, process_advanced_query, process_basic_query, process_intermediate_query,
)
import utils as _mda  # noqa: E402

# ---- 防御性补丁（同 run_mdagents.py）----
_real_parse_hierarchy = _mda.parse_hierarchy

def _safe_parse_hierarchy(info, emojis):
    try:
        return _real_parse_hierarchy(info, emojis)
    except (IndexError, AttributeError):
        from pptree import Node
        moderator = Node('moderator')
        agents = [moderator]
        for k, (expert, _h) in enumerate(info):
            name = "{} ({})".format(expert.split("-")[0].strip(), emojis[k % len(emojis)])
            agents.append(Node(name, moderator))
        return agents

_mda.parse_hierarchy = _safe_parse_hierarchy

_RealGroup = _mda.Group

class _RobustGroup(_RealGroup):
    def interact(self, comm_type, message=None, img_path=None):
        try:
            return super().interact(comm_type, message=message, img_path=img_path)
        except IndexError:
            solo = _mda.Agent("You are a helpful medical agent.", role='medical expert',
                              model_info=_Args.model)
            solo.chat("You are a helpful medical agent.")
            return solo.chat(self.question)

_mda.Group = _RobustGroup


class _Args:
    dataset = "custom"
    difficulty = "adaptive"
    model = "gpt-4o-mini"


def _decision_text(final) -> str:
    if isinstance(final, dict):
        return "\n".join(_decision_text(v) for v in final.values())
    if isinstance(final, (list, tuple)):
        return "\n".join(_decision_text(v) for v in final)
    return str(final or "")


# ===================== TopK prompt & parsing =====================
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


def build_topk_prompt(question: str, genes: list[str], seed: int = 42) -> str:
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


# ---- NDCG/hit ----
def ndcg_at_k(pred: list[str], gt: list[str], k: int = 5) -> float:
    p = pred[:k]
    dcg = 0.0
    for i, g in enumerate(p):
        try:
            idx = gt.index(g)
            dcg += 1.0 / __import__("math").log2(i + 2)
        except ValueError:
            pass
    idcg = sum(1.0 / __import__("math").log2(i + 2) for i in range(min(len(gt), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(pred: list[str], gt: list[str], k: int = 5) -> float:
    return 1.0 if any(g in gt for g in pred[:k]) else 0.0


# ---- 单题运行 ----
def run_one(pack: dict) -> dict:
    qid = pack["qid"]
    pool = [g.upper() for g in pack["eval_pool"]]
    gt = [g.upper() for g in pack["gt_ranking"]]
    valid = set(pool)
    question = pack.get("question", "")
    prompt = build_topk_prompt(question, pool)

    _sink.reset()
    diff = "?"
    try:
        diff = determine_difficulty(prompt, _Args.difficulty) or "basic"
        if diff == "basic":
            final = process_basic_query(prompt, [], _Args.model, _Args())
        elif diff == "intermediate":
            final = process_intermediate_query(prompt, [], _Args.model, _Args())
        else:
            final = process_advanced_query(prompt, _Args.model, _Args())
        text = _decision_text(final)
        ranking = parse_ranking(text, valid)
        tok = _sink.snapshot()
        pred = ranking[:5] if ranking else []
        return {"qid": qid, "pred": pred, "diff": diff,
                "ndcg5": ndcg_at_k(pred, gt), "hit5": hit_at_k(pred, gt),
                "tokens": tok, "error": ""}
    except Exception as e:
        import traceback
        tb = traceback.format_exc().strip().splitlines()[-3:]
        err = f"{type(e).__name__}: {e} | {'; '.join(tb)}"[:400]
        return {"qid": qid, "pred": [], "diff": diff,
                "ndcg5": 0.0, "hit5": 0.0,
                "tokens": _sink.snapshot(), "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # Load topK packs
    from _topk_eval import PACK_PATH
    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    if args.limit:
        packs = packs[:args.limit]

    TRACES = ROOT / "outputs" / "traces"
    print(f"[MDAgents] topK baseline, {len(packs)} 题, model={CFG['model']}", flush=True)
    print(f"  adaptive 路由 + 单题超时 {PER_Q_TIMEOUT:.0f}s", flush=True)

    results = []
    t0 = time.time()
    total_tok = 0
    out_path = TRACES / f"topk_eval_mdagents{os.environ.get('SLC_TOPK_SUFFIX','')}.json"
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    for i, pack in enumerate(packs):
        qid = pack["qid"]
        t1 = time.time()
        fut = pool.submit(run_one, pack)
        try:
            row = fut.result(timeout=PER_Q_TIMEOUT)
        except concurrent.futures.TimeoutError:
            row = {"qid": qid, "pred": [], "diff": "?",
                   "ndcg5": 0.0, "hit5": 0.0,
                   "tokens": _sink.snapshot(), "error": "TimeoutError"}
        elapsed = time.time() - t1
        total_tok += row["tokens"]["total"]
        results.append(row)
        print(f"  [{i+1:2d}] {qid[-24:]:<24s} diff={row['diff']:<12s} NDCG={row['ndcg5']:.3f} "
              f"hit={row['hit5']:.0f} tok={row['tokens']['total']} calls={row['tokens']['calls']} "
              f"{elapsed:.0f}s pred={row['pred'][:3] if row['pred'] else 'NONE'}", flush=True)
        # ---- 逐题增量保存（防中途崩溃丢失全部结果）----
        ok_so_far = [r for r in results if not r["error"]]
        ndcg_sofar = sum(r["ndcg5"] for r in ok_so_far) / len(ok_so_far) if ok_so_far else 0
        hit_sofar = sum(r["hit5"] for r in ok_so_far) / len(ok_so_far) if ok_so_far else 0
        partial = {"system": "mdagents_topk", "n": len(results), "n_ok": len(ok_so_far),
                   "n_err": len(results) - len(ok_so_far),
                   "ndcg5_running": round(ndcg_sofar, 4), "hit5_running": round(hit_sofar, 4),
                   "results": results}
        json.dump(partial, open(out_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    pool.shutdown(wait=False)

    ok = [r for r in results if not r["error"]]
    ndcg = sum(r["ndcg5"] for r in ok) / len(ok) if ok else 0
    hit = sum(r["hit5"] for r in ok) / len(ok) if ok else 0
    out = {"system": "mdagents_topk", "n": len(results), "n_ok": len(ok),
           "n_err": len(results) - len(ok), "results": results}
    json.dump(out, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[MDAgents] NDCG@5={ndcg:.4f}  hit@5={hit:.3f}  tok={total_tok}  "
          f"elapsed={time.time()-t0:.0f}s  -> topk_eval_mdagents.json", flush=True)


if __name__ == "__main__":
    main()