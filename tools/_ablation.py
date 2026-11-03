# -*- coding: utf-8 -*-
"""消融实验：复用 v20 供给（kg/ck），逐题屏蔽单智能体后重判，大幅减少实验时间。

消融项：
  ablate_knowledge  — 知识智能体（Kira/kg）输出清空，保留计算智能体
  ablate_compute    — 计算智能体（Cael/ck）输出清空，保留知识智能体

实现策略（复用最大化）：
  1) 从 v20 tracestore 直接取 supply.kg / supply.ck（JSON → pydantic）
  2) 按消融模式将该智能体的输出换为空占位符（ToolStatus.UNAVAILABLE）
  3) 只跑 Vega 决策（heuristic + auto），不重跑 supply（省 Neo4j/DESeq2/PPR 等工具调用）
  4) 与 v20 champion 出对比表

注意：
  - 消融中不包含 prior 智能体投票（tracestore 未存 prior 结果）。
    对比基线 = v20 冠军（含 prior），消融臂 = 无 prior + 屏蔽单智能体。
    prior 是正交机制，对消融归因的污染有限。
  - heuristic 决策确定性、极快；auto 决策需 LLM 调用（每题 ~30s）。
  - n=100 / seed=5 / pairwise+membership，与 v20 冠军同题库。

用法：
  python tools/_ablation.py --mode ablate_knowledge   # 消融知识智能体
  python tools/_ablation.py --mode ablate_compute      # 消融计算智能体
  python tools/_ablation.py --mode both                # 两个都跑
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knowledge_agents.answer_pipeline import AnswerTeam, _score, sample_questions
from knowledge_agents.compare_eval import load_store, SUPPORTED
from knowledge_agents.schema import (
    BackgroundKnowledge,
    ComputedKnowledge,
    ToolStatus,
)

# ---- 消融专用 trace_version（与 v20 缓存隔离） ----
TRACE_ABLATE_KG = "21"   # 消融知识智能体
TRACE_ABLATE_CK = "22"   # 消融计算智能体
CHANCE = 0.5
N = 100
SEED = 5
BUCKETS = ["pairwise", "membership"]
TRACES_DIR = ROOT / "outputs" / "traces"

# ---- 空占位符 ----
def empty_kg() -> dict:
    """知识智能体已禁用的空占位 JSON。"""
    return BackgroundKnowledge(
        question="[ablation]",
        status=ToolStatus.UNAVAILABLE,
        summary="[消融] 知识智能体 Kira 已禁用",
    ).model_dump()

def empty_ck() -> dict:
    """计算智能体已禁用的空占位 JSON。"""
    return ComputedKnowledge(
        question="[ablation]",
        status=ToolStatus.UNAVAILABLE,
        summary="[消融] 计算智能体 Cael 已禁用",
    ).model_dump()


class AblationRunner:
    def __init__(self, mode: str, n: int = N, seed: int = SEED,
                 buckets: list[str] | None = None):
        self.mode = mode
        self.n = n
        self.seed = seed
        self.buckets = buckets or BUCKETS
        self.trace_version = TRACE_ABLATE_KG if mode == "ablate_knowledge" else TRACE_ABLATE_CK
        self.team = AnswerTeam()
        self.results: list[dict] = []

    async def run(self) -> dict:
        # 1) 抽样（与 v20 冠军同题库）
        sample = sample_questions(self.n, self.buckets, self.seed)
        gt_index = {r["qid"]: r.get("_gt", {}) for r in sample}
        print(f"[消融] mode={self.mode}  trace={self.trace_version}  "
              f"抽样={len(sample)} 题 seed={self.seed} 桶={self.buckets}")

        # 2) 加载 v20 tracestore
        store = load_store()
        v20_hits = sum(1 for q in sample if q["qid"] in store)
        missing = len(sample) - v20_hits
        print(f"  v20 tracestore 命中: {v20_hits}/{len(sample)}"
              + (f"  ⚠ 缺失 {missing} 题（v500 扩池未覆盖？）" if missing else ""))

        # 3) 逐题消融
        t0 = datetime.now(timezone.utc)
        n_ok = n_err = 0
        for i, rec in enumerate(sample, 1):
            qid = rec["qid"]
            v20 = store.get(qid)
            if not v20:
                print(f"[{i:3d}/{len(sample)}] SKIP {qid[-12:]} — v20 tracestore 缺失")
                continue

            try:
                entry = await self._ablate_one(v20, rec)
                self.results.append(entry)
                n_ok += 1
                # 打印一题概要
                ha = entry["systems"]["pipeline_heuristic"]
                aa = entry["systems"]["pipeline_auto"]
                bkt = rec.get("task_bucket") or entry.get("bucket", "")
                print(f"[{i:3d}/{len(sample)}] {bkt:<9} {qid[-16:]:<16}  "
                      f"heur={ha['outcome']:<7} auto={aa['outcome']:<7}")
                sys.stdout.flush()
            except Exception as e:
                n_err += 1
                print(f"[{i:3d}/{len(sample)}] ERR {qid[-12:]} {type(e).__name__}: {str(e)[:60]}")
                sys.stdout.flush()

        elapsed_s = (datetime.now(timezone.utc) - t0).total_seconds()
        print(f"\n完成: {n_ok} 题, 错误 {n_err}, 耗时 {elapsed_s:.0f}s")

        # 4) 汇总表（vs v20 冠军基线的对比）
        summary = self._summarize(store, gt_index)
        return summary

    async def _ablate_one(self, v20_entry: dict, rec: dict) -> dict:
        """对一道题：清空 kg 或 ck → 重判 → 记录。"""
        qid = rec["qid"]
        bucket = v20_entry.get("bucket") or rec.get("task_bucket", "")
        genes = v20_entry.get("effective_genes") or []
        question = v20_entry.get("question") or rec.get("question", "")
        constraints = v20_entry.get("constraints") or rec.get("constraints", {})

        # 重建 kg/ck（JSON → pydantic）
        kg_dict = v20_entry["supply"]["kg"]
        ck_dict = v20_entry["supply"]["ck"]

        if self.mode == "ablate_knowledge":
            kg_use = BackgroundKnowledge(**empty_kg())
            ck_use = ComputedKnowledge(**ck_dict)
        elif self.mode == "ablate_compute":
            kg_use = BackgroundKnowledge(**kg_dict)
            ck_use = ComputedKnowledge(**empty_ck())
        else:
            raise ValueError(f"未知消融模式: {self.mode}")

        # 构建 supplied（不含 prior 投票 — tracestore 未存）
        supplied = {
            "kg": kg_use, "ck": ck_use,
            "genes": genes, "gene": genes[0] if genes else None,
            "bucket": bucket,
            "task_type": rec.get("task_type") or v20_entry.get("task_type", ""),
            "question": question,
            "constraints": constraints,
            "prior": None, "prior2": None,
        }

        entry = {
            "qid": qid, "bucket": bucket,
            "task_type": rec.get("task_type") or v20_entry.get("task_type", ""),
            "question": question, "constraints": constraints,
            "ground_truth": rec.get("_gt", {}), "effective_genes": genes,
            "supply": {"kg": kg_use.model_dump(), "ck": ck_use.model_dump()},
            "systems": {},
        }

        for mode_name, sys_key in (("heuristic", "pipeline_heuristic"),
                                   ("auto", "pipeline_auto")):
            ans = await self.team.decide(supplied, mode=mode_name)
            dump = json.loads(ans.model_dump_json())
            entry["systems"][sys_key] = {
                "answer": dump,
                "outcome": _score(ans, rec.get("_gt", {}), bucket),
            }

        entry["provenance"] = {
            "dataset": "clean",
            "trace_version": self.trace_version,
            "ablation": self.mode,
            "source": "v20_cache_reuse",
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        return entry

    def _summarize(self, store: dict, gt_index: dict) -> dict:
        """汇总消融臂 + v20 champion baseline 对比。"""
        # v20 baseline 统计（从 tracestore 同 sample 题目取）
        v20_results = [store[q] for q in gt_index if q in store]
        baseline = self._stats(v20_results, "pipeline_auto")
        ablation = self._stats(self.results, "pipeline_auto")

        # 打印对比表
        print("\n" + "=" * 72)
        print(f"===== 消融: {self.mode}  (trace={self.trace_version}, n={len(self.results)}@{self.seed})")
        print(f"{'指标':<16} {'v20 冠军':>10} {'消融臂':>10} {'Δ':>10}")
        print("-" * 50)

        label = {"ablate_knowledge": "消融·知识(Kira)", "ablate_compute": "消融·计算(Cael)"}[self.mode]
        rows = [
            ("准确率(全体)", "acc_overall"),
            ("覆盖率", "coverage"),
            ("准确率(已答)", "acc_answered"),
            ("净胜机会", "net_vs_chance"),
        ]
        for title, key in rows:
            bv = baseline.get(key, 0)
            av = ablation.get(key, 0)
            d = av - bv
            print(f"{title:<16} {bv:>10.3f} {av:>10.3f} {d:>+10.3f}")

        print("-" * 50)
        for b_lbl, b_rows in [("v20冠军(含prior)", v20_results), (label, self.results)]:
            c = sum(1 for r in b_rows if r["systems"].get("pipeline_auto", {}).get("outcome") == "correct")
            w = sum(1 for r in b_rows if r["systems"].get("pipeline_auto", {}).get("outcome") == "wrong")
            a = sum(1 for r in b_rows if r["systems"].get("pipeline_auto", {}).get("outcome") == "abstain")
            tot = c + w + a
            if tot:
                print(f"  {b_lbl:<16}: N={tot:3d}  对={c:3d}  错={w:3d}  弃={a:3d}  "
                      f"准确率({c/(c+w) if c+w else 0:.1%} | {c/tot:.1%})")

        # 写 JSON
        out_path = TRACES_DIR / f"ablation_{self.mode}_n{len(self.results)}_seed{self.seed}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ablation": self.mode, "trace_version": self.trace_version,
            "n": len(self.results), "seed": self.seed, "buckets": self.buckets,
            "n_valid": len(self.results),
            "summary": ablation,
            "baseline_v20": baseline,
            "results": [{k: v for k, v in r.items() if k != "supply"} for r in self.results],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果: {out_path}")

        return ablation

    @staticmethod
    def _stats(results: list[dict], sys_key: str) -> dict:
        outcomes = [r["systems"].get(sys_key, {}).get("outcome", "abstain") for r in results
                    if r.get("bucket") in SUPPORTED]
        c = outcomes.count("correct")
        w = outcomes.count("wrong")
        a = outcomes.count("abstain")
        n = c + w + a
        ans = c + w
        return dict(n=n, correct=c, wrong=w, abstain=a,
                    coverage=ans / n if n else 0,
                    acc_answered=c / ans if ans else 0,
                    acc_overall=c / n if n else 0,
                    net_vs_chance=(c / n - CHANCE) if n else 0)


# ---- 入口 ----
async def main():
    ap = argparse.ArgumentParser(description="v20 消融实验")
    ap.add_argument("--mode", choices=["ablate_knowledge", "ablate_compute", "both"],
                    default="both", help="消融哪路智能体")
    ap.add_argument("--n", type=int, default=N, help=f"题目数 (默认 {N})")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true", help="不跑 LLM，仅验证逻辑")
    args = ap.parse_args()

    if args.dry_run:
        os.environ["SLC_DECISION_MODE"] = "heuristic"

    modes = [args.mode] if args.mode != "both" else ["ablate_knowledge", "ablate_compute"]
    for m in modes:
        runner = AblationRunner(mode=m, n=args.n, seed=args.seed)
        await runner.run()
        print("\n" + "=" * 72)


if __name__ == "__main__":
    asyncio.run(main())