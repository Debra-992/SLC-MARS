# -*- coding: utf-8 -*-
"""公平对比：在『同一份干净题目样本』上同时跑三种系统，输出多指标对照表 + 逐题全过程存档。

三个系统（区分保存）：
  - 管线·heuristic   Kira∥Cael 接地证据 -> 纯启发式打分（可弃答）
  - 管线·auto(类比)  同一份证据，证据不足时 LLM 跨癌种类推（尽量不弃答）
  - 原生MetaGPT       stock MetaGPT Role，纯 LLM（无工具/检索/CRISPR）

公平做法：
  - 同一 (n, buckets, seed) 抽样 -> 三系统答完全相同的题。
  - 同时报 覆盖率 / 准确率(已答) / 准确率(全体) / 净胜机会基线，避免"狂弃答刷高精度"。

全过程留存 & 增量复用：
  - 每题完整过程（各智能体输出 + 中间证据 kg/ck + 推理 rationale + gene_scores）
    写入 outputs/traces/tracestore.jsonl（按 qid 去重、可断点续跑）。
  - 同时逐题写 outputs/traces/by_question/<qid>.json（人类可读）。
  - 已缓存且配置匹配的题不重算，扩大 N 时只跑新增题。

用法：
  python -m knowledge_agents.compare_eval --n 100 --seed 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from knowledge_agents.answer_pipeline import AnswerTeam, _score, sample_questions
from knowledge_agents.baseline_metagpt import BaselineRole, preflight_llm
from knowledge_agents.baselines.common import install_temperature_zero
from knowledge_agents.run_single_question import ROOT, extract_genes
from knowledge_agents.schema import parse_knowledge

SUPPORTED = ("pairwise", "membership")
SYSTEMS = ("pipeline_heuristic", "pipeline_auto", "baseline")
DISPLAY = {"pipeline_heuristic": "管线·heuristic", "pipeline_auto": "管线·auto(类比)",
           "baseline": "原生MetaGPT"}
TRACE_VERSION = os.environ.get("SLC_TRACE_VERSION", "20")  # 定版（frozen）= SLC-MARS v20，全量 clean 池 n=489 已完整落盘（tracestore trace_version=20，489/489 题）。判定/证据结构变更才 +1，会让旧缓存失效。v20 = v16 基座（TRIAD_VOTE + TRIAD_CRITIQUE + PRIOR_RACE=2 + L2 并行角色异构）+ R2 membership 必需性证据重排（SLC_MEM_R2=1）；历史实验臂 v10–v18 保留在 tracestore 内、门控可复现，但均非定版配置。SLC_TRACE_VERSION 可临时覆盖跑对照臂。
CHANCE = 0.5                 # pairwise / membership 均为二选一
TRACES_DIR = ROOT / "outputs" / "traces"
STORE_PATH = TRACES_DIR / "tracestore.jsonl"
BYQ_DIR = TRACES_DIR / "by_question"
MANIFEST_PATH = TRACES_DIR / "sample_manifest.json"


# ------------------------------------------------------------------ 序列化
def _dump(model) -> dict | None:
    """pydantic 模型 -> 纯 JSON dict（枚举转字符串）。None 透传。"""
    if model is None:
        return None
    return json.loads(model.model_dump_json())


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ 逐题全过程
async def trace_one(team: AnswerTeam, rec: dict, llm_ok: bool) -> dict:
    """跑一题：Kira∥Cael 供给一次 -> heuristic / auto 决策 + baseline，返回完整过程 dict。"""
    bucket = rec.get("task_bucket", "")
    genes, _ = team._effective_genes(rec)
    supplied = await team.supply(rec)                     # 工具证据只取一次，三系统共享

    out: dict = {
        "qid": rec["qid"], "bucket": bucket, "task_type": rec.get("task_type", ""),
        "question": rec["question"], "constraints": rec.get("constraints", {}),
        "ground_truth": rec.get("_gt", {}), "effective_genes": genes,
        "supply": {"kg": _dump(supplied["kg"]), "ck": _dump(supplied["ck"])},
        "systems": {},
    }

    for mode, key in (("heuristic", "pipeline_heuristic"), ("auto", "pipeline_auto")):
        ans = await team.decide(supplied, mode=mode)
        out["systems"][key] = {"answer": _dump(ans), "outcome": _score(ans, rec["_gt"], bucket)}

    # 原生 MetaGPT baseline（纯 LLM）
    if llm_ok:
        bgenes = extract_genes(rec["question"], rec.get("constraints", {}))
        role = BaselineRole()
        role.set_inputs(bucket=bucket, candidates=bgenes, question=rec["question"],
                        task_type=rec.get("task_type", ""))
        try:
            await role.run("answer")
            bans = parse_knowledge(role.rc.memory.get()[-1].content)
            out["systems"]["baseline"] = {"answer": _dump(bans), "outcome": _score(bans, rec["_gt"], bucket)}
        except Exception as e:  # noqa BLE001
            out["systems"]["baseline"] = {"answer": None, "outcome": "abstain", "error": type(e).__name__}
    else:
        out["systems"]["baseline"] = {"answer": None, "outcome": "abstain", "error": "llm_unavailable"}

    out["provenance"] = {"dataset": os.environ.get("SLC_DATASET", "clean"),
                         "trace_version": TRACE_VERSION, "baseline_llm": llm_ok, "ts": _now()}
    return out


def _reusable(entry: dict, dataset: str, llm_ok: bool) -> bool:
    p = entry.get("provenance", {})
    if entry.get("error") or p.get("dataset") != dataset or p.get("trace_version") != TRACE_VERSION:
        return False
    if bool(p.get("baseline_llm")) != bool(llm_ok):
        return False
    systems = entry.get("systems") or {}
    if set(systems) < set(SYSTEMS):
        return False
    # 必须是真正跑出的答案（含 answer 字段），异常兜底桩只有 outcome、不可复用
    return all("answer" in systems[s] for s in SYSTEMS)


# ------------------------------------------------------------------ trace store
def load_store() -> dict[str, dict]:
    store: dict[str, dict] = {}
    if STORE_PATH.exists():
        for line in STORE_PATH.open(encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    store[r["qid"]] = r
                except json.JSONDecodeError:
                    continue
    return store


def append_store(entry: dict) -> None:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    with STORE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    BYQ_DIR.mkdir(parents=True, exist_ok=True)
    (BYQ_DIR / f"{entry['qid']}.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ 指标
def _metrics(outcomes: list[str]) -> dict:
    c = outcomes.count("correct"); w = outcomes.count("wrong"); a = outcomes.count("abstain")
    n = c + w + a; ans = c + w
    acc_overall = c / n if n else 0.0
    return dict(n=n, correct=c, wrong=w, abstain=a,
                coverage=ans / n if n else 0.0,
                acc_answered=c / ans if ans else 0.0,
                acc_overall=acc_overall, net_vs_chance=acc_overall - CHANCE)


def _fmt_row(name: str, m: dict) -> str:
    return (f"| {name:<16} | {m['n']:>3} | {m['correct']:>3} | {m['wrong']:>3} | {m['abstain']:>3} "
            f"| {m['coverage']:>6.1%} | {m['acc_answered']:>7.1%} | {m['acc_overall']:>7.1%} "
            f"| {m['net_vs_chance']:>+8.1%} |")


def _table(title: str, rows: list[tuple[str, dict]]) -> str:
    n = rows[0][1]["n"] if rows else 0
    lines = [f"\n### {title}  (n={n})",
             "| 系统 | N | 对 | 错 | 弃 | 覆盖率 | 准确率(已答) | 准确率(全体) | 净胜机会基线 |",
             "|------|----:|---:|---:|---:|------:|--------:|--------:|--------:|"]
    for name, m in rows:
        lines.append(_fmt_row(name, m))
    return "\n".join(lines)


# ------------------------------------------------------------------ 主流程
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--buckets", default="pairwise,membership")
    ap.add_argument("--no-cache", action="store_true", help="忽略缓存全部重算")
    args = ap.parse_args()
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    tz = install_temperature_zero()   # SLC_FORCE_TEMP0=1 时锁死全部 LLM 采样→temp0（R2 开关 A/B 干净归因）

    sample = sample_questions(args.n, buckets, args.seed)
    dataset = os.environ.get("SLC_DATASET", "clean")
    llm_ok = await preflight_llm()
    print("=" * 84, flush=True)
    print(f"[对比评测] dataset={dataset} 抽样={len(sample)} 题 seed={args.seed} 桶={buckets} "
          f"| baseline LLM {'可用' if llm_ok else '不可用(记abstain)'} | temp0={'强制' if tz else '关'}", flush=True)

    store = {} if args.no_cache else load_store()
    team = AnswerTeam()
    results: list[dict] = []
    n_new = n_cached = 0
    for i, rec in enumerate(sample, 1):
        entry = store.get(rec["qid"])
        if entry and _reusable(entry, dataset, llm_ok):
            n_cached += 1
            tag = "CACHE"
        else:
            try:
                entry = await trace_one(team, rec, llm_ok)
            except Exception as e:  # noqa BLE001
                entry = {"qid": rec["qid"], "bucket": rec.get("task_bucket"),
                         "systems": {s: {"outcome": "abstain"} for s in SYSTEMS},
                         "provenance": {"dataset": dataset, "trace_version": TRACE_VERSION,
                                        "baseline_llm": llm_ok, "ts": _now()},
                         "error": f"{type(e).__name__}: {str(e)[:80]}"}
            append_store(entry)
            store[rec["qid"]] = entry
            n_new += 1
            tag = "NEW"
        results.append(entry)
        oc = {s: entry.get("systems", {}).get(s, {}).get("outcome", "abstain") for s in SYSTEMS}
        print(f"[{i:>3}/{len(sample)}] {tag:<5} {rec['qid'][:42]:<42} "
              f"heur={oc['pipeline_heuristic']:<7} auto={oc['pipeline_auto']:<7} "
              f"base={oc['baseline']}", flush=True)

    # 汇总（按当前样本，不掺缓存里的无关题）
    outcomes = {s: {b: [] for b in SUPPORTED} for s in SYSTEMS}
    for entry in results:
        b = entry.get("bucket")
        if b not in SUPPORTED:
            continue
        for s in SYSTEMS:
            outcomes[s][b].append(entry.get("systems", {}).get(s, {}).get("outcome", "abstain"))

    md = ["[公平对比表] 覆盖率=已答/总 | 准确率(已答)=对/已答 | 准确率(全体)=对/总(弃答计错) "
          f"| 净胜机会基线=准确率(全体)-{CHANCE:.0%}",
          f"抽样: n={len(sample)} seed={args.seed} dataset={dataset} 桶={buckets} "
          f"(新算 {n_new} / 复用缓存 {n_cached})"]
    for b in SUPPORTED:
        md.append(_table(b, [(DISPLAY[s], _metrics(outcomes[s][b])) for s in SYSTEMS]))
    md.append(_table("总体(OVERALL)", [
        (DISPLAY[s], _metrics(outcomes[s]["pairwise"] + outcomes[s]["membership"])) for s in SYSTEMS]))
    table_text = "\n".join(md)

    print("\n" + "=" * 84, flush=True)
    print(table_text, flush=True)
    print(f"\n全过程: {STORE_PATH}  |  逐题: {BYQ_DIR}\\<qid>.json", flush=True)

    # manifest：记录本次用了哪些题，供后续扩题去重复用
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps({
        "updated_at": _now(), "dataset": dataset, "seed": args.seed, "buckets": buckets,
        "n": len(sample), "trace_version": TRACE_VERSION, "baseline_llm": llm_ok,
        "qids": [r["qid"] for r in results],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    run_table = TRACES_DIR / f"compare_table_n{len(sample)}_seed{args.seed}.md"
    run_table.write_text(table_text, encoding="utf-8")
    print(f"清单:   {MANIFEST_PATH}\n表存档: {run_table}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
