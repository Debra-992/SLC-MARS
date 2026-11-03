# -*- coding: utf-8 -*-
"""保形『校准池』生成器：只跑 管线·heuristic（纯打分、无 LLM），在大规模题目上批量产出
每题的 (置信度, 对错) —— 专门喂给 conformal_report 做阈值校准与风险-覆盖评测。

为什么单独一个 dumper：
  - 保形校准需要【足够多的已答样本】才能得到可部署的紧阈值；而 heuristic 判定是确定性、
    免 LLM 的，跑几百题也只花工具/检索时间，能把校准池做大。
  - 写到独立的 calibstore.jsonl（整文件去重），不污染 compare_eval 的三系统缓存 tracestore。
  - 校准必须是【未加保形】的原始置信度分布，因此本脚本显式清除 SLC_CONFORMAL_FILE，
    绝不自我阈值截断（否则是把阈值喂给校准 = 循环论证）。

用法：
  python -m knowledge_agents.calib_dump --n 250 --seed 5
  python -m knowledge_agents.conformal_report --traces outputs/traces/calibstore.jsonl \
      --version 8 --system pipeline_heuristic --alpha 0.10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone

from knowledge_agents.answer_pipeline import AnswerTeam, _score, sample_questions
from knowledge_agents.compare_eval import TRACE_VERSION, _dump
from knowledge_agents.run_single_question import ROOT

DATASET = os.environ.get("SLC_DATASET", "clean")
CALIB_DIR = ROOT / "outputs" / "traces"
CALIB_PATH = CALIB_DIR / "calibstore.jsonl"
SUPPORTED = ("pairwise", "membership")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_calib() -> dict[str, dict]:
    store: dict[str, dict] = {}
    if CALIB_PATH.exists():
        for line in CALIB_PATH.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                store[r["qid"]] = r
            except json.JSONDecodeError:
                continue
    return store


def write_calib(store: dict[str, dict]) -> None:
    """整文件重写：保证每个 qid 只有一行，conformal_report 直接逐行读不会重复计数。"""
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CALIB_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for qid in sorted(store):
            f.write(json.dumps(store[qid], ensure_ascii=False) + "\n")
    tmp.replace(CALIB_PATH)


def _reusable(entry: dict) -> bool:
    if entry.get("error"):
        return False
    p = entry.get("provenance", {})
    if p.get("dataset") != DATASET or p.get("trace_version") != TRACE_VERSION:
        return False
    sysobj = entry.get("systems", {}).get("pipeline_heuristic") or {}
    return "answer" in sysobj


def _features(supplied: dict) -> dict:
    """从工具产出的原始数值抽出一个【免泄漏】特征向量（只用证据、绝不碰真值）。

    目标基因 = 路由主基因 supplied['gene']（membership 为单基因，pairwise 为主基因）。
    这些特征供 ⑤ 的 logistic 校准（用 500 clean 标签学习权重）使用；判定与评测仍只用
    heuristic 打分，二者互不污染。
    """
    kg, ck = supplied["kg"], supplied["ck"]
    gene = (supplied.get("gene") or "").upper()
    cancer = str((supplied.get("constraints") or {}).get("cancer") or "").lower()

    # --- KG gene_disease_paths：把这条 SLC->Pathway->Disease 链【拆细】而不是只压成复合极性 ---
    #   复合效应 gene_disease_effect(rel_AB⊗rel_BC) 的促/抑计数（旧特征，向后兼容）
    #   + rel_AB(基因->通路)、rel_BC(通路->疾病) 两段各自的方向计数
    #   + 命中本癌种疾病的促/抑拆分
    #   + 去重中介通路数（通路多样性）
    n_paths = n_promote = n_suppress = n_cancer_hit = 0
    n_ab_promote = n_ab_suppress = n_bc_promote = n_bc_suppress = 0
    n_promote_matched = n_suppress_matched = 0
    pathways: set[str] = set()
    pathways_matched: set[str] = set()

    def _sign(v):
        u = str(v or "").upper()
        return 1 if u == "PROMOTION" else (-1 if u == "SUPPRESSION" else 0)

    for ev in (getattr(kg, "evidence", []) or []):
        if ev.operation != "gene_disease_paths":
            continue
        for row in ev.rows or []:
            if str(row.get("gene", "")).upper() != gene:
                continue
            n_paths += 1
            eff = str(row.get("gene_disease_effect", "")).upper()
            if eff == "PROMOTION":
                n_promote += 1
            elif eff == "SUPPRESSION":
                n_suppress += 1
            ab, bc = _sign(row.get("rel_AB")), _sign(row.get("rel_BC"))
            n_ab_promote += ab > 0
            n_ab_suppress += ab < 0
            n_bc_promote += bc > 0
            n_bc_suppress += bc < 0
            p = (row.get("pathway") or "").upper()
            if p:
                pathways.add(p)
            d = str(row.get("disease", "")).lower()
            matched = bool(cancer) and cancer in d
            if matched:
                n_cancer_hit += 1
                if eff == "PROMOTION":
                    n_promote_matched += 1
                elif eff == "SUPPRESSION":
                    n_suppress_matched += 1
                if p:
                    pathways_matched.add(p)

    # --- PPR 网络邻近度 ---
    ppr = deg = None
    in_graph = 0
    gc = getattr(ck, "graph_computation", None)
    if gc:
        for c in gc.candidates or []:
            if str(c.gene).upper() == gene:
                ppr = c.ppr_score
                deg = c.degree_corrected_score
                in_graph = 1 if c.in_graph else 0
                break

    # --- TIMER 表达（行数/是否可用）---
    exp = getattr(ck, "expression_by_gene", {}).get(gene) or ck.expression
    timer_rows = exp.row_count if exp is not None and getattr(exp, "status", None) else None
    timer_ok = 1 if (exp is not None and getattr(exp, "status", None) is not None
                     and exp.status.is_knowledge) else 0

    # --- KM 生存（是否可用）---
    surv = getattr(ck, "survival_by_gene", {}).get(gene) or ck.survival
    km_ok = 1 if (surv is not None and getattr(surv, "status", None) is not None
                  and surv.status.is_knowledge) else 0

    # --- DESeq2：目标基因的带符号 log2FC、显著基因数、面板规模 ---
    de_l2fc = None
    de_sig = de_n = None
    de = getattr(ck, "differential_expression", None)
    if de is not None:
        de_sig = de.significant_count
        de_n = len(de.top_genes or [])
        for row in de.top_genes or []:
            g = str(row.get("gene_symbol") or row.get("gene") or "").upper()
            if g == gene:
                try:
                    de_l2fc = float(row.get("log2fc"))
                except (TypeError, ValueError):
                    de_l2fc = None
                break

    return {
        "gene": gene, "n_paths": n_paths, "n_promote": n_promote,
        "n_suppress": n_suppress, "n_cancer_hit": n_cancer_hit,
        # 图链拆细特征（rel_AB/rel_BC 两段方向、本癌种通路极性、中介通路多样性）
        "n_ab_promote": n_ab_promote, "n_ab_suppress": n_ab_suppress,
        "n_bc_promote": n_bc_promote, "n_bc_suppress": n_bc_suppress,
        "n_promote_matched": n_promote_matched, "n_suppress_matched": n_suppress_matched,
        "n_distinct_pathways": len(pathways), "n_distinct_pathways_matched": len(pathways_matched),
        "ppr": ppr, "ppr_deg_corrected": deg, "in_graph": in_graph,
        "timer_rows": timer_rows, "timer_ok": timer_ok, "km_ok": km_ok,
        "de_log2fc": de_l2fc, "de_sig_count": de_sig, "de_panel_n": de_n,
        "paralog_redundancy": round((ck.paralog_redundancy or {}).get(gene, 0.0), 4),
    }


def _kg_paths(supplied: dict) -> list[dict]:
    """持久化 gene_disease_paths 的【原始路径行】（所有候选基因，非仅主基因），
    让之后的图特征增强可离线秒算、无需重跑整池。只存证据字段，绝不含真值。"""
    rows: list[dict] = []
    for ev in (getattr(supplied["kg"], "evidence", []) or []):
        if ev.operation != "gene_disease_paths":
            continue
        for r in ev.rows or []:
            if not isinstance(r, dict):
                continue
            rows.append({
                "gene": str(r.get("gene", "")).upper(),
                "pathway": (r.get("pathway") or ""),
                "rel_AB": (r.get("rel_AB") or "").upper() or None,
                "rel_BC": (r.get("rel_BC") or "").upper() or None,
                "effect": (r.get("gene_disease_effect") or "").upper() or None,
                "disease": str(r.get("disease", "")),
            })
    return rows


async def _run_one(team: AnswerTeam, rec: dict) -> dict:
    bucket = rec.get("task_bucket", "")
    supplied = await team.supply(rec)
    ans = await team.decide(supplied, mode="heuristic")
    ck = supplied["ck"]
    return {
        "qid": rec["qid"], "bucket": bucket, "task_type": rec.get("task_type", ""),
        "question": rec["question"], "ground_truth": rec.get("_gt", {}),
        "effective_genes": supplied["genes"],
        "systems": {"pipeline_heuristic": {
            "answer": _dump(ans), "outcome": _score(ans, rec["_gt"], bucket)}},
        # 精简证据摘要（便于人读，不存整份 kg/ck 以免文件膨胀）
        "evidence_view": {
            "paralog_redundancy": {k: round(v, 3) for k, v in (ck.paralog_redundancy or {}).items()},
            "kg_evidence_n": len(getattr(supplied["kg"], "evidence", []) or []),
            "de": bool(ck.differential_expression),
        },
        # ⑤ 免泄漏特征向量 + membership 真值标签（essential=True→1）
        "features": _features(supplied),
        # 原始 SLC->Pathway->Disease 路径行（供离线图特征增强，免重跑池）
        "kg_paths": _kg_paths(supplied),
        "cancer": str((supplied.get("constraints") or {}).get("cancer") or ""),
        "label": (1 if bool((rec.get("_gt") or {}).get("answer")) else 0) if bucket == "membership" else None,
        "provenance": {"dataset": DATASET, "trace_version": TRACE_VERSION, "ts": _now()},
    }


async def main() -> None:
    # 校准需要『未截断』的原始置信度：显式关掉在途保形过滤，避免自我实现
    os.environ.pop("SLC_CONFORMAL_FILE", None)

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--buckets", default="pairwise,membership")
    ap.add_argument("--no-cache", action="store_true", help="忽略已有校准池，全部重算")
    args = ap.parse_args()
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]

    sample = sample_questions(args.n, buckets, args.seed)
    store = {} if args.no_cache else load_calib()
    team = AnswerTeam()
    n_new = n_cached = n_err = 0
    print("=" * 84, flush=True)
    print(f"[保形校准池] dataset={DATASET} version={TRACE_VERSION} 抽样={len(sample)} "
          f"seed={args.seed} 桶={buckets}（已缓存 {len(store)}）", flush=True)

    for i, rec in enumerate(sample, 1):
        entry = store.get(rec["qid"])
        if entry and _reusable(entry):
            n_cached += 1
            tag = "CACHE"
        else:
            try:
                entry = await _run_one(team, rec)
            except Exception as e:  # noqa BLE001
                entry = {"qid": rec["qid"], "bucket": rec.get("task_bucket"),
                         "systems": {"pipeline_heuristic": {"outcome": "abstain"}},
                         "provenance": {"dataset": DATASET, "trace_version": TRACE_VERSION, "ts": _now()},
                         "error": f"{type(e).__name__}: {str(e)[:80]}"}
                n_err += 1
            store[rec["qid"]] = entry
            n_new += 1
            tag = "NEW"
        oc = entry.get("systems", {}).get("pipeline_heuristic", {}).get("outcome", "abstain")
        conf = ((entry.get("systems", {}).get("pipeline_heuristic", {}).get("answer") or {})
                .get("confidence"))
        print(f"[{i:>3}/{len(sample)}] {tag:<5} {rec['qid'][:42]:<42} {oc:<7} "
              f"conf={conf if conf is not None else '-'}", flush=True)
        write_calib(store)                          # 每题落盘，断点可续

    answered = sum(1 for e in store.values()
                   if (e.get("systems", {}).get("pipeline_heuristic") or {}).get("outcome")
                   in ("correct", "wrong") and not e.get("error"))
    print("\n" + "=" * 84, flush=True)
    print(f"[完成] 新增 {n_new} / 复用 {n_cached} / 异常 {n_err} | "
          f"校准池总数={len(store)}（已答 {answered}）", flush=True)
    print(f"校准池: {CALIB_PATH}", flush=True)
    print("下一步: python -m knowledge_agents.conformal_report "
          f"--traces {CALIB_PATH} --version {TRACE_VERSION} --system pipeline_heuristic", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
