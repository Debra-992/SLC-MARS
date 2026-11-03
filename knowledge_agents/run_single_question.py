# -*- coding: utf-8 -*-
"""随机抽一道测试题，依次跑【知识检索智能体】与【计算智能体】并输出全过程。

用法：
  python -m knowledge_agents.run_single_question           # 真随机抽一题
  python -m knowledge_agents.run_single_question --qid membership_negative_ALL_Human_CUTLL1_Low-2V_00002
  python -m knowledge_agents.run_single_question --seed 42  # 可复现抽样

输出：
  1) 题目原文 + 约束 + ground_truth 正确答案
  2) 知识检索智能体（Kira / project_knowledge + text2cypher）：逐条 receipt 中间结果 + 最终 status
  3) 计算智能体（Cael / PPR+TIMER+KM+DESeq2）：逐工具中间结果 + 最终 status
  4) 证据完整性说明：未知/被拒 != 阴性，两智能体只供证据、不直接判定真值
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path

from knowledge_agents.roles.compute_role import ComputeRole
from knowledge_agents.roles.knowledge_query_role import KnowledgeQueryRole
from knowledge_agents.schema import (
    BackgroundKnowledge,
    ComputedKnowledge,
    parse_knowledge,
)

ROOT = Path(__file__).resolve().parent.parent


def _resolve_dataset() -> tuple[Path, Path]:
    """SLC_DATASET=clean|full（默认 clean，若清洗后的文件存在）。

    clean: 剔除『KO 处理基因 ≠ 提问基因』这类科学上不成立的题目后的数据集。
    """
    choice = os.environ.get("SLC_DATASET", "clean").lower()
    clean_q = ROOT / "data" / "questions_clean.jsonl"
    clean_g = ROOT / "evaluation" / "ground_truth_clean.jsonl"
    if choice == "clean" and clean_q.exists() and clean_g.exists():
        return clean_q, clean_g
    return ROOT / "data" / "questions.jsonl", ROOT / "evaluation" / "ground_truth.jsonl"


QUESTIONS, GTRUTH = _resolve_dataset()

_GENE_RE = re.compile(r"\b(SLC[A-Z0-9]+)\b")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_genes(question: str, constraints: dict) -> list[str]:
    """题面出现的全部 SLC 候选基因（pairwise 有两基因、topk 有多个）。"""
    genes = _GENE_RE.findall(question)
    if not genes and constraints.get("gene"):
        genes = [constraints["gene"]]
    # 去重保序
    return list(dict.fromkeys(genes))


def format_ground_truth(gt: dict, task_bucket: str) -> str:
    """三种桶的地面真值结构不同：membership->answer / pairwise->winner / topk->有序列表。"""
    if task_bucket == "membership":
        return f"answer={gt.get('answer')}"
    if task_bucket == "pairwise":
        return f"winner={gt.get('winner')}"
    if task_bucket == "topk":
        # topk 的 ground_truth 直接是排序基因列表
        return f"ranking={gt.get('ranking') or gt}"
    return json.dumps(gt, ensure_ascii=False)


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


async def run_knowledge(qid: str, question: str) -> BackgroundKnowledge:
    hr("[智能体 1] 知识检索智能体 Kira (KnowledgeQueryRole)")
    role = KnowledgeQueryRole()
    # 喂 qid= 前缀，令 _infer_qid 精确命中白名单查询（不依赖题面正则）
    await role.run(f"qid={qid} {question}")
    kg = parse_knowledge(role.rc.memory.get()[-1].content)
    assert isinstance(kg, BackgroundKnowledge)

    print(f"  主知识 status = {kg.status.value}  source = {kg.source}")
    print(f"  候选基因 = {kg.candidates}")
    print(f"  题目约束 requested_context = {json.dumps(kg.requested_context, ensure_ascii=False)}")
    print("  --- 中间结果：逐条证据 receipt ---")
    if not kg.evidence:
        print("    (无证据回执：主知识未取回，保持未知)")
    for ev in kg.evidence:
        rows = ev.rows or []
        head = json.dumps(rows[0], ensure_ascii=False, default=str)[:120] if rows else "-"
        print(f"    * op={ev.operation:<16} status={ev.status.value:<12} rows={len(rows):<3} 首行={head}")
        if ev.cypher:
            print(f"        cypher={ev.cypher[:140]}")
    if kg.cypher:
        print(f"  --- text2cypher 补充 ---")
        print(f"    cypher={kg.cypher[:200]}")
        print(f"    返回行数={len(kg.cypher_rows)}")
    print(f"  最终 summary = {kg.summary}")
    print(f"  public_web_used={kg.public_web_used}  source_independence={kg.source_independence}")
    return kg


async def run_compute(qid: str, genes: list[str], cancer: str, question: str, bucket: str) -> ComputedKnowledge:
    hr("[智能体 2] 计算智能体 Cael (ComputeRole)")
    role = ComputeRole()
    role.set_task(qid=qid, gene=(genes[0] if genes else None), genes=genes, cancer=cancer, task_bucket=bucket)
    await role.run(f"gene={genes[0] if genes else ''} cancer={cancer} bucket={bucket} {question}")
    ck = parse_knowledge(role.rc.memory.get()[-1].content)
    assert isinstance(ck, ComputedKnowledge)

    print(f"  路由 task_bucket = {bucket!r}  genes={genes} cancer={cancer}")
    print("  --- 中间结果：逐计算工具 ---")
    gc = ck.graph_computation
    if gc:
        top = ", ".join(f"{c.gene}={c.degree_corrected_score}" for c in (gc.candidates or [])[:5]) or "无"
        print(f"    * PPR    : status={gc.status.value} seeds={gc.seed_count} "
              f"nodes={gc.node_count} edges={gc.edge_count} top5=[{top}]")
    if ck.expression_by_gene:
        for g, ex in ck.expression_by_gene.items():
            print(f"    * TIMER[{g}] : status={ex.status.value} row_count={ex.row_count} "
                  f"scope={ex.evidence_scope} boxplot={ex.boxplot_path}")
    elif ck.expression:
        ex = ck.expression
        print(f"    * TIMER  : status={ex.status.value} row_count={ex.row_count} "
              f"scope={ex.evidence_scope} boxplot={ex.boxplot_path}")
    de = ck.differential_expression
    if de:
        top = ", ".join(f"{g.get('gene')}({g.get('log2fc'):.2f})" for g in (de.top_genes or [])[:5]
                        if g.get("log2fc") is not None) or "无"
        print(f"    * DESeq2 : status={de.status.value} significant={de.significant_count} "
              f"sample_counts={de.sample_counts} top5=[{top}]")
    if ck.survival_by_gene:
        for g, km in ck.survival_by_gene.items():
            stats = (km.stats_text or "").splitlines()
            print(f"    * KM[{g}]    : status={km.status.value} scope={km.evidence_scope} "
                  f"stats_head={stats[0] if stats else ''!r} plot={km.plot_path}")
    elif ck.survival:
        km = ck.survival
        stats = (km.stats_text or "").splitlines()
        print(f"    * KM     : status={km.status.value} scope={km.evidence_scope} "
              f"stats_head={stats[0] if stats else ''!r} plot={km.plot_path}")
    if ck.skipped_tools:
        print(f"    (未路由/跳过: {ck.skipped_tools})")
    print(f"  最终 status = {ck.status.value}")
    print(f"  最终 summary = {ck.summary}")
    return ck


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default=None, help="指定题目 qid（默认随机）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    args = ap.parse_args()

    questions = load_jsonl(QUESTIONS)
    gt_index = {r["qid"]: r for r in load_jsonl(GTRUTH)}

    if args.qid:
        q = next((r for r in questions if r["qid"] == args.qid), None)
        if q is None:
            raise SystemExit(f"未找到 qid={args.qid}")
    else:
        rng = random.Random(args.seed)
        q = rng.choice(questions)

    qid = q["qid"]
    constraints = q.get("constraints", {})
    genes = extract_genes(q["question"], constraints)
    gene = genes[0] if genes else None
    cancer = constraints.get("cancer", "")
    bucket = q.get("task_bucket", "")
    task_type = q.get("task_type", "")
    gt = gt_index.get(qid, {}).get("ground_truth", {})

    hr(f"[题目] 随机抽样（共 {len(questions)} 题）")
    print(f"  qid        = {qid}")
    print(f"  task_bucket= {bucket}   task_type={task_type}")
    print(f"  constraints= {json.dumps(constraints, ensure_ascii=False)}")
    print(f"  question   = {q['question']}")
    print(f"  题面候选基因 = {genes}  (路由主基因 gene={gene}  cancer={cancer})")
    print(f"  ground_truth -> {format_ground_truth(gt, bucket)}")

    kg = await run_knowledge(qid, q["question"])
    ck = await run_compute(qid, genes, cancer, q["question"], bucket)

    hr("[汇总] 两个智能体输出 vs 正确答案")
    print(f"  知识检索 Kira : status={kg.status.value:<12} 证据条数={len(kg.evidence)}")
    print(f"  计算智能 Cael : status={ck.status.value:<12} "
          f"(ppr={'Y' if ck.graph_computation else '-'} timer={'Y' if ck.expression else '-'} "
          f"de={'Y' if ck.differential_expression else '-'} km={'Y' if ck.survival else '-'})")
    print(f"  ground_truth  : {format_ground_truth(gt, bucket)}")
    print("  说明：两智能体仅供给『背景知识 + 候选数值证据』，membership 真值判定")
    print("        需下游判定角色结合 CRISPR 筛选读数完成；工具未知/被拒绝不作阴性。")


if __name__ == "__main__":
    asyncio.run(main())
