# -*- coding: utf-8 -*-
"""导出 baseline 同题输入：与定版 v20 公平对比完全相同的题集（seed5, clean, n=489）。

产出两文件（严格分离，防泄漏）：
  outputs/traces/baselines_input.json  —— 任何 runner 可读：qid/bucket/question/candidates/constraints
  outputs/traces/baselines_gt.json     —— 仅 merge_baselines.py 评分用，baseline runner 禁读

候选基因规则与管线一致：pairwise=题面两基因；membership= 'Is/Does <GENE>' 真正在问的基因
（排除 treatment 里 'vs SLCxxx.KO' 的被敲除基因）。

用法：
  .\\.venv\\Scripts\\python.exe -m knowledge_agents.baselines.export_inputs           # 默认：定版 v20 manifest 那批题（A 集，逐字节不变）
  .\\.venv\\Scripts\\python.exe -m knowledge_agents.baselines.export_inputs --all     # 全量：clean 集全部 pairwise/membership 题（排除 topk），绕过 manifest
"""
from __future__ import annotations

import argparse
import json
import re

from knowledge_agents.baselines.common import GT_PATH, INPUT_PATH, TRACES

_GENE_RE = re.compile(r"\b(SLC[A-Z0-9]+)\b")
_ROOT = TRACES.parents[1]


def load_jsonl(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def candidates_for(rec: dict) -> list[str]:
    question = rec["question"]
    cons = rec.get("constraints") or {}
    genes = _GENE_RE.findall(question)
    if not genes and cons.get("gene"):
        genes = [cons["gene"]]
    genes = list(dict.fromkeys(g.upper() for g in genes))
    if rec.get("task_bucket") == "membership":
        m = re.search(r"\b(?:Is|Does)\s+(SLC[A-Z0-9]+)\b", question, re.I)
        if m:
            return [m.group(1).upper()]
    return genes


def _build_inputs(qids, questions, gt):
    inputs, missing = [], []
    for q in qids:
        rec = questions.get(q)
        if not rec:
            missing.append(q)
            continue
        inputs.append({
            "qid": q,
            "bucket": rec.get("task_bucket", ""),
            "task_type": rec.get("task_type", ""),
            "question": rec["question"],
            "candidates": candidates_for(rec),
            "constraints": rec.get("constraints", {}),
        })
    if missing:
        raise SystemExit(f"清单中 {len(missing)} 个 qid 在 questions_clean 缺失: {missing[:3]}")
    bad = [i["qid"] for i in inputs if not i["candidates"]]
    if bad:
        raise SystemExit(f"{len(bad)} 题抽不出候选基因，先修 export: {bad[:3]}")
    return inputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="导出 clean 集全部 pairwise/membership 题（排除 topk），绕过定版 manifest")
    args = ap.parse_args()

    questions = {r["qid"]: r for r in load_jsonl(_ROOT / "data" / "questions_clean.jsonl")}
    gt = {r["qid"]: r.get("ground_truth", {}) for r in load_jsonl(_ROOT / "evaluation" / "ground_truth_clean.jsonl")}

    if args.all:
        # 全量：直接枚举 clean 集里的评测桶（topk 不属于 baseline 二选一/是否口径，排除）。
        # 不读、不写 sample_manifest.json —— 与正在跑的 v20(compare_eval) 零交集。
        qids = [q for q, r in questions.items() if r.get("task_bucket") in {"pairwise", "membership"}]
    else:
        mani = json.loads((TRACES / "sample_manifest.json").read_text(encoding="utf-8"))
        # 定版题集：manifest.trace_version 必须是冻结的 v20（同一共享证据 backbone，题面/候选与
        # 基线可同题对位；v20 只在判定层加表决与 R2 重排，不改题面）。
        assert str(mani.get("trace_version")) == "20" and mani.get("dataset") == "clean", \
            f"manifest 非定版线 v20/clean: {mani.get('trace_version')}/{mani.get('dataset')}"
        qids = mani["qids"]

    inputs = _build_inputs(qids, questions, gt)

    INPUT_PATH.write_text(json.dumps(inputs, ensure_ascii=False, indent=1), encoding="utf-8")
    GT_PATH.write_text(json.dumps({q: gt.get(q, {}) for q in qids}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    n_pw = sum(1 for i in inputs if i["bucket"] == "pairwise")
    print(f"导出 {len(inputs)} 题（pairwise {n_pw} / membership {len(inputs) - n_pw}）-> {INPUT_PATH.name}, {GT_PATH.name}")


if __name__ == "__main__":
    main()
