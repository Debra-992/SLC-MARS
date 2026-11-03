# -*- coding: utf-8 -*-
"""SLC-MARS 定版（v20）：七系统 × 双数据集汇总表（+ 一个共享证据诊断臂）。

数据来源：
  * pipeline-heuristic / pipeline-auto：tracestore.jsonl，
    取 trace_version ∈ TRACE_VERSIONS 的最新行；
  * pure_llm / lit_rag / metagpt / autogen / mdagents：answers_<system>.jsonl
    （题面驱动，与版本无关，A/B 用同一批作答切片）——这五个即论文 Table 1 的五个基线；
  * metaGPT-native(shared-evidence)：tracestore systems.baseline，让原生 MetaGPT
    直接消费本管线产出的同一份证据。它与 answers_metagpt.jsonl 的 standalone MetaGPT
    是两个不同系统（后者只有框架自带知识、被提示禁止弃答），两行不可互换引用。
评分口径与 answer_pipeline._score 一致（error => 诚实 abstain）。

数据集 B 由 tools/label_antimemorization.py 的双探针标注定义：
  B := class ∈ {novel, unstable, probe_error}（pure-LLM 记忆不可靠/盲区题）。
  注意：B 的判定用 pure-llm 的 probe1 复用，故 pure-llm 在 B 上 acc 偏低是构造使然
  （unstable 题含"probe1 错"），属预期行为，解读时须说明。

用法： .\.venv\Scripts\python.exe -m knowledge_agents.baselines.merge_baselines
输出： outputs/traces/baselines_table_master.md（+ 控制台）
"""
from __future__ import annotations

import json
import math
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa BLE001
        pass

from knowledge_agents.baselines.common import GT_PATH, INPUT_PATH, TRACES

PIPE_ARMS = [("pipeline-heuristic", "tracestore", "pipeline_heuristic"),
             ("pipeline-auto", "tracestore", "pipeline_auto")]
NEW_ARMS = [("pure-llm", "answers", "pure_llm"),
            ("lit-rag", "answers", "lit_rag"),
            ("MetaGPT", "answers", "metagpt"),
            ("autogen", "answers", "autogen"),
            ("mdagents", "answers", "mdagents")]
# 独立诊断臂：原生 MetaGPT 直接消费本管线已产出的同一份证据（tracestore systems.baseline），
# 与上面 answers_metagpt.jsonl 的 standalone MetaGPT 是两个不同系统，勿混用：
# standalone 只做"框架自带知识"对照；native-with-evidence 才是"给 MetaGPT 我们的证据"。
EXTRA_ARMS = [("metaGPT-native(shared-evidence)", "tracestore", "baseline")]
ALL_ARMS = PIPE_ARMS + NEW_ARMS + EXTRA_ARMS
TRACE_VERSIONS = {"20"}  # 管线臂：定版 SLC-MARS v20 行（全量 clean 池 n=489 已跑满，无需旧臂兑底）；历史实验臂 v10-v18 留存 tracestore 但不入表


def load_jsonl(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def outcome_of(row: dict, gt: dict, bucket: str) -> str:
    """与 answer_pipeline._score 同口径的字典版。"""
    if row.get("error") or row.get("abstained"):
        return "abstain"
    if bucket == "pairwise":
        w = str(row.get("winner") or "").upper()
        if not w:
            return "abstain"
        return "correct" if w == str(gt.get("winner", "")).upper() else "wrong"
    a = row.get("answer")
    if a is None:
        return "abstain"
    return "correct" if bool(a) == bool(gt.get("answer")) else "wrong"


def collect(qids: set[str], gt: dict, bucket_of: dict[str, str]) -> dict:
    """-> {arm: {qid: outcome}}，同一 qid 追加历史取最后一条。"""
    latest: dict[str, dict] = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        v = str((r.get("provenance") or {}).get("trace_version") or "")
        if v not in TRACE_VERSIONS:
            continue
        old = latest.get(r["qid"])
        if old is not None and str((old.get("provenance") or {}).get("trace_version") or "0") > v:
            continue                                     # 已有更高版本行 -> 低版本不倒灌
        latest[r["qid"]] = r
    trace_rows = {q: r for q, r in latest.items() if q in qids}

    outs: dict[str, dict[str, str]] = {}
    for name, kind, key in ALL_ARMS:
        per_q: dict[str, str] = {}
        if kind == "tracestore":
            for q in qids:
                r = trace_rows.get(q)
                if r is None:
                    per_q[q] = "missing"
                    continue
                o = (r.get("systems", {}).get(key) or {}).get("outcome")
                per_q[q] = o if o in ("correct", "wrong", "abstain") else "abstain"
        else:
            p = TRACES / f"answers_{key}.jsonl"
            rows: dict[str, dict] = {}
            if p.exists():
                for r in load_jsonl(p):
                    rows[r["qid"]] = r
            for q in qids:
                r = rows.get(q)
                per_q[q] = outcome_of(r, gt.get(q, {}), bucket_of[q]) if r else "missing"
        outs[name] = per_q
    return outs


def metrics(per_q: dict, keep: set[str]) -> dict:
    c = w = a = m = 0
    for q, o in per_q.items():
        if q not in keep:
            continue
        if o == "missing":
            m += 1
        elif o == "correct":
            c += 1
        elif o == "wrong":
            w += 1
        else:
            a += 1
    n = c + w + a
    ans = c + w
    return dict(n=n, c=c, w=w, ab=a, missing=m,
                acc_all=(c / n if n else float("nan")),
                acc_ans=(c / ans if ans else float("nan")),
                cov=(ans / n if n else float("nan")))


def main() -> None:
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((TRACES / "sample_manifest.json").read_text(encoding="utf-8"))
    a_qids = set(manifest["qids"])
    bucket_of = {i["qid"]: i["bucket"] for i in inputs}

    # 数据集 B：防记忆题集（标注缺失则整体退回仅 A 表）
    # 定版口径 = 全量池标注 dataset_b_qids_all489.json（|B|=265）；
    # 旧 dataset_b_qids.json 是 n=100 时代的残留清单，仅在 all489 缺失时兜底，切勿混用两套口径。
    b_path = TRACES / "dataset_b_qids_all489.json"
    if not b_path.exists():
        b_path = TRACES / "dataset_b_qids.json"
    b_qids = set(json.loads(b_path.read_text(encoding="utf-8"))["b_qids"]) if b_path.exists() else None

    datasets = [("A", a_qids)]
    if b_qids is not None:
        datasets.append(("B", b_qids & a_qids))
    outs_all = {tag: collect(keep, gt, bucket_of) for tag, keep in datasets}

    lines = [f"# SLC-MARS 定版（v20）· 七系统 × 双数据集对比（同 {len(a_qids)} 题 · seed{manifest.get('seed')} · 同一 _score 口径）",
             "",
             f"- 数据集A = 全部题（管线臂取 trace_version={sorted(TRACE_VERSIONS)}）；"
             f"数据集B = 防记忆题集（pure-LLM 双探针非双对：novel∪unstable∪probe_error，清单 {b_path.name}）"
             + (f"，|B|={len(datasets[1][1])}" if b_qids is not None else "（标注未生成，本轮仅出 A 表）"),
             "- 注意：B 的构造引用了 pure-llm probe1，pure-llm 在 B 上表现下降是构造使然（预期），解读需附说明。",
             ""]
    for tag, keep in datasets:
        outs = outs_all[tag]
        lines += [f"# ===== 数据集 {tag}（n={len(keep)}）=====", ""]
        for bucket in ("pairwise", "membership"):
            sub_keep = {q for q in keep if bucket_of.get(q) == bucket}
            lines += _render(f"{tag} · {bucket}（n={len(sub_keep)}）", outs, sub_keep)
        lines += _render(f"{tag} · OVERALL（n={len(keep)}）", outs, keep)

    # 运行健康度
    lines.append("## 运行健康度（answers_*.jsonl error 行 / A∪B 缺跑）")
    for name, kind, key in NEW_ARMS:
        p = TRACES / f"answers_{key}.jsonl"
        rows: dict[str, dict] = {}
        if p.exists():
            for r in load_jsonl(p):
                rows[r["qid"]] = r
        err = sum(1 for r in rows.values() if r.get("error"))
        lines.append(f"- {name}: 有作答 {len(rows)}，其中 error {err}")

    md = "\n".join(lines) + "\n"
    out = TRACES / "baselines_table_master.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"[merge] -> {out}")


def _render(title: str, outs: dict, keep: set[str]) -> list[str]:
    lines = [f"## {title}", "",
             "| 系统 | N | ✓ | ✗ | ab | 覆盖率 | acc(全体) | acc(已答) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, *_ in ALL_ARMS:
        m = metrics(outs[name], keep)
        miss = f" (缺{m['missing']})" if m["missing"] else ""
        cov = f"{m['cov']:.1%}" if m['n'] else "-"
        aa = f"{m['acc_all']:.1%}" if m['n'] else "-"
        an = f"{m['acc_ans']:.1%}" if m['n'] and not math.isnan(m['acc_ans']) else "-"
        lines.append(f"| {name} | {m['n']}{miss} | {m['c']} | {m['w']} | {m['ab']} | {cov} | {aa} | {an} |")
    lines.append("")
    return lines


if __name__ == "__main__":
    main()
