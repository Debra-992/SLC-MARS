# -*- coding: utf-8 -*-
"""v17 判决终表：v17-auto（L2 挂轻检索票）vs v16-auto（新冠军）vs autogen vs mdagents。
同 100 题、同一 _score 口径、只读 tracestore/answers/GT，不触端点。

判据：
  1) 标题 acc(全体) v17 是否 > v16（主判据：机制是否传导到标题）；
  2) B·pairwise 是否脱离"平 heuristic"（v17 的直接靶点：票源去同源）；
  3) v17 vs v16 逐题翻转净增减（配对证据，比标题差更稳）。
用法: .\\.venv\\Scripts\\python.exe tools\\_final_v17.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa BLE001
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knowledge_agents.baselines.common import GT_PATH, INPUT_PATH, TRACES  # noqa: E402
from knowledge_agents.baselines.merge_baselines import load_jsonl, outcome_of  # noqa: E402


def trace_by_version(arm: str, version: str) -> dict[str, str]:
    latest: dict[str, dict] = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        v = str((r.get("provenance") or {}).get("trace_version") or "")
        if v == version:
            latest[r["qid"]] = r
    out: dict[str, str] = {}
    for q, r in latest.items():
        o = (r.get("systems", {}).get(arm) or {}).get("outcome")
        out[q] = o if o in ("correct", "wrong", "abstain") else "abstain"
    return out


def answers_outcome(system: str, qids: set[str], bucket_of: dict, gt: dict) -> dict[str, str]:
    rows: dict[str, dict] = {}
    p = TRACES / f"answers_{system}.jsonl"
    if p.exists():
        for r in load_jsonl(p):
            rows[r["qid"]] = r
    out = {}
    for q in qids:
        r = rows.get(q)
        out[q] = outcome_of(r, gt.get(q, {}), bucket_of[q]) if (r and q in bucket_of) else "missing"
    return out


def metrics(per_q: dict[str, str], keep: set[str]) -> dict:
    c = w = a = m = 0
    for q in keep:
        o = per_q.get(q, "missing")
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
    return dict(n=n, c=c, w=w, ab=a, miss=m,
                acc_all=(c / n if n else math.nan),
                acc_ans=(c / ans if ans else math.nan),
                cov=(ans / n if n else math.nan))


def main() -> None:
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((TRACES / "sample_manifest.json").read_text(encoding="utf-8"))
    a_qids = set(manifest["qids"])
    bucket_of = {i["qid"]: i["bucket"] for i in inputs}
    b_path = TRACES / "dataset_b_qids.json"
    b_qids = set(json.loads(b_path.read_text(encoding="utf-8"))["b_qids"]) & a_qids if b_path.exists() else set()

    arms = {
        "v16-auto": trace_by_version("pipeline_auto", "16"),
        "v16-heur": trace_by_version("pipeline_heuristic", "16"),
        "v17-auto": trace_by_version("pipeline_auto", "17"),
        "autogen": answers_outcome("autogen", a_qids, bucket_of, gt),
        "mdagents": answers_outcome("mdagents", a_qids, bucket_of, gt),
    }
    v17_present = sum(1 for q in a_qids if q in arms["v17-auto"])
    print(f"# v17 判决终表  A n={len(a_qids)}  B n={len(b_qids)}  (v17 行覆盖={v17_present}/{len(a_qids)})")
    if v17_present < len(a_qids):
        print(f">>> 注意：v17@n100 未跑完（仅 {v17_present} 题），以下为部分快照，勿定论。")
    print()

    b_pair = {q for q in b_qids if bucket_of.get(q) == "pairwise"}
    for label, keep in (("A 全100", a_qids), ("A 非B", a_qids - b_qids), ("B 防记忆", b_qids),
                        ("A·pairwise", {q for q in a_qids if bucket_of.get(q) == "pairwise"}),
                        ("A·membership", {q for q in a_qids if bucket_of.get(q) == "membership"}),
                        ("B·pairwise(靶点)", b_pair)):
        if not keep:
            continue
        print(f"===== {label} (n={len(keep)}) =====")
        print(f"  {'arm':10s} {'✓':>3} {'✗':>3} {'ab':>3} {'缺':>3}  {'覆盖':>6} {'acc全体':>8} {'acc已答':>8}")
        for name, per_q in arms.items():
            m = metrics(per_q, keep)
            fa = f"{m['acc_all']:.1%}" if m['n'] and not math.isnan(m['acc_all']) else "-"
            fn = f"{m['acc_ans']:.1%}" if m['n'] and not math.isnan(m['acc_ans']) else "-"
            cv = f"{m['cov']:.1%}" if m['n'] else "-"
            print(f"  {name:10s} {m['c']:>3} {m['w']:>3} {m['ab']:>3} {m['miss']:>3}  {cv:>6} {fa:>8} {fn:>8}")
        print()

    # v17 vs v16 配对翻转（主机制证据）
    v16, v17 = arms["v16-auto"], arms["v17-auto"]
    both = [q for q in a_qids if q in v16 and q in v17]
    c2w = [q for q in both if v16[q] == "correct" and v17[q] == "wrong"]
    w2c = [q for q in both if v16[q] == "wrong" and v17[q] == "correct"]
    ab2a = [q for q in both if v16[q] == "abstain" and v17[q] != "abstain"]
    a2ab = [q for q in both if v16[q] != "abstain" and v17[q] == "abstain"]
    print(f"===== v17-auto vs v16-auto 逐题配对（分母 {len(both)}）=====")
    print(f"  wrong->correct {len(w2c)} | correct->wrong {len(c2w)} | 净 {len(w2c)-len(c2w):+d} 题")
    print(f"  弃权->作答 {len(ab2a)} | 作答->弃权 {len(a2ab)}")
    bp_both = [q for q in both if q in b_pair]
    bp_w2c = sum(1 for q in bp_both if v16[q] == "wrong" and v17[q] == "correct")
    bp_c2w = sum(1 for q in bp_both if v16[q] == "correct" and v17[q] == "wrong")
    print(f"  靶点切片 B·pairwise（n={len(bp_both)}）：w2c {bp_w2c} vs c2w {bp_c2w}，净 {bp_w2c-bp_c2w:+d}")
    m16, m17 = metrics(v16, a_qids), metrics(v17, a_qids)
    if m17['n'] == len(a_qids) and m16['n'] == len(a_qids):
        verdict = "PASS(晋升候选)" if m17['acc_all'] > m16['acc_all'] else "NEGATIVE(保持 v16)"
        print(f"  判据：v17 {m17['acc_all']:.1%} vs v16 {m16['acc_all']:.1%}  ->  {verdict}")


if __name__ == "__main__":
    main()
