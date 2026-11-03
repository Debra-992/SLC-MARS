# -*- coding: utf-8 -*-
"""v20 判决终表：v20-auto（R2 membership 必需性重排）vs v16-auto（冠军）vs autogen vs mdagents。
同 100 题、同一 _score 口径，只读不触端点。判据：
  1) 主判据：v20 全体 acc > v16 的 70.0%；
  2) 靶点：membership 切片必须显著动（R2 只替换启发式兜底路径）；pairwise 不应退化；
  3) 配对翻转 v20-vs-v16（membership 子集）；R2 触发题数（rationale 含 [R2 必需性重排）。
用法: .venv\\Scripts\\python.exe tools\\_final_v20.py
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knowledge_agents.baselines.common import GT_PATH, INPUT_PATH, TRACES  # noqa: E402
from knowledge_agents.baselines.merge_baselines import load_jsonl, outcome_of  # noqa: E402


def trace_by_version(arm: str, version: str) -> dict[str, str]:
    latest: dict[str, dict] = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        if str((r.get("provenance") or {}).get("trace_version") or "") == version:
            latest[r["qid"]] = r
    return {q: (((r.get("systems", {}).get(arm) or {}).get("outcome") or "abstain")
                if ((r.get("systems", {}).get(arm) or {}).get("outcome") in ("correct", "wrong", "abstain"))
                else "abstain") for q, r in latest.items()}


def trace_rat(arm: str, version: str) -> dict[str, str]:
    latest: dict[str, dict] = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        if str((r.get("provenance") or {}).get("trace_version") or "") == version:
            latest[r["qid"]] = r
    return {q: str(((r.get("systems", {}).get(arm) or {}).get("answer") or {}).get("rationale") or "")
            for q, r in latest.items()}


def answers_outcome(system: str, qids: set, bucket_of: dict, gt: dict) -> dict[str, str]:
    rows: dict[str, dict] = {}
    p = TRACES / f"answers_{system}.jsonl"
    if p.exists():
        for r in load_jsonl(p):
            rows[r["qid"]] = r
    return {q: (outcome_of(rows[q], gt.get(q, {}), bucket_of[q])
                if q in rows and q in bucket_of else "missing") for q in qids}


def metrics(per_q: dict[str, str], keep: set) -> dict:
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
                acc_all=(c / n if n else math.nan), acc_ans=(c / ans if ans else math.nan),
                cov=(ans / n if n else math.nan))


def main() -> None:
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((TRACES / "sample_manifest.json").read_text(encoding="utf-8"))
    a_qids = set(manifest["qids"])
    bucket_of = {i["qid"]: i["bucket"] for i in inputs}
    # 定版口径：全量池防记忆清单 all489（|B|=265）；旧 n=100 清单仅在缺失时兜底
    b_path = TRACES / "dataset_b_qids_all489.json"
    if not b_path.exists():
        b_path = TRACES / "dataset_b_qids.json"
    b_qids = set(json.loads(b_path.read_text(encoding="utf-8"))["b_qids"]) & a_qids if b_path.exists() else set()

    arms = {
        "v16-auto": trace_by_version("pipeline_auto", "16"),
        "v20-auto": trace_by_version("pipeline_auto", "20"),
        "autogen": answers_outcome("autogen", a_qids, bucket_of, gt),
        "mdagents": answers_outcome("mdagents", a_qids, bucket_of, gt),
    }
    v20n = sum(1 for q in a_qids if q in arms["v20-auto"])
    print(f"# v20(R2) 判决终表  A n={len(a_qids)}  B n={len(b_qids)}  (v20 行覆盖={v20n}/{len(a_qids)})")
    if v20n < len(a_qids):
        print(f">>> 注意：v20@n100 未跑完（仅 {v20n} 题），以下为部分快照，勿定论。")
    print()

    memb = {q for q in a_qids if bucket_of.get(q) == "membership"}
    r2 = trace_rat("pipeline_auto", "20")
    r2_fired = {q for q in memb if "[R2 必需性重排" in r2.get(q, "")}
    print(f"# membership n={len(memb)}  其中 R2 触发={len(r2_fired)}（应≈启发式兜底路径题数）\n")

    for label, keep in (("A 全100", a_qids), ("B 防记忆", b_qids),
                        ("A·membership(靶点)", memb), ("A·pairwise", a_qids - memb)):
        if not keep:
            continue
        print(f"===== {label} (n={len(keep)}) =====")
        print(f"  {'arm':10s} {'✓':>3} {'✗':>3} {'ab':>3} {'缺':>3}  {'覆盖':>6} {'acc全体':>8} {'acc已答':>8}")
        for name, per_q in arms.items():
            mm = metrics(per_q, keep)
            fa = f"{mm['acc_all']:.1%}" if mm['n'] and not math.isnan(mm['acc_all']) else "-"
            fn = f"{mm['acc_ans']:.1%}" if mm['n'] and not math.isnan(mm['acc_ans']) else "-"
            cv = f"{mm['cov']:.1%}" if mm['n'] else "-"
            print(f"  {name:10s} {mm['c']:>3} {mm['w']:>3} {mm['ab']:>3} {mm['miss']:>3}  {cv:>6} {fa:>8} {fn:>8}")
        print()

    v16, v20 = arms["v16-auto"], arms["v20-auto"]
    both = [q for q in a_qids if q in v16 and q in v20]
    mboth = [q for q in both if q in memb]
    w2c = sum(1 for q in mboth if v16[q] == "wrong" and v20[q] == "correct")
    c2w = sum(1 for q in mboth if v16[q] == "correct" and v20[q] == "wrong")
    print("===== v20-auto vs v16-auto 配对（membership 靶点）=====")
    print(f"  membership n={len(mboth)}：wrong->correct {w2c} | correct->wrong {c2w} | 净 {w2c-c2w:+d}")
    hm = [q for q in mboth if q in r2_fired]
    hw = sum(1 for q in hm if v16[q] == "wrong" and v20[q] == "correct")
    hc = sum(1 for q in hm if v16[q] == "correct" and v20[q] == "wrong")
    print(f"  R2 触发题 n={len(hm)}：w2c {hw} | c2w {hc} | 净 {hw-hc:+d}（这是 R2 真正的弹着）")
    m16, m20 = metrics(v16, a_qids), metrics(v20, a_qids)
    pw2c = sum(1 for q in both if q not in memb and v16[q] == "correct" and v20[q] == "wrong")
    print(f"  pairwise 意外退化(c->w)={pw2c}（R2 不应动 pairwise，>0 需查）")
    if m20['n'] == len(a_qids) and m16['n'] == len(a_qids):
        verdict = "PASS(晋升候选)" if m20['acc_all'] > m16['acc_all'] else "NEGATIVE(保持 v16)"
        print(f"  判据：v20 {m20['acc_all']:.1%} vs v16 {m16['acc_all']:.1%}  ->  {verdict}")


if __name__ == "__main__":
    main()
