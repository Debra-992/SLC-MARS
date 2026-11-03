# -*- coding: utf-8 -*-
"""扩测三方终表：v16-auto vs v12-auto vs mdagents（同 100 题 · 同一 _score 口径）。

重点问题：v16 三通道表决把 v12 的『诚实弃权』结构性转成作答（因先验通道 L1/L2 是
纯 LLM、几乎不弃权，无多数退 v12 规则时 prior 胜证据），行为逼近从不弃答的 mdagents。
本脚本量化：
  1) 三臂 全体口径 / 已答口径 / 覆盖率 / B 防记忆；
  2) v16 相对 v12 的翻转：correct->wrong / wrong->correct / abstain->作答(对·错) / 作答->abstain；
  3) 关键判据：v16-auto 全体口径是否 > mdagents（冠军能否借补覆盖反超穷举作答基线）。

只读 tracestore + answers_mdagents + GT，不触端点，可与链尾安全并行。
用法: .\\.venv\\Scripts\\python.exe tools\\_final_n100.py
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
    """返回 {qid: outcome}，取指定 trace_version 的行（同 qid 最后写覆盖）。"""
    latest: dict[str, dict] = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        v = str((r.get("provenance") or {}).get("trace_version") or "")
        if v != version:
            continue
        latest[r["qid"]] = r
    out: dict[str, str] = {}
    for q, r in latest.items():
        o = (r.get("systems", {}).get(arm) or {}).get("outcome")
        out[q] = o if o in ("correct", "wrong", "abstain") else "abstain"
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

    # mdagents 作答（按 GT 现打 _score）
    md_rows: dict[str, dict] = {}
    p = TRACES / "answers_mdagents.jsonl"
    if p.exists():
        for r in load_jsonl(p):
            md_rows[r["qid"]] = r
    md = {}
    for q in a_qids:
        r = md_rows.get(q)
        if not r or q not in bucket_of:
            md[q] = "missing"
        else:
            md[q] = outcome_of(r, gt.get(q, {}), bucket_of[q])

    arms = {
        "v12-auto": trace_by_version("pipeline_auto", "12"),
        "v12-heur": trace_by_version("pipeline_heuristic", "12"),
        "v16-auto": trace_by_version("pipeline_auto", "16"),
        "mdagents": md,
    }
    v16_present = sum(1 for q in a_qids if q in arms["v16-auto"])
    print(f"# 扩测三方终表  A n={len(a_qids)}  B n={len(b_qids)}  "
          f"(v16 行覆盖={v16_present}/{len(a_qids)} · md 有答={sum(1 for q in a_qids if md[q]!='missing')}/{len(a_qids)})")
    if v16_present < len(a_qids):
        print(f">>> 注意：v16@n100 尚未跑完（仅 {v16_present} 题入库），以下 v16 数字为部分快照，勿定论。")
    print()

    for label, keep in (("A 全100", a_qids), ("A 非B", a_qids - b_qids), ("B 防记忆", b_qids),
                        ("A·pairwise", {q for q in a_qids if bucket_of.get(q) == "pairwise"}),
                        ("A·membership", {q for q in a_qids if bucket_of.get(q) == "membership"})):
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

    # v16 vs v12 翻转（auto）+ v12 弃权题被 v16 救回情况
    v12, v16 = arms["v12-auto"], arms["v16-auto"]
    both = [q for q in a_qids if q in v12 and q in v16]
    c2w = [q for q in both if v12[q] == "correct" and v16[q] == "wrong"]
    w2c = [q for q in both if v12[q] == "wrong" and v16[q] == "correct"]
    ab2ans_correct = [q for q in both if v12[q] == "abstain" and v16[q] == "correct"]
    ab2ans_wrong = [q for q in both if v12[q] == "abstain" and v16[q] == "wrong"]
    ans2ab = [q for q in both if v12[q] != "abstain" and v16[q] == "abstain"]
    ab12 = [q for q in both if v12[q] == "abstain"]
    print("===== v16-auto vs v12-auto 逐题翻转（可比分母 %d）=====" % len(both))
    print(f"  v12 弃权 {len(ab12)} 题 -> v16 救回作答 {len(ab2ans_correct) + len(ab2ans_wrong)}"
          f"（其中✓{len(ab2ans_correct)} ✗{len(ab2ans_wrong)}），仍弃 {len(ab12) - len(ab2ans_correct) - len(ab2ans_wrong)}")
    print(f"  correct->wrong {len(c2w)} | wrong->correct {len(w2c)} | 作答->abstain {len(ans2ab)}")
    net = (len(w2c) + len(ab2ans_correct)) - len(c2w)
    print(f"  全体口径净增减（+救回✓ -翻转✗）= {net:+d} 题 = {net / max(len(both),1):+.1%}pp")
    print(f"  判据：v16-auto acc(全体) vs mdagents acc(全体) -> "
          f"{metrics(v16, a_qids)['acc_all']:.1%} vs {metrics(md, a_qids)['acc_all']:.1%}"
          if metrics(md, a_qids)['n'] else "  (mdagents 未就绪)")


if __name__ == "__main__":
    main()
