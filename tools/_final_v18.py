# -*- coding: utf-8 -*-
"""v18 判决终表：v18-auto（membership 负证据通道）vs v16-auto（冠军）vs autogen vs mdagents。
同 100 题、同一 _score 口径，只读不触端点。判据：
  1) 主判据：v18 全体 acc > v18 基线 v16 的 70.0%；
  2) 靶点判据：B·membership 与 A·membership —— 负证据通道的弹着切片必须显著动；
  3) 配对翻转 v18-vs-v16（全量 + B·membership 子集）；membership 没动的臂=机制空转。
  4) 双口径选择性弃权：conf 已持久化 => 任意 θ 离线模拟（免重跑），同题配对出
     AUC-RC / 固定风险下最大覆盖 / 最大已答口径，判 v18 证据化置信度能否优于 v16。
用法: .venv\\Scripts\\python.exe tools\\_final_v18.py
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
        if str((r.get("provenance") or {}).get("trace_version") or "") == version:
            latest[r["qid"]] = r
    return {q: ((r.get("systems", {}).get(arm) or {}).get("outcome") or "abstain")
            if ((r.get("systems", {}).get(arm) or {}).get("outcome") in ("correct", "wrong", "abstain"))
            else "abstain" for q, r in latest.items()}


def answers_outcome(system: str, qids: set[str], bucket_of: dict, gt: dict) -> dict[str, str]:
    rows: dict[str, dict] = {}
    p = TRACES / f"answers_{system}.jsonl"
    if p.exists():
        for r in load_jsonl(p):
            rows[r["qid"]] = r
    return {q: (outcome_of(rows[q], gt.get(q, {}), bucket_of[q])
                if q in rows and q in bucket_of else "missing") for q in qids}


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
                acc_all=(c / n if n else math.nan), acc_ans=(c / ans if ans else math.nan),
                cov=(ans / n if n else math.nan))


def trace_co(arm: str, version: str) -> dict[str, tuple]:
    """qid -> (outcome, conf, rationale)，供双口径弃权模拟（conf 已从 trace 持久化）。"""
    latest: dict[str, dict] = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        if str((r.get("provenance") or {}).get("trace_version") or "") == version:
            latest[r["qid"]] = r
    out: dict[str, tuple] = {}
    for q, r in latest.items():
        s = (r.get("systems", {}).get(arm) or {})
        a = s.get("answer") or {}
        o = s.get("outcome") or "abstain"
        if o not in ("correct", "wrong", "abstain"):
            o = "abstain"
        try:
            conf = float(a.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        out[q] = (o, conf, str(a.get("rationale") or ""))
    return out


def rc_pts(rows: list[tuple]) -> list[tuple]:
    """按 conf 降序逐题纳入作答 => [(coverage, risk)]（risk=累计错误率）。rows=[(conf,outcome)]。"""
    n = len(rows)
    if not n:
        return []
    ans = sorted([(c, o) for c, o in rows if o in ("correct", "wrong")], key=lambda t: -t[0])
    pts, err = [], 0
    for i, (_, o) in enumerate(ans, 1):
        if o == "wrong":
            err += 1
        pts.append((i / n, err / i))
    return pts


def auc_rc(pts: list[tuple]) -> float:
    return (sum((pts[k][0] - pts[k - 1][0]) * (pts[k][1] + pts[k - 1][1]) / 2
                for k in range(1, len(pts))) if len(pts) > 1 else math.nan)


def risk_at(pts: list[tuple], target: float) -> float:
    c = [(cc, r) for cc, r in pts if cc <= target + 1e-9]
    return c[-1][1] if c else math.nan


def cover_at_risk(pts: list[tuple], cap: float) -> float:
    """风险 <= cap 时可达的最大覆盖率（曲线非单调，取满足上限的最大 coverage 点）。"""
    best = math.nan
    for cc, r in pts:
        if r <= cap:
            best = cc
    return best


def best_ans_theta(rows: list[tuple]) -> tuple | None:
    """扫 θ 使已答口径 acc_ans 最大化（弃权砍低置信题换可靠性）。返回 (θ, acc_ans, cov)。"""
    n = len(rows)
    best = None
    for x in range(0, 46):
        th = round(x * 0.02, 3)
        c = w = 0
        for conf, o in rows:
            if o in ("correct", "wrong") and conf >= th:
                if o == "correct":
                    c += 1
                else:
                    w += 1
        ans = c + w
        if ans == 0:
            continue
        aa = c / ans
        if best is None or aa > best[1]:
            best = (th, aa, ans / n)
    return best


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
        "v18-auto": trace_by_version("pipeline_auto", "18"),
        "v18-heur": trace_by_version("pipeline_heuristic", "18"),
        "autogen": answers_outcome("autogen", a_qids, bucket_of, gt),
        "mdagents": answers_outcome("mdagents", a_qids, bucket_of, gt),
    }
    v18n = sum(1 for q in a_qids if q in arms["v18-auto"])
    print(f"# v18 判决终表  A n={len(a_qids)}  B n={len(b_qids)}  (v18 行覆盖={v18n}/{len(a_qids)})")
    if v18n < len(a_qids):
        print(f">>> 注意：v18@n100 未跑完（仅 {v18n} 题），以下为部分快照，勿定论。")
    print()

    memb = {q for q in a_qids if bucket_of.get(q) == "membership"}
    bmemb = memb & b_qids
    for label, keep in (("A 全100", a_qids), ("A 非B", a_qids - b_qids), ("B 防记忆", b_qids),
                        ("A·membership", memb), ("B·membership(靶点)", bmemb),
                        ("A·pairwise", a_qids - memb)):
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

    v16, v18 = arms["v16-auto"], arms["v18-auto"]
    both = [q for q in a_qids if q in v16 and q in v18]
    c2w = [q for q in both if v16[q] == "correct" and v18[q] == "wrong"]
    w2c = [q for q in both if v16[q] == "wrong" and v18[q] == "correct"]
    print(f"===== v18-auto vs v16-auto 逐题配对（分母 {len(both)}）=====")
    print(f"  wrong->correct {len(w2c)} | correct->wrong {len(c2w)} | 净 {len(w2c)-len(c2w):+d} 题")
    bm_both = [q for q in both if q in bmemb]
    bm_w2c = sum(1 for q in bm_both if v16[q] == "wrong" and v18[q] == "correct")
    bm_c2w = sum(1 for q in bm_both if v16[q] == "correct" and v18[q] == "wrong")
    print(f"  靶点 B·membership（n={len(bm_both)}）：w2c {bm_w2c} vs c2w {bm_c2w}，净 {bm_w2c-bm_c2w:+d}")
    mm_both = [q for q in bm_both + [q for q in both if q in memb and q not in b_qids]]
    print(f"  全 membership 配对（n={len([q for q in both if q in memb])}）："
          f"w2c {sum(1 for q in both if q in memb and v16[q]=='wrong' and v18[q]=='correct')}"
          f" vs c2w {sum(1 for q in both if q in memb and v16[q]=='correct' and v18[q]=='wrong')}")
    m16, m18 = metrics(v16, a_qids), metrics(v18, a_qids)
    if m18['n'] == len(a_qids) and m16['n'] == len(a_qids):
        verdict = "PASS(晋升候选)" if m18['acc_all'] > m16['acc_all'] else "NEGATIVE(保持 v16)"
        print(f"  判据：v18 {m18['acc_all']:.1%} vs v16 {m16['acc_all']:.1%}  ->  {verdict}")

    # ④ 双口径选择性弃权（conf 已持久化 => 任意 θ 离线模拟，免重跑）
    v16co, v18co = trace_co("pipeline_auto", "16"), trace_co("pipeline_auto", "18")

    def _dirty(d: dict, q: str) -> bool:
        t = d.get(q)
        return (not t) or (t[0] == "abstain" and "LLM 不可用" in t[2])

    pset = [q for q in a_qids if q in v16co and q in v18co
            and not _dirty(v18co, q) and not _dirty(v16co, q)]
    rows16 = [(v16co[q][1], v16co[q][0]) for q in pset]
    rows18 = [(v18co[q][1], v18co[q][0]) for q in pset]
    print(f"\n===== ④ 双口径选择性弃权（同题配对 清洁集 n={len(pset)}）=====")
    if len(pset) < 10:
        print("  样本过小，仅示意。")
    p16, p18 = rc_pts(rows16), rc_pts(rows18)
    a16, a18 = auc_rc(p16), auc_rc(p18)
    print(f"  AUC-RC↓：v16 {a16:.4f} | v18 {a18:.4f}  ->  "
          f"{'v18 更优(不确定性质量好)' if a18 < a16 else 'v16 不逊于 v18'}")
    print(f"  {'口径':22s} {'v16':>10} {'v18':>10}")
    for cap in (0.05, 0.10, 0.15):
        c16, c18 = cover_at_risk(p16, cap), cover_at_risk(p18, cap)
        f16 = f"{c16:.0%}" if not math.isnan(c16) else "达不到"
        f18 = f"{c18:.0%}" if not math.isnan(c18) else "达不到"
        print(f"  风险≤{cap:.0%} 时最大覆盖{'':>6}{f16:>10} {f18:>10}")
    for tgt in (0.70, 0.80, 0.90):
        r16, r18 = risk_at(p16, tgt), risk_at(p18, tgt)
        print(f"  覆盖{tgt:.0%} 处风险{'':>10}{r16:>9.1%} {r18:>9.1%}")
    b16, b18 = best_ans_theta(rows16), best_ans_theta(rows18)
    fmt = lambda b: (f"θ*={b[0]:.2f} 已答{b[1]:.1%} (cov {b[2]:.0%})" if b else "-")
    print(f"  最优弃权工作点：v16 {fmt(b16)} | v18 {fmt(b18)}")


if __name__ == "__main__":
    main()
