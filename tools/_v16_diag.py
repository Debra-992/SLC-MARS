# -*- coding: utf-8 -*-
"""① 置信信号离线重建 + 错因解剖（全程只读 v16/v18 已持久化 trace，免端点）。

PART A：从 answer 字段(gene_scores/decision_mode/rationale)重建候选置信信号，
        与 raw conf、autogen、mdagents 同口径算 AUC-RC/ECE，看换信号源能否夺回不确定性质量。
PART B：把 v16-auto 答错的题按 decision_mode / 一致-judge 分支 / bucket 归类，
        附 rationale 摘录，看具体死因。
用法: .venv\\Scripts\\python.exe tools\\_v16_diag.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa BLE001
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knowledge_agents.baselines.common import TRACES, GT_PATH, INPUT_PATH  # noqa: E402
from knowledge_agents.baselines.merge_baselines import load_jsonl, outcome_of  # noqa: E402


def auc_rc(rows):
    n = len(rows)
    ans = sorted([(c, o) for c, o in rows if o != "abstain"], key=lambda t: -t[0])
    pts, err = [], 0
    for i, (_, o) in enumerate(ans, 1):
        if o == "wrong":
            err += 1
        pts.append((i / n, err / i))
    auc = sum((pts[k][0]-pts[k-1][0])*(pts[k][1]+pts[k-1][1])/2 for k in range(1, len(pts))) if len(pts) > 1 else math.nan
    return auc, pts


def ece(rows, bins=10):
    answered = [r for r in rows if r[1] != "abstain"]
    if not answered:
        return math.nan
    tot, acc = len(answered), 0.0
    for b in range(bins):
        lo, hi = b/bins, (b+1)/bins
        sub = [r for r in answered if lo <= r[0] < hi or (hi == 1.0 and r[0] == 1.0)]
        if not sub:
            continue
        cb = sum(s[0] for s in sub)/len(sub)
        ab = sum(1 for s in sub if s[1] == "correct")/len(sub)
        acc += len(sub)/tot*abs(cb-ab)
    return acc


def norm(x, lo, hi):
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (x-lo)/(hi-lo)))


def build_signals():
    lat = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        v = str((r.get("provenance") or {}).get("trace_version") or "")
        if v in ("16", "18"):
            lat[(v, r["qid"])] = r
    rows = []  # per v16 answered question: dict of signals + outcome
    modes = Counter()
    for (v, q), r in lat.items():
        if v != "16":
            continue
        s = (r.get("systems", {}) or {}).get("pipeline_auto") or {}
        a = s.get("answer") or {}
        o = s.get("outcome") or "abstain"
        if o not in ("correct", "wrong", "abstain"):
            o = "abstain"
        raw = a.get("confidence") or 0.0
        dm = str(a.get("decision_mode") or "")
        rat = str(a.get("rationale") or "")
        gs = a.get("gene_scores") or {}
        vals = sorted((float(x) for x in gs.values()), reverse=True)
        margin = (vals[0]-vals[1]) if len(vals) >= 2 else (vals[0] if vals else 0.0)
        agree = "v11 一致" in rat
        judge = ("+judge" in dm) or ("法官" in rat)
        used_llm = bool(a.get("used_llm"))
        modes[dm] += 1
        rows.append(dict(q=q, bucket=r.get("bucket"), outcome=o, raw=float(raw),
                         margin=float(margin), agree=agree, judge=judge,
                         used_llm=used_llm, dm=dm, rat=rat,
                         cthr=a.get("conformal_threshold"), winner=a.get("winner"),
                         answer=a.get("answer"), gt=r.get("ground_truth")))
    return rows, modes


def main():
    rows, modes = build_signals()
    n = len(rows)
    print(f"# v16-auto 已存 {n} 题；decision_mode 分布：")
    for dm, c in modes.most_common():
        print(f"    {c:>3}  {dm}")

    # PART A: 信号 AUC-RC / ECE 对比
    def as_pairs(key):
        return [(float(r[key]), r["outcome"]) for r in rows]

    # 派生信号：agree/judge/heuristic 分层 + gene margin
    def tier(r):
        base = 0.5
        if r["agree"]:
            base = 0.85          # 双通道同向：最强
        elif r["judge"]:
            base = 0.62          # 法官裁定
        elif not r["used_llm"]:
            base = 0.70          # 纯启发式(证据接地) > LLM 自报
        else:
            base = 0.5           # LLM 主审自报：最弱
        return base + 0.15*norm(r["margin"], 0, 1.0)

    sigs = {
        "raw_conf":    as_pairs("raw"),
        "gene_margin": [(r["margin"], r["outcome"]) for r in rows],
        "tier":        [(tier(r), r["outcome"]) for r in rows],
        "tier×raw":    [(tier(r) * max(r["raw"], 0.05), r["outcome"]) for r in rows],
        "tier+margin": [(tier(r)+0.3*norm(r["margin"], 0, 1.0), r["outcome"]) for r in rows],
    }
    # 基线（同函数、公平）
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    bucket_of = {i["qid"]: i["bucket"] for i in inputs}
    for sysname in ("autogen", "mdagents"):
        p = TRACES / f"answers_{sysname}.jsonl"
        seen = {}
        if p.exists():
            for rr in load_jsonl(p):
                seen[rr["qid"]] = rr
        pr = []
        for q, rr in seen.items():
            o = outcome_of(rr, gt.get(q, {}), bucket_of.get(q, ""))
            if o in ("correct", "wrong"):
                pr.append((float(rr.get("confidence") or 0.0), o))
        sigs[sysname] = pr

    print("\n# PART A  信号 AUC-RC↓ / ECE↓（v16 answered + 基线）")
    print(f"  {'signal':12s} {'n':>4} {'AUC-RC':>8} {'ECE':>7}")
    for name, pr in sorted(sigs.items(), key=lambda kv: auc_rc(kv[1])[0] if kv[1] else 9):
        auc, _ = auc_rc(pr)
        print(f"  {name:12s} {len(pr):>4} {auc:>8.4f} {ece(pr):>7.3f}")

    # PART B: 错因解剖
    wrong = [r for r in rows if r["outcome"] == "wrong"]
    print(f"\n# PART B  v16-auto 答错 {len(wrong)} 题归类")
    byb = defaultdict(Counter)
    for r in wrong:
        tag = "agree" if r["agree"] else ("judge" if r["judge"] else
              ("heuristic" if not r["used_llm"] else "llm-chief"))
        byb[r["bucket"]][tag] += 1
    for b, cnt in byb.items():
        print(f"  [{b}] n={sum(cnt.values())}: " + ", ".join(f"{k}={v}" for k, v in cnt.most_common()))
    print("\n  逐题（bucket | 路径 | conf | 预测 vs GT | rationale头）：")
    for r in sorted(wrong, key=lambda x: (str(x["bucket"]), -x["raw"])):
        pred = r["winner"] if r["bucket"] == "pairwise" else r["answer"]
        gtv = r["gt"].get("winner") if r["bucket"] == "pairwise" else r["gt"].get("answer")
        head = " ".join(r["rat"][:110].split())
        print(f"   {r['bucket'][:5]:5s} {'agree' if r['agree'] else ('judge' if r['judge'] else ('heur' if not r['used_llm'] else 'llm')):5s} "
              f"c={r['raw']:.2f} pred={pred} gt={gtv} | {head}")


if __name__ == "__main__":
    main()
