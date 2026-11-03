# -*- coding: utf-8 -*-
"""③ 新口径报表：Risk-Coverage 曲线 / AUC / 选择性弃权 θ 模拟 / 校准误差(ECE)。

背景：全体口径惩罚诚实弃权、已答口径奖励硬答——两个口径都只反映"工作点"的一个切面。
RC 曲线把弃权阈值 θ 变成连续旋钮：横轴=覆盖率（按置信度降序作答的比例），纵轴=风险
（该覆盖下的错误率）。曲线下面积(AUC-RC)越低 = 系统的"不确定性质量"越高——知道自己
什么时候在猜。这是管线（conf 有证据语义）对裸 LLM 基线（conf 是口头票）的独有卖点。

用法: .venv\\Scripts\\python.exe tools\\_risk_cover.py [trace_version]
     默认对 16 与 18 两个臂 + autogen/mdagents 出表；任意 θ 直接模拟，无需重跑。
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


def load_rows(versions: list[str], systems: list[str], qids: set[str],
              bucket_of: dict, gt: dict) -> dict[str, list[dict]]:
    """arm 名 -> [{outcome, conf}]。管线臂取 tracestore(auto)，基线臂取 answers_*.jsonl。"""
    out: dict[str, list[dict]] = {}
    latest: dict[tuple, dict] = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        v = str((r.get("provenance") or {}).get("trace_version") or "")
        if v in versions:
            latest[(v, r["qid"])] = r
    for v in versions:
        for sysname in systems:
            arm = f"v{v}-{sysname.replace('pipeline_', '')}"
            rows = []
            for q in qids:
                r = latest.get((v, q))
                if not r:
                    continue
                s = (r.get("systems", {}) or {}).get(sysname) or {}
                a = s.get("answer") or {}
                o = s.get("outcome") or "abstain"
                try:
                    conf = float(a.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
                rows.append({"outcome": o if o in ("correct", "wrong", "abstain") else "abstain",
                             "conf": conf})
            out[arm] = rows
    for sysname in ("autogen", "mdagents"):
        p = TRACES / f"answers_{sysname}.jsonl"
        if not p.exists():
            continue
        seen: dict[str, dict] = {}
        for r in load_jsonl(p):
            seen[r["qid"]] = r
        rows = []
        for q in qids:
            r = seen.get(q)
            if not r:
                continue
            o = outcome_of(r, gt.get(q, {}), bucket_of.get(q, ""))
            try:
                conf = float(r.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            rows.append({"outcome": o if o in ("correct", "wrong", "abstain") else "abstain",
                         "conf": conf})
        out[sysname] = rows
    return out


def rc_curve(rows: list[dict]) -> tuple[list[tuple[float, float]], float]:
    """按 conf 降序逐题纳入作答 => (coverage, risk=错误率) 锚点列 + AUC-RC（越小越好）。"""
    n = len(rows)
    if not n:
        return [], math.nan
    answered = [r for r in rows if r["outcome"] != "abstain"]
    answered.sort(key=lambda r: -r["conf"])
    pts, err = [], 0
    for i, r in enumerate(answered, 1):
        if r["outcome"] == "wrong":
            err += 1
        pts.append((i / n, err / i))
    if len(pts) >= 2:
        auc = sum((pts[k][0] - pts[k - 1][0]) * (pts[k][1] + pts[k - 1][1]) / 2
                  for k in range(1, len(pts)))
    else:
        auc = math.nan
    return pts, auc


def theta_sim(rows: list[dict], theta: float) -> dict:
    """模拟选择性弃权工作点 θ：conf<θ 一律视为弃答。返回三指标。"""
    n = len(rows)
    c = w = ab = 0
    for r in rows:
        if r["outcome"] == "abstain" or (r["outcome"] != "abstain" and r["conf"] < theta):
            ab += 1
        elif r["outcome"] == "correct":
            c += 1
            w += 0
        else:
            w += 1
    ans = c + w
    return dict(cov=ans / n if n else 0.0,
                acc_all=c / n if n else 0.0,
                acc_ans=c / ans if ans else 0.0)


def ece(rows: list[dict], bins: int = 10) -> float:
    """期望校准误差：只算已答题（conf 应≈P(对|conf)）。"""
    answered = [r for r in rows if r["outcome"] != "abstain"]
    if not answered:
        return math.nan
    tot = len(answered)
    acc = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sub = [r for r in answered if lo <= r["conf"] < hi or (hi == 1.0 and r["conf"] == 1.0)]
        if not sub:
            continue
        conf_bar = sum(r["conf"] for r in sub) / len(sub)
        acc_bar = sum(1 for r in sub if r["outcome"] == "correct") / len(sub)
        acc += len(sub) / tot * abs(conf_bar - acc_bar)
    return acc


def main() -> None:
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((TRACES / "sample_manifest.json").read_text(encoding="utf-8"))
    qids = set(manifest["qids"])
    bucket_of = {i["qid"]: i["bucket"] for i in inputs}
    versions = sys.argv[1:] or ["16", "18"]
    arms = load_rows(versions, ["pipeline_auto"], qids, bucket_of, gt)
    print(f"# Risk-Coverage / 选择性弃权分析  A n={len(qids)}  版本={versions}")
    print(f"{'arm':10s} {'n':>4} {'AUC-RC↓':>8} {'ECE↓':>7}  工作点(θ模拟): θ=0 全体 | 最优θ*")
    for arm, rows in arms.items():
        if not rows:
            print(f"{arm:10s}    0        -       -  (无行)")
            continue
        pts, auc = rc_curve(rows)
        base = theta_sim(rows, 0.0)
        best = None
        for th in [round(x * 0.05, 2) for x in range(1, 17)]:
            s = theta_sim(rows, th)
            if best is None or s["acc_all"] > best[1]["acc_all"]:
                best = (th, s)
        e = ece(rows)
        print(f"{arm:10s} {len(rows):>4} {auc:>8.4f} {e:>7.3f}  θ=0: {base['acc_all']:.1%}"
              f" (cov {base['cov']:.0%}) | θ*={best[0]:.2f}: {best[1]['acc_all']:.1%}"
              f" (cov {best[1]['cov']:.0%}, 已答口径 {best[1]['acc_ans']:.1%})")
    # RC 锚点：管线各臂 vs 基线的 80%/90% 覆盖处风险
    print("\nRC 锚点(覆盖率截断处的已答错误率↓):")
    print(f"{'arm':10s} {'cov@100%':>9} {'@90%':>7} {'@80%':>7} {'@70%':>7}")
    for arm, rows in arms.items():
        pts, _ = rc_curve(rows)
        if not pts:
            continue
        def at(target):
            cand = [(c, r) for c, r in pts if c <= target + 1e-9]
            return f"{cand[-1][1]:.1%}" if cand else "-"
        print(f"{arm:10s} {at(1.0):>9} {at(0.9):>7} {at(0.8):>7} {at(0.7):>7}")
    print("\n注：AUC-RC/ECE 越低越好；基线从不弃权 => 其曲线起点即全部题，conf 为自报口头置信度，")
    print("    与管线证据置信度的可比性本身即是『不确定性质量』的比较。")


if __name__ == "__main__":
    main()
