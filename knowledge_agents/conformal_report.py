# -*- coding: utf-8 -*-
"""保形弃答校准与风险-覆盖评测：从 tracestore 读每题(置信度, 对错)，按 qid 切成【不相交】的
校准集 / 测试集，在校准集上算 conformal 阈值，再在测试集上报告 覆盖率 / 已答错误率，
对比"未加保形的基线操作点" vs"保形操作点"，并导出可直接喂给管线的阈值文件。

无 LLM / 无工具依赖：只消费已缓存的 trace（compare_eval 产出的 tracestore.jsonl）。

用法：
  python -m knowledge_agents.conformal_report                 # 默认对 3 系统 × 2 桶出曲线
  python -m knowledge_agents.conformal_report --system pipeline_heuristic --alpha 0.10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from knowledge_agents import conformal as cf
from knowledge_agents.run_single_question import ROOT

SYSTEMS = ("pipeline_heuristic", "pipeline_auto", "baseline")
BUCKETS = ("pairwise", "membership")
DEFAULT_TRACES = ROOT / "outputs" / "traces" / "tracestore.jsonl"
ALPHA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def _load(traces_path: Path, version: str | None) -> list[dict]:
    rows: list[dict] = []
    with traces_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("error"):
                continue
            if version and (e.get("provenance") or {}).get("trace_version") != version:
                continue
            rows.append(e)
    return rows


def _in_calibration(qid: str, seed: int, frac: float) -> bool:
    h = int(hashlib.sha1(f"{seed}|{qid}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def _pred_from(entry: dict, system: str) -> cf.Pred | None:
    """从某系统的答案抽取 (score=1-conf, loss=答错?1:0)。仅当该系统"已作答"才有效。"""
    sysobj = entry.get("systems", {}).get(system) or {}
    outcome = sysobj.get("outcome")
    if outcome not in ("correct", "wrong"):
        return None
    conf = float((sysobj.get("answer") or {}).get("confidence", 0.0) or 0.0)
    return cf.Pred(score=cf.nonconformity(conf), loss=1 if outcome == "wrong" else 0)


def _split(rows: list[dict], system: str, bucket: str, seed: int, frac: float):
    calib: list[cf.Pred] = []
    test: list[cf.Pred] = []
    test_total = 0
    for e in rows:
        if e.get("bucket") != bucket:
            continue
        is_cal = _in_calibration(e["qid"], seed, frac)
        if not is_cal:
            test_total += 1              # 该桶测试集全体（含基线本身弃答的题）
        p = _pred_from(e, system)
        if p is None:
            continue
        (calib if is_cal else test).append(p)
    return calib, test, test_total


def _fmt_curve(pairs: list[cf.Pred], alphas: list[float], n_total: int) -> list[str]:
    lines = ["  | α(目标·全体口径) | τ̂ | 覆盖率 | 已答错误率 | 错/总 |",
             "  |---:|---:|---:|---:|---:|"]
    for a in alphas:
        tau = cf.calibrate(pairs, a)
        r = cf.apply_threshold(pairs, tau, n_total)
        tautxt = "—" if tau is None else f"{tau:.3f}"
        lines.append(f"  | {a:.2f} | {tautxt} | {r['coverage']:.1%} | {r['risk']:.1%} | "
                     f"{r['wrong']}/{r['n_total']} |")
    return lines


def main() -> None:
    # Windows 控制台默认 GBK，报告含 ⇒/≥ 等符号会 UnicodeEncodeError；
    # 先把 stdout 切成 UTF-8（无法编码字符以 ? 兜底），产物文件本就按 utf-8 落盘。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(DEFAULT_TRACES))
    ap.add_argument("--version", default=None, help="仅用该 trace_version（如 6）")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--alpha", type=float, default=0.10, help="写入阈值文件的目标全体口径风险")
    ap.add_argument("--system", default="pipeline_heuristic",
                    choices=list(SYSTEMS) + ["all"], help="为哪个系统导出部署阈值")
    args = ap.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"[保形] 找不到 trace 文件：{traces_path}")
        return
    rows = _load(traces_path, args.version)
    print("=" * 84)
    print(f"[保形弃答校准] trace 条数={len(rows)} version={args.version or '任意'} "
          f"校准比例={args.calib_frac} seed={args.seed} 部署 α={args.alpha}")

    thresholds_out: dict[str, float] = {}
    report: list[str] = ["[保形弃答] 校准/测试按 qid 不相交切分；覆盖率以测试全体为分母(含基线弃答)；"
                         "保证 E[错答且选中]≤α ⇒ 已答错误率≤α/覆盖率", ""]

    systems = SYSTEMS if args.system == "all" else (args.system,)
    for system in systems:
        report.append(f"### 系统：{system}")
        for bucket in BUCKETS:
            calib, test, test_total = _split(rows, system, bucket, args.seed, args.calib_frac)
            n_all = test_total if test_total else len(test)
            base = cf.apply_threshold(test, tau=1.0, n_total=n_all)  # 全接受=未加保形的基线操作点
            report.append(f"\n- {bucket}  校准 n={len(calib)} / 测试 n={n_all} "
                          f"(已答 {len(test)}，基线本身弃答 {n_all - len(test)})")
            report.append(f"  基线(全接受)：覆盖率={base['coverage']:.1%} 已答错误率={base['risk']:.1%}")
            if calib:
                tau = cf.calibrate(calib, args.alpha)
                r = cf.apply_threshold(test, tau, n_all)
                conf_thr = None if tau is None else round(1.0 - tau, 4)
                report.append(f"  保形(α={args.alpha}, τ̂={'—' if tau is None else format(tau,'.3f')}"
                              f" ⇒ conf≥{'—' if conf_thr is None else conf_thr})："
                              f"覆盖率={r['coverage']:.1%} 已答错误率={r['risk']:.1%} "
                              f"[错/总={r['wrong']}/{r['n_total']}]")
                if system == (args.system if args.system != "all" else "pipeline_heuristic") and conf_thr is not None:
                    thresholds_out[bucket] = conf_thr
            if calib:
                report += ["  风险-覆盖曲线（校准自洽，示意单调权衡）："] + _fmt_curve(calib, ALPHA_GRID, len(calib))
            else:
                report.append("  （校准集为空：无法拟合阈值，保持原判定）")
            # ECE（置信度是否名副其实）—— 用测试已答样本
            cc = [(1.0 - p.score, p.loss) for p in test]
            if cc:
                report.append(f"  ECE(测试已答, 置信 vs 实际)={cf.ece(cc):.3f}")
        report.append("")

    text = "\n".join(report)

    # 先落盘（产物生成绝不依赖控制台编码），再打印
    out_dir = traces_path.parent
    (out_dir / "conformal_report.md").write_text(text, encoding="utf-8")
    thresholds_path = out_dir / "conformal_thresholds.json"
    if thresholds_out:
        payload = {"alpha": args.alpha, "system": args.system, "seed": args.seed,
                   "calib_frac": args.calib_frac, "thresholds": thresholds_out,
                   "note": "conf<threshold -> 弃答；由 SLC_CONFORMAL_FILE 载入决策层"}
        thresholds_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(text)
    if thresholds_out:
        print(f"\n阈值文件: {thresholds_path}  thresholds={thresholds_out}")
    else:
        print(f"\n（α={args.alpha} 下无可认证阈值：校准集过小/风险下界 1/(N+1) 超过 α -> 保形正确地选择全弃答）")
    print(f"报告: {out_dir / 'conformal_report.md'}")


if __name__ == "__main__":
    main()
