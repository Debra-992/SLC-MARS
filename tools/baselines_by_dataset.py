# -*- coding: utf-8 -*-
"""基线 × A/B 数据集 分层评测（只读，不触端点；不动 pipeline）。

数据集切分沿用"之前的双探针防记忆规则"：
  A = 全部 489（baselines_input.json 全量 qid）
  B = 防记忆子集 = dataset_b_qids_all489.json["b_qids"] ∩ A   （B := novel ∪ unstable）
五个 baseline（pure_llm/lit_rag/autogen/mdagents/metagpt）各自在同一 A、同一 B、并按桶
（membership / pairwise）分别报：acc已答、acc全、覆盖率。覆盖率<95% 标 [快照·未定版]。
用法: .venv\\Scripts\\python.exe tools\\baselines_by_dataset.py
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

SYSTEMS = ["pure_llm", "lit_rag", "autogen", "mdagents", "metagpt"]
B_PATH = TRACES / "dataset_b_qids_all489.json"
REPORT = TRACES / "baselines_by_dataset_report.txt"


def answers_outcome(system: str, qids: set, bucket_of: dict, gt: dict) -> dict[str, str]:
    rows: dict[str, dict] = {}
    p = TRACES / f"answers_{system}.jsonl"
    if p.exists():
        for r in load_jsonl(p):
            rows[r["qid"]] = r            # 同 qid 后写覆盖先写（pending 口径）
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
    return dict(n=n, c=c, w=w, ab=a, miss=m, tot=len(keep),
                acc_all=(c / len(keep) if keep else math.nan),
                acc_ans=(c / ans if ans else math.nan),
                cov=(ans / len(keep) if keep else math.nan))


def fmt(v, pct=True):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def main() -> None:
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    bucket_of = {i["qid"]: i["bucket"] for i in inputs}
    a_qids = set(bucket_of)
    b_qids = (set(json.loads(B_PATH.read_text(encoding="utf-8"))["b_qids"]) & a_qids) if B_PATH.exists() else set()
    memb = {q for q in a_qids if bucket_of[q] == "membership"}
    pw = a_qids - memb

    lines: list[str] = []
    def emit(s: str = ""):
        print(s, flush=True)
        lines.append(s)

    if not b_qids:
        emit(f">>> 缺 B 清单（{B_PATH.name}）——双探针标注未完成，仅先出 A(全部) 快照。")
    emit(f"# 基线 × A/B 数据集分层评测    A(全部)=n{len(a_qids)}  B(防记忆)=n{len(b_qids)}  "
         f"(A 桶: mem={len(memb)} pw={len(pw)})")
    emit(f"# 注: 覆盖<95% 标 [快照·未定版]；acc已答=correct/(correct+wrong)，acc全=correct/子集全体。")
    emit()

    for sysname in SYSTEMS:
        per_q = answers_outcome(sysname, a_qids, bucket_of, gt)
        mA = metrics(per_q, a_qids)
        tag = "" if mA["cov"] >= 0.95 else "  [快照·未定版]"
        emit(f"===== baseline: {sysname}{tag} =====")
        hdr = f"  {'切片':<14} {'n':>4} {'✓':>3} {'✗':>3} {'ab':>3} {'缺':>3}  {'覆盖':>6}  {'acc全':>7}  {'acc已答':>7}"
        emit(hdr)
        slices = [("A·全部", a_qids), ("A·membership", a_qids & memb), ("A·pairwise", a_qids & pw)]
        if b_qids:
            bA = b_qids
            slices += [("B·防记忆", bA), ("B·membership", bA & memb), ("B·pairwise", bA & pw)]
        for label, keep in slices:
            mm = metrics(per_q, keep)
            cov = f"{mm['cov']:.0%}" if mm["tot"] else "-"
            emit(f"  {label:<14} {mm['tot']:>4} {mm['c']:>3} {mm['w']:>3} {mm['ab']:>3} {mm['miss']:>3}"
                 f"  {cov:>6}  {fmt(mm['acc_all']):>7}  {fmt(mm['acc_ans']):>7}")
        emit()

    # B 相对 A 的增益/落差速览（acc已答）：防记忆集最能体现"记忆 vs 推理"
    if b_qids:
        emit("===== B(防记忆) vs A(全部)  acc已答 落差速览（越负=越依赖记忆）=====")
        emit(f"  {'system':<10} {'A已答acc':>9} {'B已答acc':>9} {'Δ(B-A)':>8}  {'B覆盖':>6}")
        for sysname in SYSTEMS:
            per_q = answers_outcome(sysname, a_qids, bucket_of, gt)
            aA, bB = metrics(per_q, a_qids), metrics(per_q, b_qids)
            d = (bB["acc_ans"] - aA["acc_ans"]) if (not math.isnan(aA["acc_ans"]) and not math.isnan(bB["acc_ans"])) else float("nan")
            ds = f"{d*100:+.1f}" if not math.isnan(d) else "-"
            emit(f"  {sysname:<10} {fmt(aA['acc_ans']):>9} {fmt(bB['acc_ans']):>9} {ds:>8}  {fmt(bB['cov']):>6}")
        emit()

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit(f"[written] {REPORT}")


if __name__ == "__main__":
    main()
