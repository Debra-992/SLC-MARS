# -*- coding: utf-8 -*-
"""温度0 R2 开关 A/B 判决表（按 数据集 A=全部 / B=防记忆 分列）。

两臂（都在 SLC_FORCE_TEMP0=1 下重算，唯一变量=SLC_MEM_R2）：
  armA = R2 OFF（trace_version=a0）  armB = R2 ON（trace_version=b0）
只用 pipeline_auto 臂（R2 挂在此臂的 membership 启发式兜底路径）。只读 tracestore，不触端点。
判据：B(防记忆)∩membership 的净 fix 才是 R2 真·泛化弹着；pairwise 两臂应≈0 变化（R2 不该动）。
用法: .venv\\Scripts\\python.exe tools\\_final_r2_t0.py
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
from knowledge_agents.baselines.common import INPUT_PATH, TRACES  # noqa: E402
from knowledge_agents.baselines.merge_baselines import load_jsonl  # noqa: E402

ARM = "pipeline_auto"
V_OFF, V_ON = "a0", "b0"
R2_MARK = "[R2 必需性重排"


def trace_by_version(version: str) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        if str((r.get("provenance") or {}).get("trace_version") or "") == version:
            latest[r["qid"]] = r
    return latest


def outcome(store: dict[str, dict], q: str) -> str:
    r = store.get(q)
    if not r:
        return "missing"
    oc = (r.get("systems", {}).get(ARM) or {}).get("outcome")
    return oc if oc in ("correct", "wrong", "abstain") else "abstain"


def metrics(store: dict[str, dict], keep) -> dict:
    c = w = a = m = 0
    for q in keep:
        o = outcome(store, q)
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


def fmt(v):
    return f"{v:.1%}" if v is not None and not (isinstance(v, float) and math.isnan(v)) else "-"


def main() -> None:
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    bucket_of = {i["qid"]: i["bucket"] for i in inputs}
    a_qids = set(json.loads((TRACES / "sample_manifest.json").read_text(encoding="utf-8"))["qids"])
    b_path = TRACES / "dataset_b_qids.json"
    b_qids = (set(json.loads(b_path.read_text(encoding="utf-8"))["b_qids"]) & a_qids) if b_path.exists() else set()

    off = trace_by_version(V_OFF)
    on = trace_by_version(V_ON)
    n_off = sum(1 for q in a_qids if q in off)
    n_on = sum(1 for q in a_qids if q in on)
    memb = {q for q in a_qids if bucket_of.get(q) == "membership"}
    pw = a_qids - memb

    print(f"# 温度0 R2 开关 A/B   A(全部)=n{len(a_qids)}  B(防记忆)=n{len(b_qids)}  "
          f"覆盖 armA(R2off)={n_off} armB(R2on)={n_on}")
    if n_off < len(a_qids) or n_on < len(a_qids):
        print(f">>> 注意：某臂未跑满 100（off {n_off}/on {n_on}），以下为部分快照，勿定论。")
    print()

    def block(label, keep):
        print(f"===== {label} (n={len(keep)}) =====")
        print(f"  {'切片':<12} {'R2off 覆盖/acc全/acc答':<30} {'R2on 覆盖/acc全/acc答':<30} {'Δacc全':>8}")
        for sub, sset in (("总体", keep),
                          ("membership", keep & memb),
                          ("pairwise", keep & pw)):
            if not sset:
                continue
            mo, mn = metrics(off, sset), metrics(on, sset)
            dl = (mn["acc_all"] - mo["acc_all"]) if (mo["n"] and mn["n"]) else float("nan")
            cv_o = f"{mo['cov']:.0%}" if mo["n"] else "-"
            cv_n = f"{mn['cov']:.0%}" if mn["n"] else "-"
            l_ = f"{cv_o}/{fmt(mo['acc_all'])}/{fmt(mo['acc_ans'])}"
            r_ = f"{cv_n}/{fmt(mn['acc_all'])}/{fmt(mn['acc_ans'])}"
            d_ = f"{dl*100:>+7.1f}" if not math.isnan(dl) else "     -"
            print(f"  {sub:<12} {l_:<30} {r_:<30} {d_}")
        print()

    block("A 全部", a_qids)
    block("B 防记忆", b_qids)

    # 配对翻转（R2off -> R2on）：only where both arms present
    def pair(keep):
        both = [q for q in keep if q in off and q in on]
        w2c = sum(1 for q in both if outcome(off, q) == "wrong" and outcome(on, q) == "correct")
        c2w = sum(1 for q in both if outcome(off, q) == "correct" and outcome(on, q) == "wrong")
        return both, w2c, c2w
    print("===== membership 配对翻转（R2off -> R2on）=====")
    ab, aw, ac = pair(a_qids & memb)
    bb, bw, bc = pair(b_qids & memb)
    r2_fired = {q for q in memb if R2_MARK in str((on.get(q, {}).get("systems", {}).get(ARM, {})
                    .get("answer", {}) or {}).get("rationale") or "")}
    print(f"  A∩mem n={len(ab)}: w2c {aw} | c2w {ac} | 净 {aw-ac:+d}")
    print(f"  B∩mem n={len(bb)}: w2c {bw} | c2w {bc} | 净 {bw-bc:+d}   ← 真·泛化弹着(防记忆)")
    print(f"  R2 触发(arm B, rationale 含标记)={len(r2_fired)}")
    pb, pw2c, pwc = pair(pw)
    print(f"  [sanity] pairwise 变化 c->w={pwc} w->c={pw2c}（R2 不该动 pairwise，应≈0）")
    verdict = "POSITIVE(R2 在防记忆集净增益)" if (bw - bc) > 0 else \
        ("NEUTRAL(R2 无显著净益)" if (bw - bc) == 0 else "NEGATIVE(R2 反而伤防记忆)")
    print(f"  判据(B∩mem 净 {bw-bc:+d}): {verdict}")


if __name__ == "__main__":
    main()
