# -*- coding: utf-8 -*-
"""效率—精度 总表：5 基线 + 管线 v20，全 489，A(全部)/B(防记忆) × membership/pairwise，含 token/题。

只读，不触端点。数据源：
  - 基线精度：outputs/traces/answers_<sys>.jsonl（同 qid 后写覆盖），outcome_of 判分。
  - 基线 token：同文件顶层 tokens.total（按 qid 去重后取有 token 行的均值）。
    注意 autogen 因自带磁盘缓存(cache_seed) 会漏计（标 *）。
  - 管线精度：tracestore.jsonl 里 provenance.trace_version=="20"（v500 冠军 v20 全 489）。
  - 管线 token：pipeline_token_probe.json（独立探针实测；缺则标"待探针"）。
数据集：A=489（baselines_input 全量）；B=dataset_b_qids_all489.json ∩ A。
用法: .venv\\Scripts\\python.exe tools\\_final_master_table.py
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

BASELINES = ["pure_llm", "lit_rag", "autogen", "mdagents", "metagpt"]
PIPE_VERSION = "20"          # v500 冠军 v20 全 489（默认温度）
PIPE_MODES = [("pipeline_heuristic", "管线v20·heuristic"), ("pipeline_auto", "管线v20·auto")]
B_PATH = TRACES / "dataset_b_qids_all489.json"
PROBE_PATH = TRACES / "pipeline_token_probe.json"
STORE_PATH = TRACES / "tracestore.jsonl"
REPORT = TRACES / "efficiency_accuracy_master.txt"


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v*100:.1f}%"


def base_rows(system):
    """同 qid 后写覆盖，返回 {qid: row}。"""
    rows = {}
    p = TRACES / f"answers_{system}.jsonl"
    if p.exists():
        for r in load_jsonl(p):
            rows[r["qid"]] = r
    return rows


def metrics(out_by_q, keep):
    c = w = a = m = 0
    for q in keep:
        o = out_by_q.get(q, "missing")
        if o == "missing":
            m += 1
        elif o == "correct":
            c += 1
        elif o == "wrong":
            w += 1
        else:
            a += 1
    ans = c + w
    tot = len(keep)
    return dict(c=c, w=w, ab=a, miss=m, tot=tot, ans=ans,
                acc_ans=(c / ans if ans else math.nan),
                acc_all=(c / tot if tot else math.nan),
                cov=(ans / tot if tot else math.nan))


def token_per_q(rows):
    vals = [r["tokens"]["total"] for r in rows.values()
            if isinstance(r.get("tokens"), dict) and r["tokens"].get("total")]
    return (sum(vals) / len(vals)) if vals else math.nan


def trace_by_version(version):
    latest = {}
    if not STORE_PATH.exists():
        return latest
    for line in STORE_PATH.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r.get("provenance") or {}).get("trace_version") == version:
            latest[r["qid"]] = r
    return latest


def pipe_out_by_q(store, mode):
    return {q: (e.get("systems", {}).get(mode, {}) or {}).get("outcome", "missing")
            for q, e in store.items()}


def main():
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    bucket_of = {i["qid"]: i["bucket"] for i in inputs}
    a_qids = set(bucket_of)
    b_qids = (set(json.loads(B_PATH.read_text(encoding="utf-8"))["b_qids"]) & a_qids) if B_PATH.exists() else set()
    memb = {q for q in a_qids if bucket_of[q] == "membership"}
    pw = a_qids - memb
    memb_b, pw_b = (b_qids & memb), (b_qids & pw)

    probe = json.loads(PROBE_PATH.read_text(encoding="utf-8")) if PROBE_PATH.exists() else None
    pipe_tok = probe["mean_total"] if probe else None
    pipe_calls = probe.get("mean_calls") if probe else None
    pipe_bsplit = probe.get("mean_total_by_bucket") if probe else None
    pipe_nsplit = probe.get("n_by_bucket") if probe else None

    L = []
    def emit(s=""):
        print(s, flush=True); L.append(s)

    emit("# 效率—精度 总表（全 489）   主口径 acc已答=correct/(correct+wrong)；acc全=correct/子集全体")
    emit(f"# A(全部)=n{len(a_qids)} (mem={len(memb)} pw={len(pw)})   "
         f"B(防记忆)=n{len(b_qids)} (mem={len(memb_b)} pw={len(pw_b)})   覆盖<95% 标 †")
    if probe:
        _bs = "  ".join(f"{k}≈{v}" + (f"(n{pipe_nsplit.get(k)})" if pipe_nsplit else "")
                        for k, v in (pipe_bsplit or {}).items())
        emit(f"# 管线 token/题：独立探针实测 有效 n={probe.get('valid_n', probe.get('ok_n','?'))}/{probe.get('probe_n','?')}"
             f"（含 supply+heuristic+auto 两次 decide，不含 MetaGPT；calls==0 端点废行已剔除）"
             + (f"；分桶 {_bs}" if _bs else "") + "。基线 token/题取各 answers_*.jsonl 去重均值。")
    else:
        emit("# 管线 token/题：待探针（pipeline_token_probe.json 未生成）→ 该列显示 '待探针'。")
    emit()

    slices = [("A·全部", a_qids), ("A·mem", memb), ("A·pw", pw),
              ("B·全部", b_qids), ("B·mem", memb_b), ("B·pw", pw_b)]

    def emit_table(title, key):
        emit(f"===== {title} =====")
        head = f"  {'系统':<20}" + "".join(f"{lbl:>9}" for lbl, _ in slices) + f"{'Δ(B-A)':>9}{'token/题':>10}"
        emit(head)
        for sysname in BASELINES:
            rows = base_rows(sysname)
            out_by_q = {q: (outcome_of(rows[q], gt.get(q, {}), bucket_of[q]) if q in rows else "missing") for q in a_qids}
            ms = [metrics(out_by_q, keep) for _, keep in slices]
            covA = ms[0]["cov"]
            mark = "†" if covA < 0.95 else " "
            tpq = token_per_q(rows)
            tps = ("~%.0f" % tpq) if not math.isnan(tpq) else "-"
            d = (ms[3]["acc_ans"] - ms[0]["acc_ans"]) if not (math.isnan(ms[3]["acc_ans"]) or math.isnan(ms[0]["acc_ans"])) else math.nan
            cells = "".join(f"{fmt(m[key]):>9}" for m in ms)
            emit(f"  {sysname:<19}{mark}{cells}{(f'{d*100:+.1f}' if not math.isnan(d) else '-'):>9}{tps:>10}")
        store = trace_by_version(PIPE_VERSION)
        done = set(store)
        for mode, disp in PIPE_MODES:
            out_by_q = pipe_out_by_q(store, mode)
            ms = [metrics(out_by_q, keep & done) for _, keep in slices]
            covA = ms[0]["cov"]
            mark = "†" if covA < 0.95 else " "
            d = (ms[3]["acc_ans"] - ms[0]["acc_ans"]) if not (math.isnan(ms[3]["acc_ans"]) or math.isnan(ms[0]["acc_ans"])) else math.nan
            tps = ("~%d" % pipe_tok) if pipe_tok else "待探针"
            cells = "".join(f"{fmt(m[key]):>9}" for m in ms)
            emit(f"  {disp:<19}{mark}{cells}{(f'{d*100:+.1f}' if not math.isnan(d) else '-'):>9}{tps:>10}")
        emit(f"  （管线 v20 已算 {len(done)}/{len(a_qids)}；token/题列：管线=探针均值{' calls≈%s' % pipe_calls if pipe_calls else ''}）")
        emit()

    emit_table("acc已答（主口径）", "acc_ans")
    emit_table("acc全（弃/缺计错，惩罚弃权刷分）", "acc_all")

    emit("注：†=该系统 A 覆盖<95%（样本不足，慎作定论，尤其 mdagents 仅答部分题）；")
    emit("    autogen token 已审计为准确值（最终489跑批前已设 cache_seed=None 关磁盘缓存；")
    emit("    489 题仅 4 行 0-token=error 题被剔除，485 题 calls=2/题 全真实计量，见 tools/_autogen_tok_audit.py）；Δ(B-A) 越负=越依赖记忆；")
    emit("    管线 token/题来自小样本探针，为估计值（已标探针 n）。")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    emit(f"[written] {REPORT}")


if __name__ == "__main__":
    main()
