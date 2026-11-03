# -*- coding: utf-8 -*-
"""管线 v16 / v20 在全 489 上的分层战绩（A=全部 / B=防记忆 × membership/pairwise）。

两臂（都用冠军基座 + 默认温度、非 temp0，反映生产配置）：
  v16 = SLC_MEM_R2=0（trace_version=16f489）   v20 = SLC_MEM_R2=1（trace_version=20f489）
两个决策模式都报：pipeline_heuristic（可弃答）与 pipeline_auto（LLM 类比兜底·尽量不弃答）。
只读 tracestore，不触端点。A/B 口径与 baselines_by_dataset.py 对齐（A=INPUT_PATH 全 489，B=dataset_b_qids_all489）。
用法: .venv\\Scripts\\python.exe tools\\_final_pipeline_full489.py
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
from knowledge_agents.baselines.common import INPUT_PATH, TRACES, GT_PATH  # noqa: E402
from knowledge_agents.baselines.merge_baselines import load_jsonl, outcome_of  # noqa: E402

MODES = ("pipeline_heuristic", "pipeline_auto")
# v500 扩489 把冠军 v20 写进版本 "20"（默认温度）；v16 仅 100-A 旧缓存作参照。
VERSIONS = {"v20 冠军(R2 on, v500扩489中)": "20"}


def trace_by_version(version: str) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        if str((r.get("provenance") or {}).get("trace_version") or "") == version:
            latest[r["qid"]] = r
    return latest


def outcome(store: dict[str, dict], q: str, mode: str) -> str:
    r = store.get(q)
    if not r:
        return "missing"
    oc = (r.get("systems", {}).get(mode) or {}).get("outcome")
    return oc if oc in ("correct", "wrong", "abstain") else "abstain"


def metrics(store: dict[str, dict], keep, mode: str) -> dict:
    c = w = a = m = 0
    for q in keep:
        o = outcome(store, q, mode)
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
    return f"{v*100:.1f}%" if v is not None and not (isinstance(v, float) and math.isnan(v)) else "-"


class _Tee:
    def __init__(self, o, f):
        self.o, self.f = o, f
    def write(self, s):
        self.o.write(s); self.f.write(s)
    def flush(self):
        try:
            self.o.flush(); self.f.flush()
        except Exception:
            pass


def main() -> None:
    out_path = TRACES / "pipeline_full489_report.txt"
    TRACES.mkdir(parents=True, exist_ok=True)
    _fh = open(out_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _fh)
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    bucket_of = {i["qid"]: i["bucket"] for i in inputs}
    a_qids = set(bucket_of)
    memb = {q for q in a_qids if bucket_of[q] == "membership"}
    pw = a_qids - memb
    b_path = TRACES / "dataset_b_qids_all489.json"
    b_qids = (set(json.loads(b_path.read_text(encoding="utf-8"))["b_qids"]) & a_qids) if b_path.exists() else set()

    stores = {name: trace_by_version(v) for name, v in VERSIONS.items()}
    print(f"# 管线全 489 分层   A(全部)=n{len(a_qids)}  B(防记忆)=n{len(b_qids)}  (mem={len(memb)} pw={len(pw)})")
    for name, st in stores.items():
        have = sum(1 for q in a_qids if q in st)
        flag = "" if have >= len(a_qids) * 0.95 else "  <<< 未跑满，快照勿定论"
        print(f"  {name}: tracestore 命中 {have}/{len(a_qids)}{flag}")
    if not b_qids:
        print("  >>> 警告：dataset_b_qids_all489.json 缺失，B 段跳过。")
    print()

    for name, st in stores.items():
        print(f"===== {name}  (trace_version={VERSIONS[name]}) =====")
        for mode in MODES:
            print(f"  -- {mode} --")
            print(f"    {'切片':<14}{'n':>4}{'✓':>5}{'✗':>5}{'弃':>4}{'缺':>4}{'覆盖':>8}{'acc全':>8}{'acc答':>8}")
            slices = [("A·全部", a_qids), ("A·membership", memb), ("A·pairwise", pw)]
            if b_qids:
                slices += [("B·防记忆", b_qids), ("B·membership", b_qids & memb), ("B·pairwise", b_qids & pw)]
            for sub, sset in slices:
                if not sset:
                    continue
                mo = metrics(st, sset, mode)
                print(f"    {sub:<14}{mo['n']:>4}{mo['c']:>5}{mo['w']:>5}{mo['ab']:>4}{mo['miss']:>4}"
                      f"{fmt(mo['cov']):>8}{fmt(mo['acc_all']):>8}{fmt(mo['acc_ans']):>8}")
        print()

    # v16 vs v20 主口径对照（auto 模式·防集净益）：仅当两臂都在时
    if b_qids and "v16(R2 off)" in stores and "v20(R2 on)" in stores:
        print("===== v16(R2off) -> v20(R2on) 配对翻转（auto 模式，两臂都在的题）=====")
        off, on = stores["v16(R2 off)"], stores["v20(R2 on)"]

        def pair(keep, mode):
            both = [q for q in keep if q in off and q in on]
            w2c = sum(1 for q in both if outcome(off, q, mode) == "wrong" and outcome(on, q, mode) == "correct")
            c2w = sum(1 for q in both if outcome(off, q, mode) == "correct" and outcome(on, q, mode) == "wrong")
            return len(both), w2c, c2w
        for mode in MODES:
            for sub, sset in (("A", a_qids), ("B(防记忆)", b_qids)):
                n, x, y = pair(sset, mode)
                print(f"  {mode:<18} {sub:<10} n={n:<4} w2c={x} c2w={y} 净={x-y:+d}")
    # ---- 基线参照（A/B 上 acc已答，同口径 answer 层）----
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))

    def base_slice(s, keep):
        rows = {}
        for r in load_jsonl(TRACES / ('answers_' + s + '.jsonl')):
            rows[r['qid']] = r
        c = w = ab = done = 0
        for q in keep:
            r = rows.get(q)
            if not r or r.get('error'):
                continue
            o = outcome_of(r, gt.get(q, {}), bucket_of[q]); c += o == 'correct'; w += o == 'wrong'
            ab += o not in ('correct', 'wrong'); done += 1
        ans = c + w
        return done, c, w, ab, (c / ans if ans else math.nan), (c / done if done else math.nan)

    print("===== 基线参照（同 A/B 口径，acc已答；样本量=该切片已答数）=====")
    print(f"    {'系统':<10}{'A·全部 已答/acc答':<22}{'B·防记忆 已答/acc答':<24}")
    for s in ('metagpt', 'autogen', 'pure_llm', 'mdagents', 'lit_rag'):
        da, ca, wa, aba, aaa, ata = base_slice(s, a_qids)
        db, cb, wb, abb, abb_a, _ = base_slice(s, b_qids)
        print(f"    {s:<10}n{da}/489  {fmt(aaa):<12}n{db}/265  {fmt(abb_a):<12}")
    print()

    print("\n[written] outputs/traces/pipeline_full489_report.txt")


if __name__ == "__main__":
    main()
