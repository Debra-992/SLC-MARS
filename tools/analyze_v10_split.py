# -*- coding: utf-8 -*-
"""离线：把 v10 对比(n=40) 的 pairwise heuristic(=图模型)臂按【训练内/留出】拆开，诚实报告。"""
from __future__ import annotations
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(BASE, *a)


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    mani = json.load(open(P("outputs", "traces", "sample_manifest.json"), encoding="utf-8"))
    assert mani.get("trace_version") == "10", mani.get("trace_version")
    qids = set(mani["qids"])
    art = json.load(open(P("outputs", "traces", "pairwise_graph_model.json"), encoding="utf-8"))
    tr = set(art["trained_qids"])
    rows = load_jsonl(P("outputs", "traces", "tracestore.jsonl"))
    # tracestore 是追加式历史：同一 qid 取【最后一条】，且必须是本次 v10 运行写的
    latest: dict[str, dict] = {}
    for r in rows:
        latest[r["qid"]] = r
    rows = [r for r in latest.values()
            if (r.get("provenance") or {}).get("trace_version") == "10"]

    def tally(filt, arm):
        c = w = a = 0
        for r in rows:
            q = r.get("qid")
            if q not in qids or r.get("bucket") != "pairwise" or not filt(q):
                continue
            o = (r.get("systems", {}).get(arm) or {}).get("outcome")
            if o == "correct":
                c += 1
            elif o == "wrong":
                w += 1
            else:
                a += 1
        n = c + w + a
        acc_ans = c / (c + w) if (c + w) else float("nan")
        acc_all = c / n if n else float("nan")
        return n, c, w, a, acc_ans, acc_all

    print(f"manifest qids={len(qids)}  trained_qids(池)={len(tr)}")
    for arm in ("pipeline_heuristic", "pipeline_auto", "baseline"):
        print(f"\n== pairwise 臂 [{arm}] ==")
        for name, filt in (("训练内(in-sample)", lambda q: q in tr),
                           ("留出(held-out)", lambda q: q not in tr),
                           ("全体", lambda q: True)):
            n, c, w, a, aa, ab = tally(filt, arm)
            if n == 0:
                print(f"  {name:<18} n=0")
                continue
            print(f"  {name:<18} n={n:<3} ✓{c} ✗{w} abstain{a}  acc(已答)={aa:.3f}  acc(全体)={ab:.3f}")


if __name__ == "__main__":
    main()
