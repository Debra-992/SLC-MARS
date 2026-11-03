# -*- coding: utf-8 -*-
"""v12 管线自身在扩测 n=100 上的成绩（分桶/分臂 + B 集），并与 mdagents 已有覆盖做同题配对。"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
T = Path(r"outputs/traces")
mani = json.loads((T / "sample_manifest.json").read_text(encoding="utf-8"))
qids = mani["qids"]
bq = set(json.loads((T / "dataset_b_qids.json").read_text(encoding="utf-8"))["b_qids"])

# v12 管线行（同 qid 最新写）
v12 = {}
for l in (T / "tracestore.jsonl").read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    if str((r.get("provenance") or {}).get("trace_version")) == "12":
        v12[r["qid"]] = r

# mdagents 已答（非 error）
md = {}
for l in (T / "answers_mdagents.jsonl").read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    if not r.get("error"):
        md[r["qid"]] = r


def acc(rows_arm, keep):
    c = n = ab = 0
    for q in keep:
        if q not in v12:
            continue
        s = v12[q]["systems"][rows_arm]
        o = s.get("outcome")
        n += 1
        c += (o == "correct")
        ab += (o == "abstain")
    return c, n, ab, (round(100 * c / n, 1) if n else 0)


ALL = qids
Bset = [q for q in qids if q in bq]
NONB = [q for q in qids if q not in bq]
PW = [q for q in qids if v12.get(q, {}).get("systems", {}).get("pipeline_auto", {})
      and (q.split("_")[0] == "pairwise")]
print(f"v12 管线覆盖: {len([q for q in qids if q in v12])}/{len(qids)}  B集={len(Bset)}  mdagents已答={len(md)}")

for arm in ("pipeline_auto", "pipeline_heuristic"):
    print(f"\n=== {arm} ===")
    for lab, keep in (("全100", ALL), ("非B", NONB), ("B防记忆", Bset)):
        c, n, ab, p = acc(arm, keep)
        print(f"  {lab:8s} n={n:3d}  correct={c:3d} abstain={ab:2d}  acc={p}%")

# 同题配对 v12 vs mdagents（只用 mdagents 已答的题，公平口径）
common = [q for q in qids if q in md and q in v12]
print(f"\n=== 同题配对 v12 vs mdagents（当前 md 已覆盖 {len(common)} 题）===")
for arm in ("pipeline_auto", "pipeline_heuristic"):
    vc = sum(1 for q in common if v12[q]["systems"][arm].get("outcome") == "correct")
    # mdagents outcome 需按 GT 重判：merge_baselines 才有 _score；这里粗略用其 answer 与管线 GT 对照不便
    print(f"  {arm}: v12 correct {vc}/{len(common)} = {round(100*vc/len(common),1)}%")
print("  (mdagents 精确 acc 待其补齐 100 题后由 merge_baselines 统一 _score 口径产出)")
