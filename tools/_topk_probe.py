"""一次性探针：topk 题面 + GT 格式"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.clean_questions import load_jsonl

qs = [r for r in load_jsonl(ROOT / "data" / "questions_clean.jsonl") if r.get("task_bucket") == "topk"]
gts = {r["qid"]: r for r in load_jsonl(ROOT / "evaluation" / "ground_truth_clean.jsonl")}
print(f"{'qid':<48} {'k'} {'ranking (GT)'}")
print("-" * 100)
for q in qs:
    gt = gts.get(q["qid"], {})
    gt_inner = gt.get("ground_truth", gt)
    ranking = gt_inner.get("ranking", "N/A")
    k = q.get("constraints", {}).get("k", "?")
    if isinstance(ranking, list):
        ranking = " → ".join(ranking)
    print(f"{q['qid']:<48} {k} {ranking}")

# 详看一道 topk 题 + GT
print("\n\n=== 第一道 topk 完整题面 ===")
q0 = qs[0]
gt0 = gts.get(q0["qid"], {})
print(json.dumps(q0, ensure_ascii=False, indent=2))
print("\nGT:")
print(json.dumps(gt0, ensure_ascii=False, indent=2, default=str))