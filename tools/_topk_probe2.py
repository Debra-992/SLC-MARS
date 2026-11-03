"""topk GT 全貌"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

lines = open(ROOT / "evaluation" / "ground_truth_clean.jsonl", encoding="utf-8").readlines()
topk_gt = [json.loads(l) for l in lines if "topk" in l]
print(f"Total topk GT entries: {len(topk_gt)}")
for g in topk_gt:
    ranking = g.get("ground_truth", [])
    print(f"  {g['qid']}: {ranking}")

# 也看 questions 的 k 值
from tools.clean_questions import load_jsonl
qs = [r for r in load_jsonl(ROOT / "data" / "questions_clean.jsonl") if r.get("task_bucket") == "topk"]
print(f"\nTotal topk questions: {len(qs)}")
for q in qs:
    k = q.get("constraints", {}).get("k", "?")
    print(f"  {q['qid']} k={k}  task_type={q.get('task_type')}")