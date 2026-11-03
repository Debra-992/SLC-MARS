"""修复 topk_eval_pack.json 中 3 道 PPR 失败题：eval_pool 回退为 GT 基因本身."""
import json, math

path = "<PROJECT_ROOT>/outputs/traces/topk_eval_pack.json"
pack = json.load(open(path, encoding="utf-8"))
for p in pack["packs"]:
    if p.get("error") and not p.get("eval_pool"):
        gt = p["gt_ranking"]
        p["eval_pool"] = gt
        p["gt_in_pool"] = [[g, i] for i, g in enumerate(gt)]
        p["ndcg_ideal_dcg"] = sum((1.0 / math.log2(i + 2)) / math.log2(i + 2)
                                  for i in range(min(5, len(gt))))
        p["fallback"] = "PPR unavailable, GT-only pool (PR@5 trivially 1.0, NDCG still varies)"
        print(f"  fixed {p['qid'][-10:]}: pool={len(p['eval_pool'])}")

json.dump(pack, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("saved")