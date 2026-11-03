"""验证 topk_eval_pack.json"""
import json
pack = json.load(open("<PROJECT_ROOT>/outputs/traces/topk_eval_pack.json", encoding="utf-8"))
print(f"Packs: {len(pack['packs'])}")
for p in pack['packs']:
    err = p.get('error','')
    print(f"  {p['qid'][-10:]}  pool={len(p['eval_pool']):2d}  GT-in-pool={len(p.get('gt_in_pool',[]))}/5  PPR_NDCG5={p.get('ppr_baseline_ndcg5',0):.3f}  {'ERR:'+err if err else ''}")

ok = [p for p in pack['packs'] if not p.get('error')]
if ok:
    print(f"\nPPR NDCG@5 mean (8 successful): {sum(p.get('ppr_baseline_ndcg5',0) for p in ok)/len(ok):.3f}")

p0 = pack['packs'][0]
print(f"\n=== 首题详情 ===")
print(f"question: {p0['question'][:120]}...")
print(f"GT ranking: {p0['gt_ranking']}")
print(f"eval_pool ({len(p0['eval_pool'])} genes): {p0['eval_pool']}")
print(f"gt_in_pool: {p0['gt_in_pool']}")
print(f"PPR top-5: {[c[0] for c in p0['ppr_candidates'][:5]]}")

# 检查失败题
fail = [p for p in pack['packs'] if p.get('error')]
if fail:
    print(f"\n=== {len(fail)} 失败题 ===")
    for p in fail:
        print(f"  {p['qid']}: {p['error']}")
        print(f"    GT={p['gt_ranking']}")
        print(f"    eval_pool={p['eval_pool']}")