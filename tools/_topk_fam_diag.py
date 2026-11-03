# -*- coding: utf-8 -*-
"""杠杆3 机制诊断：家族先验把沉默基因救到什么位置？离进前5还差多少？

背景：_topk_fam 已把 21/29 图沉默基因拿到家族先验分，但整臂 NDCG@5 只从 0.346→0.350。
本脚本只读、不写产物，逐题报告：
  - 沉默 GT 基因在 v20graph 序里的名次（应该几乎全在池尾）
  - 家族先验值 vs 该题 v20graph 的第5名门槛分（entry bar）
  - 家族先验能挤进前5的基因数（先验 > entry bar 且该基因属 GT）
一次看清是"先验量级太小"还是"救到了但门槛更高"。

用法：
  . tools\\slc_mars_env.ps1
  .venv\\Scripts\\python.exe tools\\_topk_fam_diag.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))
sys.path.insert(0, str(ROOT / "tools"))

from knowledge_agents.graph_features import gene_graph_feat
from knowledge_agents.tools.slc_backend import get_default_backend
from _topk_head import (fetch_path_rows, cap_per_gene, load_model, prob_of, CAP, PACK_PATH)
from _topk_fam import subfamily, family_prior_map

BAR = "\u2550" * 100


def main():
    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    m = load_model()
    be = get_default_backend()
    if type(be).__name__ == "UnavailableBackend":
        raise SystemExit("MCP 后端不可用（先 . tools\\slc_mars_env.ps1）")

    tot_silent_gt = tot_above = tot_rescued_none = 0
    n_lifted = 0
    for pack in packs:
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        gt_set = set(gt)
        cancer = (pack.get("constraints") or {}).get("cancer", "")

        capped = cap_per_gene(fetch_path_rows(be, pool, cancer), CAP)
        if not capped:
            print(f"{qid[-24:]:<24} 无 path 行，跳过")
            continue
        feats = {g: gene_graph_feat(g, capped, cancer) for g in pool}
        direct = {g: prob_of(feats[g], m) for g in pool}
        silent = [g for g in pool if feats[g]["n_paths"] == 0]
        silent_gt = [g for g in silent if g in gt_set]
        fams = {subfamily(g) for g in silent if subfamily(g)}
        prior = family_prior_map(be, fams, cancer, m) if fams else {}

        order = sorted(pool, key=lambda g: (-direct[g], g))
        rank_of = {g: i + 1 for i, g in enumerate(order)}
        bar5 = order[4]                      # 第5名基因
        entry = direct[bar5]                 # 进前5的门槛分

        print(f"{BAR}\n{qid[-26:]:<26} cancer={cancer[:18]:<18} pool={len(pool)} "
              f"silentGT={len(silent_gt)} entry#5={entry:.4f}({bar5})")
        for g in silent_gt:
            fam = subfamily(g)
            p = prior.get(fam)
            tot_silent_gt += 1
            if p is None:
                print(f"    {g:<12} fam={str(fam):<8} 先验=无(全族皆沉默/无数字亚家族) "
                      f"v20graph名次={rank_of[g]}/{len(pool)}")
                tot_rescued_none += 1
                continue
            above = p > entry
            if above:
                tot_above += 1
            print(f"    {g:<12} fam={fam:<8} 先验={p:.4f} {'>' if above else '<'} "
                  f"门槛{entry:.4f}  {'★可进前5' if above else '仍进不了前5'}"
                  f"  v20graph名次={rank_of[g]}/{len(pool)}")
            # 家族先验相对该题非沉默基因中位数的位置
            med = sorted(direct.values())[len(direct) // 2]
            n_lifted += 1

        vals = sorted(prior.values())
        if vals:
            print(f"    家族先验分布：min={vals[0]:.4f} 中位={vals[len(vals)//2]:.4f} "
                  f"max={vals[-1]:.4f} | 非沉默基因分中位={sorted(direct.values())[len(direct)//2]:.4f}")

    print(f"\n{BAR}\n汇总：沉默GT基因 {tot_silent_gt} 个；"
          f"拿到家族先验 {n_lifted} 个；先验>该题第5名门槛（能挤进前5） {tot_above} 个；"
          f"无可用旁系 {tot_rescued_none} 个")


if __name__ == "__main__":
    main()
