# -*- coding: utf-8 -*-
"""杠杆3 诊断：量化 v20graph 的"图沉默基因"问题——到底有没有提升空间。

图沉默基因 = 在癌种优先截断后的 path 行里 n_paths==0 的候选基因。pairwise 图 logistic 对它们
只能给"全 0 特征 → 基线概率"，排序退化成按名字典序、几乎必然沉底。若这些沉默基因里有【本该命中
GT】的，就是杠杆3（亚家族旁系 path 传播补特征）能救回的空间；若 GT 基因本身都有路径证据，则杠杆3
也帮不上，应诚实停手。

只读 kg_cypher 取 path 行，不调组学/LLM，秒级。
用法：
  . tools\\slc_mars_env.ps1
  .venv\\Scripts\\python.exe tools\\_topk_silent_diag.py
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
from _topk_head import fetch_path_rows, cap_per_gene, CAP, PACK_PATH, load_model, prob_of


def main():
    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    be = get_default_backend()
    if type(be).__name__ == "UnavailableBackend":
        raise SystemExit("后端不可用，先 . tools\\slc_mars_env.ps1")

    tot_silent = tot_silent_gt = tot_gt = 0
    per_q = []
    for pack in packs:
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        cancer = (pack.get("constraints") or {}).get("cancer", "")
        capped = cap_per_gene(fetch_path_rows(be, pool, cancer), CAP)
        feats = {g: gene_graph_feat(g, capped, cancer) for g in pool}
        silent = [g for g in pool if feats[g]["n_paths"] == 0]
        silent_gt = [g for g in silent if g in set(gt)]
        gt_set = set(gt)
        # GT 里"非沉默"的基因数（=杠杆3救不到、v20graph 本就能给分的）
        n_gt_silent = len(silent_gt)
        tot_silent += len(silent); tot_silent_gt += n_gt_silent; tot_gt += len(gt_set)
        per_q.append((pack["qid"][-22:], len(pool), len(silent), n_gt_silent, len(gt_set),
                      ",".join(sorted(silent_gt)) or "-"))
        print(f"{pack['qid'][-24:]:<24} pool={len(pool):>2d} silent={len(silent):>2d} "
              f"GT中沉默={n_gt_silent}/{len(gt_set):>2d}  沉默GT基因: {','.join(sorted(silent_gt)) or '-'}")

    print("=" * 72)
    print(f"合计：图沉默候选基因 {tot_silent} 个；其中【落入 GT】的 {tot_silent_gt} 个"
          f"（GT 基因总数 {tot_gt}）")
    print(f"杠杆3 的理论可救上限 = 沉默且属 GT 的基因 = {tot_silent_gt} 个"
          f"（占全部 GT 的 {tot_silent_gt/max(1,tot_gt):.0%}）")


if __name__ == "__main__":
    main()
