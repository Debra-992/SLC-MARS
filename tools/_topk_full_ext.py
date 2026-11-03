# -*- coding: utf-8 -*-
"""B 路线 serving 侧旁路臂：v20full_ext = v20full + 外部 DESeq2 扩展面板（只补缺、不改判）。

【为什么不需要动冠军路径】
  冠军（v20/v16 三通道二分类判定 + 总表 + 消融）读的 DE 面板来自
  knowledge_agents.tools.computation_tools.differential_expression → de_workspace_cache，
  本臂【一行都不改那条链路】：不写 de_workspace_cache、不写 GDCdata、不 import
  decision_action/calib_dump/prewarm。扩展面板只在【本进程内存里】挂到 ck 对象上，
  进程退出即消失，磁盘上的冠军工件逐字节不变。

【注入规则（严格只补缺）】
  仅当该题癌种的原生 DE 通道"没有可用面板"（ck.differential_expression 为 None，
  或 status 非 ok / top_genes 空）时，才用 de_panel_ext.json 的同类癌种面板替换；
  原生已 ok 的癌种（BRCA/COAD/STOMAD/GBM/KIRC…）一律保持原样 → 保证：
    · 未覆盖癌种的题，预测必须与 topk_eval_v20full.json 逐题相同（脚本末尾自检）；
    · 冠军二分类与总表看到的 DE 完全不变。
  替换后按 computation_action 同一口径重算 paralog_redundancy（面板派生通道）。

【量纲对齐】de_panel_ext.json 由 tools/_de_ext_build.py 产出，列名/排序/截断
  （gene_symbol/gene/log2fc/p_value/adj_p_value，padj 升序 head(400)，NaN→null）
  与原生 tool_operations._de 逐字一致，故 de_log2fc / de_sig_count / de_panel_n /
  paralog_redundancy 四个键都是模型训练时见过的同一 DESeq2 统计量，
  不重演 lever1 的"非 DESeq2 统计量 → z 分数 out-of-distribution"。

用法：
  . tools\\slc_mars_env.ps1
  .venv\\Scripts\\python.exe tools\\_topk_full_ext.py
"""
from __future__ import annotations
import asyncio, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "knowledge_agents" / "baselines"))
sys.path.insert(0, str(ROOT / "tools"))

from knowledge_agents.feature_vector import FULL_COLS, vec, features_for_gene
from knowledge_agents.schema import DifferentialExpression
from knowledge_agents.tools import paralog as paralog_mod
from knowledge_agents.tools.slc_backend import get_default_backend
from knowledge_agents.roles.compute_role import ComputeRole
# 与 v20full 共用同一取数/截断/打分/指标实现，两臂唯一差别就是 DE 面板来源
from _topk_full import afetch_path_rows, load_model, prob_of, fetch_ck, MODEL_PATH
from _topk_head import cap_per_gene, metrics_of, ppr_ranking, CAP, PACK_PATH, TRACES

EXT_PATH = TRACES / "de_panel_ext.json"
BASE_ARM = TRACES / "topk_eval_v20full.json"


def load_ext() -> dict[str, dict]:
    """扩展面板按癌种标签建索引；文件不存在就直接空表（本臂退化为 v20full，诚实报告）。"""
    if not EXT_PATH.is_file():
        return {}
    data = json.loads(EXT_PATH.read_text(encoding="utf-8"))
    return {p["cancer_type"]: p for p in data.get("panels") or [] if p.get("status") == "ok"}


def panel_usable(de) -> bool:
    """原生面板是否已可用：status=ok 且有 top_genes（冠军 serving 实际读的就是这两项）。
    ToolStatus 是 str-Enum，str(enum) 可能给出 'ToolStatus.OK'，故统一取 .value 比对。"""
    if de is None:
        return False
    st = getattr(de, "status", None)
    st = getattr(st, "value", st)
    return str(st) == "ok" and bool(de.top_genes)


def inject(ck, cancer: str, genes: list[str], ext: dict[str, dict]) -> str:
    """只补缺的内存注入。返回诊断标签：kept_native / injected / no_ext_panel。"""
    p = ext.get((cancer or "").strip())
    if ck is None:
        return "no_ck"
    if panel_usable(ck.differential_expression):
        return "kept_native"
    if p is None:
        return "no_ext_panel"
    fields = set(DifferentialExpression.model_fields)
    ck.differential_expression = DifferentialExpression(**{k: v for k, v in p.items() if k in fields})
    # 面板派生通道随之重算（与 computation_action L134-141 同一函数、同一 k_sat）
    panel = paralog_mod.panel_from_de(ck.differential_expression)
    ck.paralog_redundancy = {}
    ck.paralog_partners = {}
    if panel and genes:
        red, partners = paralog_mod.redundancy_map(
            genes, panel, k_sat=float(os.environ.get("SLC_PARALOG_KSAT", "4")))
        ck.paralog_redundancy = dict(red)
        ck.paralog_partners = dict(partners)
    return "injected"


async def main():
    packs = json.loads(PACK_PATH.read_text(encoding="utf-8"))["packs"]
    m = load_model()
    ext = load_ext()
    print(f"v20full_ext | CAP={CAP} | C={m.get('C')} | 列={len(FULL_COLS)} | 题数={len(packs)} "
          f"| 扩展面板癌种={sorted(ext)} | 模型={MODEL_PATH.name}")
    if not ext:
        print("!! de_panel_ext.json 缺失/无 ok 面板 —— 本臂将等于 v20full（仅作回归自检）")
    print("=" * 92)

    be = get_default_backend()
    if type(be).__name__ == "UnavailableBackend":
        raise SystemExit("MCP 后端不可用（先 . tools\\slc_mars_env.ps1 且确保 Neo4j 可达）")

    role = ComputeRole()
    rows = []
    for i, pack in enumerate(packs):
        qid = pack["qid"]
        pool = [g.upper() for g in pack["eval_pool"]]
        gt = [g.upper() for g in pack["gt_ranking"]]
        cancer = (pack.get("constraints") or {}).get("cancer", "")
        question = pack.get("question", "")

        ppr_pred = ppr_ranking(pack)
        ppr_block = None if ppr_pred is None else metrics_of(ppr_pred, gt)
        ppr_named = len({g.upper() for g, _s in (pack.get("ppr_candidates") or [])
                         if g.upper() in set(pool)})

        raw = await afetch_path_rows(be, pool, cancer)
        capped = cap_per_gene(raw, CAP)
        ck = await fetch_ck(role, qid, pool, cancer, question)
        tag = inject(ck, cancer, pool, ext) if ck is not None else "no_ck"

        if not capped and ck is None:
            err = "no_evidence(不可测)"
            rows.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                             pred=None, ppr=ppr_block, ndcg5=0.0, kendall_tau=0.0,
                             precision5=0.0, hit5=0.0, mrr=0.0, error=err, de_inject=tag,
                             tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0}))
            print(f"[{i+1:2d}/11] {qid[-24:]:<24} ERR {err}")
            continue

        feats = {g: features_for_gene(g, capped, cancer, ck) for g in pool}
        probs = {g: prob_of(vec(feats[g], FULL_COLS), m) for g in pool}
        ranked = sorted(pool, key=lambda g: (-probs[g], g))
        pred = ranked[:5]
        gm = metrics_of(pred, gt)

        rows.append(dict(qid=qid, pool_size=len(pool), gt=gt, ppr_named=ppr_named,
                         pred=pred, ppr=ppr_block, seeds=[], n_seed_ok=1, per_seed=[],
                         rationale="", tokens={"prompt": 0, "completion": 0, "total": 0, "calls": 0},
                         error="", de_inject=tag, **gm))
        de = getattr(ck, "differential_expression", None) if ck else None
        deN = len(de.top_genes or []) if de is not None else 0
        cov = sum(1 for g in pool if feats[g]["de_log2fc"] is not None)
        print(f"[{i+1:2d}/11] {qid[-24:]:<24} pool={len(pool):>2d} named={ppr_named:>2d} "
              f"capped={len(capped):>3d} {cancer:<18s} de={tag:<14s} panel={deN:>3d} "
              f"池内带log2fc={cov:>2d} | NDCG={gm['ndcg5']:.3f} hit@5={gm['hit5']:.0f} "
              f"τ={gm['kendall_tau']:+.3f}")

    ok = [r for r in rows if not r["error"]]
    json.dump(dict(system="v20full_ext", hint="none", seeds=[], thinking="OFF", dry_run=False,
                   n=len(rows), n_ok=len(ok), n_err=len(rows) - len(ok),
                   cap=CAP, model="fullfeature_membership_pointwise",
                   de_ext_source=str(EXT_PATH.name), results=rows),
              open(TRACES / "topk_eval_v20full_ext.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---- 旁路惰性自检：没注入面板的题必须与 v20full 逐题同预测、同指标 ----
    print(f"\n写入 topk_eval_v20full_ext.json  (ok={len(ok)}/{len(rows)})")
    if BASE_ARM.is_file():
        base = {r["qid"]: r for r in json.loads(BASE_ARM.read_text(encoding="utf-8"))["results"]}
        drift, same = [], 0
        for r in rows:
            b = base.get(r["qid"])
            if not b:
                continue
            if r["de_inject"] != "injected":
                if b.get("pred") != r["pred"]:
                    drift.append((r["qid"], r["de_inject"], b.get("pred"), r["pred"]))
                else:
                    same += 1
        changed = [r["qid"] for r in rows if r["de_inject"] == "injected"]
        print(f"自检 | 未注入题={len(rows)-len(changed)} 其中预测与 v20full 完全一致={same} "
              f"不一致={len(drift)}（必须为 0，否则旁路不再惰性）")
        for q, t, bp, rp in drift:
            print(f"   !! {q} [{t}] base={bp} ext={rp}")
        print(f"注入题={len(changed)}：{[q.replace('topk_','')[-20:] for q in changed]}")
        for key in ("ndcg5", "kendall_tau", "precision5", "hit5", "mrr"):
            v0 = sum(b[key] for b in base.values() if b and not b.get("error")) / max(1, sum(1 for b in base.values() if b and not b.get("error")))
            v1 = sum(r[key] for r in ok) / len(ok)
            print(f"   {key:<12s} v20full={v0:.4f} -> v20full_ext={v1:.4f}  Δ={v1-v0:+.4f}")
    else:
        print("（topk_eval_v20full.json 不在，跳过并表自检）")


if __name__ == "__main__":
    asyncio.run(main())
