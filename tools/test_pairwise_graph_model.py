# -*- coding: utf-8 -*-
"""离线测试：pairwise 图特征 logistic 接入 Vega 判定层（v10）。
覆盖：模型分支按 argmax 选 winner、无图证据诚实回退规则、env 未设=纯 v9 规则、坏工件回退。"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# metagpt.actions 的导入链会拉 chardet（其 mypyc 编译扩展在部分 Windows 上被 OS
# 应用控制策略拦截）。本测试只走启发式打分、不用 chardet，故在其失败时塞一个纯 Python 桩，
# 仅在真 chardet 不可用时兜底，正常工作环境不受影响。
try:  # noqa: SIM105
    import chardet  # noqa: F401
except Exception:  # noqa: BLE001
    import types

    _stub = types.ModuleType("chardet")
    _stub.detect = lambda b: {"encoding": "utf-8", "confidence": 1.0, "language": ""}  # type: ignore[attr-defined]
    _stub.__version__ = "0"  # type: ignore[attr-defined]
    sys.modules["chardet"] = _stub

from knowledge_agents.actions import decision_action as da  # noqa: E402
from knowledge_agents.graph_features import PAIR_COLS  # noqa: E402
from knowledge_agents.schema import (  # noqa: E402
    BackgroundKnowledge, ComputedKnowledge, KGEvidence, ToolStatus,
)

ad = da.AnswerDecision()
CK = ComputedKnowledge(question="t")
ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("PASS " if cond else "FAIL ") + name)
    ok += int(cond)
    fail += int(cond is False)


def kg_paths(rows):
    kg = BackgroundKnowledge(question="t", status=ToolStatus.OK)
    kg.evidence.append(KGEvidence(operation="gene_disease_paths", rows=rows))
    return kg


def row(gene, eff="PROMOTION"):
    return {"gene": gene, "disease": "breast cancer(BC)", "rel_AB": eff, "rel_BC": eff,
            "gene_disease_effect": eff, "pathway": "proliferation"}


def write_artifact(coef_over=None, only_col=None):
    """造一个可控工件：默认只有 n_paths 权重=+1（其余 0），winner=路径多的基因。"""
    coef = [0.0] * len(PAIR_COLS)
    if only_col:
        coef[PAIR_COLS.index(only_col)] = 1.0
    if coef_over:
        coef = coef_over
    art = {"task": "pairwise_graph_logistic", "features": list(PAIR_COLS),
           "mu": [0.0] * len(PAIR_COLS), "sd": [1.0] * len(PAIR_COLS),
           "coef": coef, "intercept": 0.0, "C": 0.01, "seed": 5, "train_frac": 0.6,
           "train_gene_rows": 10, "trained_qids": [], "holdout": {"n_questions": 0, "acc": 0.74},
           "no_signal_behavior": "fallback_rules"}
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(art, f, ensure_ascii=False)
    return path


# 每个用例前清缓存（模块级缓存按 path 存，env 变即换 key，但显式清更稳）
def use_env(path):
    da._PAIRWISE_MODEL_CACHE.clear()
    if path:
        os.environ["SLC_PAIRWISE_MODEL"] = path
    else:
        os.environ.pop("SLC_PAIRWISE_MODEL", None)


try:
    # (1) 模型分支：A 有 3 条路径、B 有 1 条 -> n_paths 权重最高者 A 胜
    p = write_artifact(only_col="n_paths")
    use_env(p)
    kg = kg_paths([row("GPA"), row("GPA"), row("GPA"), row("GPB")])
    a = ad._heuristic("q", kg, CK, "pairwise", ["GPB", "GPA"], "pairwise_positive", {"cancer": "breast"})
    check("model branch picks higher n_paths gene (A)", a.winner == "GPA" and a.abstained is False)
    check("model branch marks v10 rationale", "v10" in a.rationale)
    check("model branch confidence = winner prob in [0,1]", 0.0 <= a.confidence <= 1.0)

    # (2) 两候选皆无 gene_disease_paths（空证据）-> 回退规则（不伪造 winner 由模型给）
    p2 = write_artifact(only_col="n_paths")
    use_env(p2)
    kg_empty = BackgroundKnowledge(question="t", status=ToolStatus.OK)
    a2 = ad._heuristic("q", kg_empty, CK, "pairwise", ["GX", "GY"], "pairwise_positive", {"cancer": "breast"})
    check("no graph evidence -> fallback (rationale NOT v10-model)", "图特征logistic·v10" not in a2.rationale)

    # (3) env 未设 -> 纯 v9 规则路径（即使有图证据也不走模型）
    use_env(None)
    kg3 = kg_paths([row("GPA"), row("GPA"), row("GPA"), row("GPB")])
    a3 = ad._heuristic("q", kg3, CK, "pairwise", ["GPB", "GPA"], "pairwise_positive", {"cancer": "breast"})
    check("env unset -> pure v9 rules (no v10 marker)", "图特征logistic·v10" not in a3.rationale)

    # (4) 坏工件（不存在/JSON 坏）-> 回退规则、不抛异常
    fd, bad = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write("{not valid json")
    use_env(bad)
    try:
        a4 = ad._heuristic("q", kg_paths([row("GPA")]), CK, "pairwise",
                           ["GPA", "GPB"], "pairwise_positive", {"cancer": "breast"})
        check("broken artifact -> fallback, no crash", "图特征logistic·v10" not in a4.rationale)
    except Exception as e:  # noqa BLE001
        check("broken artifact -> fallback, no crash", False)
        print("   raised:", type(e).__name__, e)
    finally:
        os.remove(bad)

    # (5) 真工件（若存在）能加载并对真实分布打分（smoke，不硬编码期望）
    real = os.path.join("outputs", "traces", "pairwise_graph_model.json")
    if os.path.exists(real):
        use_env(real)
        kgr = kg_paths([row("SLC7A11"), row("SLC7A11"), row("SLC7A5")])
        ar = ad._heuristic("q", kgr, CK, "pairwise", ["SLC7A5", "SLC7A11"],
                           "pairwise_positive", {"cancer": "breast"})
        check("real artifact: decisive, in-candidates winner",
              ar.winner in ("SLC7A5", "SLC7A11") and ar.abstained is False)

    print(f"\n合计 PASS={ok} FAIL={fail}")
    sys.exit(1 if fail else 0)
finally:
    use_env(None)
