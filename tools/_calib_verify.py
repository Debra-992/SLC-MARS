# -*- coding: utf-8 -*-
"""校验 conf_calib：① 烘焙常量在 v16@n=100 上复现拟合 AUC-RC(in-sample≈0.2173)；
② SLC_CONF_CALIB 关闭时 apply_calibration 是彻底 no-op（confidence 逐字节不变）。
用法: .venv\\Scripts\\python.exe tools\\_calib_verify.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from _v16_diag import auc_rc, ece, load_jsonl  # noqa: E402
from knowledge_agents.baselines.common import TRACES  # noqa: E402
from knowledge_agents import conf_calib  # noqa: E402


def to_ns(a: dict) -> SimpleNamespace:
    return SimpleNamespace(gene_scores=a.get("gene_scores") or {},
                           decision_mode=a.get("decision_mode") or "",
                           rationale=a.get("rationale") or "",
                           used_llm=bool(a.get("used_llm")),
                           abstained=bool(a.get("abstained")),
                           confidence=float(a.get("confidence") or 0.0))


def main():
    pairs = []          # (calib_conf, outcome)
    raws = []           # (orig_conf, outcome)
    n_abstain = 0
    for r in load_jsonl(TRACES / "tracestore.jsonl"):
        if str((r.get("provenance") or {}).get("trace_version") or "") != "16":
            continue
        s = (r.get("systems", {}) or {}).get("pipeline_auto") or {}
        a = s.get("answer") or {}
        o = s.get("outcome") or "abstain"
        if o not in ("correct", "wrong"):
            n_abstain += 1
            continue
        ns = to_ns(a)
        pairs.append((conf_calib.calibrated_confidence(ns), o))
        raws.append((ns.confidence, o))

    a_cal, _ = auc_rc(pairs)
    a_raw, _ = auc_rc(raws)
    print(f"# v16@n=100 复现（in-sample，权重即在本题集拟合，乐观）")
    print(f"  calib  AUC-RC={a_cal:.4f} ECE={ece(pairs):.3f}")
    print(f"  raw    AUC-RC={a_raw:.4f} ECE={ece(raws):.3f}  (对照：v16 现用)")
    print(f"  abstain 行(未计)={n_abstain}")
    print(f"  期望 in-sample≈0.2173（同 _calib_fit full-fit）；诚实样本外期望 0.249±0.007")

    # 门控 no-op 校验
    ns = to_ns({"gene_scores": {"A": 5.0}, "decision_mode": "heuristic", "used_llm": False,
                "abstained": False, "confidence": 0.999})
    os.environ.pop("SLC_CONF_CALIB", None)
    before = ns.confidence
    conf_calib.apply_calibration(ns)
    print(f"\n# gate OFF：confidence {before} -> {ns.confidence}  unchanged={before == ns.confidence}")
    os.environ["SLC_CONF_CALIB"] = "1"
    conf_calib.apply_calibration(ns)
    print(f"# gate ON ：confidence -> {ns.confidence}  (heur+margin5.0 应显著下调)")


if __name__ == "__main__":
    main()
