"""de_panel_ext quick dump"""
import json; from pathlib import Path
ext = json.loads((Path("<PROJECT_ROOT>/outputs/traces/de_panel_ext.json")).read_text("utf-8"))
for p in ext["panels"]:
    ct = p["cancer_type"]
    top = p.get("top_genes") or []
    sig = sum(1 for g in top if float(g.get("adj_p_value") or 1) < 0.05)
    print(f"{ct:<40} genes={len(top):>3} sig={sig:>3}  status={p.get('status')}  contrast={p.get('contrast_kind','?')}")
print(f"\nv20 cancer list:")
import re
store = [json.loads(l) for l in open("<PROJECT_ROOT>/outputs/traces/tracestore.jsonl","r",encoding="utf-8") if l.strip()]
tv20 = [r for r in store if str((r.get("provenance",{})).get("trace_version","")) == "20"]
cancers = sorted(set((r.get("constraints") or {}).get("cancer","") for r in tv20 if (r.get("constraints") or {}).get("cancer")))
for c in cancers:
    n = sum(1 for r in tv20 if (r.get("constraints") or {}).get("cancer","") == c)
    print(f"  {c:<40} n={n:>3}")