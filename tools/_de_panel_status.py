"""de_panel 状态探针：DE cache + 扩展面板 + v20 癌种覆盖"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 1) de_workspace_cache
de_dir = ROOT / "outputs/slc-tools-cache/de_workspace_cache"
cached = {}
for f in de_dir.glob("*.json"):
    d = json.loads(f.read_text("utf-8"))
    ct = d.get("cancer_type", "?")
    top = d.get("top_genes") or []
    sig = sum(1 for g in top if float(g.get("adj_p_value") or 1) < 0.05)
    cached[ct] = dict(file=f.name, n_genes=len(top), sig=sig)
print("=== de_workspace_cache ===")
for ct, info in sorted(cached.items()):
    print(f"  {ct:<35}  genes={info['n_genes']:>3}  sig(padj<0.05)={info['sig']:>3}")

# 2) de_panel_ext.json
ext_path = ROOT / "outputs/traces/de_panel_ext.json"
panels = {}
if ext_path.exists():
    ext = json.loads(ext_path.read_text("utf-8"))
    panels = ext.get("panels", {})
    print(f"\n=== de_panel_ext.json (built={ext.get('built_at','?')}) ===")
    print(f"  source: {ext.get('source','?')}")
    print(f"  n_per_group: {ext.get('n_per_group','?')}")
    print(f"  cancers: {len(panels)}")
    for ct, info in sorted(panels.items()):
        top = info.get("top_genes") or []
        sig = sum(1 for g in top if float(g.get("adj_p_value") or 1) < 0.05)
        print(f"  {ct:<35}  genes={len(top):>3}  sig={sig:>3}")
else:
    print("de_panel_ext.json NOT FOUND")

# 3) v20 questions: cancer coverage
store = [json.loads(l) for l in (ROOT / "outputs/traces/tracestore.jsonl").read_text("utf-8").splitlines() if l.strip()]
tv20 = [r for r in store if str((r.get("provenance", {})).get("trace_version", "")) == "20"]
v20_cancers = set()
for r in tv20:
    c = (r.get("constraints") or {}).get("cancer", "")
    if c:
        v20_cancers.add(c)
all_cancers = sorted(v20_cancers)
print(f"\n=== v20 question cancer coverage ({len(tv20)} rows) ===")
print(f"  unique cancers: {len(all_cancers)}")
de_ok = [c for c in all_cancers if c in cached]
de_ext = [c for c in all_cancers if c not in cached and c in panels]
de_miss = [c for c in all_cancers if c not in cached and c not in panels]
print(f"  covered by de_cache: {len(de_ok)}")
print(f"  covered by ext only: {len(de_ext)}")
print(f"  no DE panel       : {len(de_miss)}")
if de_miss:
    print("  MISSING:")
    for c in de_miss:
        print(f"    - {c}")

# 4) tracestore DE status per question
de_ok_q = de_miss_q = de_none_q = 0
for r in tv20:
    ck = (r.get("supply") or {}).get("ck", {})
    de = ck.get("differential_expression")
    if not de:
        de_none_q += 1
    elif de.get("status") == "ok":
        de_ok_q += 1
    else:
        de_miss_q += 1
print(f"\n  DE status in tracestore: ok={de_ok_q} miss={de_miss_q} none={de_none_q}")