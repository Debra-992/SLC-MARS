"""topk 题癌种清单 + DE 面板覆盖"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# topk questions
pack = json.loads((ROOT / "outputs/traces/topk_eval_pack.json").read_text("utf-8"))

# DE cache index
de_dir = ROOT / "outputs/slc-tools-cache/de_workspace_cache"
cached = {}
for f in de_dir.glob("*.json"):
    d = json.loads(f.read_text("utf-8"))
    ct = d.get("cancer_type", "?")
    top = d.get("top_genes") or []
    cached[ct] = len(top) > 0

# ext panel cancers
ext_path = ROOT / "outputs/traces/de_panel_ext.json"
ext_cancers = set()
if ext_path.exists():
    ext = json.loads(ext_path.read_text("utf-8"))
    for p in ext.get("panels", []):
        ext_cancers.add(p["cancer_type"])

# Print table
print("topk questions + DE coverage:")
print(f"{'qid':<44} {'cancer':<22} {'pool':>4} {'DEcache':>7} {'ext':>4}")
print("-" * 85)
for p in pack:
    qid = p["qid"]
    gt = p.get("gt_ranking", [])
    q = p.get("question", "")
    m = re.search(r"cancer[:\s]*([A-Za-z\-\s]+?)(?:,|\.|$|\n)", q, re.I)
    cancer = m.group(1).strip() if m else "?"
    pool_size = len(p.get("eval_pool") or [])
    has_de = "OK" if cached.get(cancer) else "-"
    has_ext = "ext" if cancer in ext_cancers else "-"
    fallback = "FB" if p.get("fallback") or p.get("error") else ""
    print(f"  {qid:<42} {cancer:<22} {pool_size:>4} {has_de:>7} {has_ext:>4}  {fallback}")

# Summary
need_ext = [p for p in pack if 
    not cached.get((m := re.search(r"cancer[:\s]*([A-Za-z\-\s]+?)(?:,|\.|$|\n)", p["question"], re.I)) and m.group(1).strip() or "?", False)
    and (m.group(1).strip() if m else "?") not in ext_cancers]
print(f"\nneed DE ext: {len(need_ext)} questions")
for p in need_ext:
    qid = p["qid"]
    m = re.search(r"cancer[:\s]*([A-Za-z\-\s]+?)(?:,|\.|$|\n)", p["question"], re.I)
    cancer = m.group(1).strip() if m else "?"
    print(f"  {cancer}")