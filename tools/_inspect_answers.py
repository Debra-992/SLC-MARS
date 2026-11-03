"""Quick inspect of answer JSONL files to understand winner/outcome structure."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "outputs" / "traces"

for name in ["answers_pure_llm.jsonl", "answers_mdagents.jsonl"]:
    path = TRACES / name
    if not path.exists():
        continue
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().splitlines()]
    print(f"=== {name}: {len(lines)} records ===")
    winners = set(str(r.get("winner", "N/A")) for r in lines)
    print(f"  winner values: {winners}")
    buckets = set(r.get("bucket", "N/A") for r in lines)
    print(f"  bucket values: {buckets}")
    bw = Counter((r.get("bucket","?"), str(r.get("winner","?"))) for r in lines)
    for (b, w), c in sorted(bw.items()):
        print(f"    bucket={b:<12s} winner={w:<10s} count={c}")
    abst = sum(1 for r in lines if r.get("abstained"))
    print(f"  abstained: {abst}")
    for r in lines[:3]:
        w = r.get("winner","?")
        print(f"  [{r['qid'][:50]}] bucket={r.get('bucket')} winner={w} abst={r.get('abstained')} ans={str(r.get('answer',''))[:40]}")
    print()