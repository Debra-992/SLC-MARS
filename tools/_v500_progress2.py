"""快速探针: v500 进度 + 错误检查"""
import json
from pathlib import Path

TRACES = Path("<PROJECT_ROOT>/outputs/traces")

# 1) tracestore
store_path = TRACES / "tracestore.jsonl"
lat = {}
for l in store_path.read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    try:
        r = json.loads(l)
        lat[r["qid"]] = r
    except Exception:
        pass

tv20_qids = {q for q, r in lat.items() if str((r.get("provenance") or {}).get("trace_version", "")) == "20"}
print(f"tv20 qids in tracestore: {len(tv20_qids)}")

# 2) manifest
manifest_path = TRACES / "sample_manifest.json"
if manifest_path.exists():
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool = set(m.get("qids", []))
    done = tv20_qids & pool
    print(f"manifest pool: {len(pool)}  |  done: {len(done)}  |  remaining: {len(pool - done)}  ({100*len(done)/max(1,len(pool)):.0f}%)")
    # estimate ETA: each question ~3.3 min at full speed
    remaining = len(pool - done)
    if remaining > 0:
        mins = remaining * 3.3
        h = int(mins // 60)
        m = int(mins % 60)
        print(f"ETA: ~{h}h {m}m")

# 3) check for errors in tracestore tv20
errors = [r for r in lat.values()
          if str((r.get("provenance") or {}).get("trace_version", "")) == "20"
          and r.get("error")]
if errors:
    print(f"\nERRORS in tv20: {len(errors)}")
    for e in errors[:3]:
        print(f"  {e['qid']}: {e['error'][:80]}")

# 4) check last 3 timestamps
import datetime
def ts(r):
    return r.get("provenance", {}).get("ts", "")
tv20_rows = [r for r in lat.values() if r["qid"] in tv20_qids]
tv20_rows.sort(key=ts)
for r in tv20_rows[-3:]:
    elapsed = ""
    try:
        dt = datetime.datetime.strptime(ts(r), "%Y-%m-%dT%H:%M:%SZ")
        elapsed = f" ({(datetime.datetime.utcnow() - dt).total_seconds()/3600:.1f}h ago)"
    except Exception:
        pass
    print(f"  {r['qid'][-10:]} ts={ts(r)}{elapsed}")