"""Connectivity probe for the Neo4j SLC knowledge graph.

Not a unit test. Dot-source the frozen environment first so the endpoint and the
read-only credentials come from configuration rather than being hardcoded here:

    . tools\\slc_mars_env.ps1
    .\.venv\\Scripts\\python.exe tools\\probe_neo4j.py
"""
import os
import sys

try:
    from neo4j import GraphDatabase
except ModuleNotFoundError:
    sys.exit("neo4j driver not installed in this interpreter (needs .venv).")

URI = os.environ.get("NEO4J_URI", "")
USER = os.environ.get("NEO4J_USER", "")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
if not URI or not USER or not PASSWORD:
    sys.exit("NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD unset. "
             "Dot-source tools\\slc_mars_env.ps1 (it loads secrets.local.ps1) first.")

d = GraphDatabase.driver(URI, auth=(USER, PASSWORD), connection_timeout=5)
d.verify_connectivity()
print("connected:", URI)
with d.session() as s:
    labels = [r["l"] for r in s.run("MATCH (n) UNWIND labels(n) AS l RETURN DISTINCT l LIMIT 30")]
    print("labels:", labels)
    rels = [r["t"] for r in s.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t LIMIT 20")]
    print("rel types:", rels)
    for lab in labels[:12]:
        c = s.run(f"MATCH (n:`{lab}`) RETURN count(n) AS c").single()["c"]
        print(f"  {lab}: {c}")
    try:
        n = s.run("MATCH (g:SLCGene) RETURN count(g) AS c").single()["c"]
        print(f"  SLCGene nodes: {n}")
    except Exception as e:
        print("  SLCGene check failed:", e)
d.close()
