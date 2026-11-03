# -*- coding: utf-8 -*-
"""验证经 kg_cypher 内联查询能否为 topk 候选池取到 gene_disease_paths（与 project_knowledge 同式）。只读。"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knowledge_agents.tools.slc_backend import get_default_backend
from knowledge_agents.graph_features import gene_graph_feat, PAIR_COLS

pack = json.loads((ROOT / "outputs/traces/topk_eval_pack.json").read_text(encoding="utf-8"))
p0 = pack["packs"][0]
pool = [g.upper() for g in p0["eval_pool"]]
cancer = p0["constraints"].get("cancer", "")
glist = ",".join("'" + g.replace("'", "") + "'" for g in pool)
cy = (
    "MATCH (g:SLCGene)-[:IS_SOURCE]->(e:RelaEvent)<-[:IS_TARGET]-(d:Disease) "
    f"WHERE g.name IN [{glist}] "
    "OPTIONAL MATCH (p:Pathway)-[:IS_MEDIATOR]->(e) "
    "RETURN g.name AS gene, p.name AS pathway, e.rel_AB AS rel_AB, e.rel_BC AS rel_BC, "
    "CASE WHEN e.rel_AB = e.rel_BC THEN 'PROMOTION' ELSE 'SUPPRESSION' END AS gene_disease_effect, "
    "d.name AS disease, d.diseaseSort AS disease_sort "
    f"ORDER BY CASE WHEN toLower(d.name) CONTAINS toLower('{cancer.replace(chr(39), '')}') THEN 0 ELSE 1 END, d.name, g.name"
)
be = get_default_backend()
res = be.call("kg_cypher", {"query": cy, "limit": 1000})
print("status", res.get("status"), "count", res.get("count"), "err", res.get("error"))
rows = res.get("rows") or []
bygene = collections.Counter(str(r.get("gene", "")).upper() for r in rows)
print("distinct genes with >=1 row:", len(bygene), "/", len(pool))
print("max rows single gene:", max(bygene.values()) if bygene else 0)
for g in pool[:6]:
    fv = gene_graph_feat(g, rows, cancer)
    print(" ", g, {k: fv[k] for k in ("n_paths", "n_promote", "n_suppress", "n_cancer_hit")})
