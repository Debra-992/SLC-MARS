# -*- coding: utf-8 -*-
"""Neo4j 图谱 schema 深度自省：labels / relTypes / propertyKeys + 每种关系的端点标签、
属性键、计数与样例行（重点看 relevant / rel AB / rel BC / 时间字段）。只读。"""
from __future__ import annotations
import json
import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    connection_timeout=5, max_transaction_retry_time=0,
)
driver.verify_connectivity()


def rows(session, q, params=None):
    return [r.data() for r in session.run(q, params or {})]


with driver.session() as s:
    labels = [r["label"] for r in rows(s, "CALL db.labels() YIELD label RETURN label")]
    rels = [r["relationshipType"] for r in rows(s, "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")]
    props = [r["propertyKey"] for r in rows(s, "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey")]
    print("== LABELS ==", labels)
    print("\n== RELATIONSHIP TYPES ==", rels)
    print("\n== PROPERTY KEYS ==", props)

    # 每种 label 的节点数 + 属性键样例
    print("\n== NODE LABELS: count + sample property keys ==")
    for lab in labels:
        try:
            cnt = rows(s, f"MATCH (n:`{lab}`) RETURN count(n) AS c")[0]["c"]
            keys = rows(s, f"MATCH (n:`{lab}`) RETURN keys(n) AS k LIMIT 1")[0]["k"] if cnt else []
            one = rows(s, f"MATCH (n:`{lab}`) RETURN n AS n LIMIT 1")
            sample = one[0]["n"] if one else None
            print(f"  [{lab}] n={cnt} keys={keys}")
            if sample is not None:
                print(f"      sample: {json.dumps({k: str(v)[:40] for k, v in dict(sample).items()}, ensure_ascii=False)}")
        except Exception as e:
            print(f"  [{lab}] ERR {e}")

    # 每种 relationship：端点标签组合 + 属性键 + 计数 + 样例行
    print("\n== RELATIONSHIPS: endpoints / property keys / count / sample ==")
    for rel in rels:
        try:
            cnt = rows(s, f"MATCH ()-[r:`{rel}`]->() RETURN count(r) AS c")[0]["c"]
            keys = rows(s, f"MATCH ()-[r:`{rel}`]->() RETURN keys(r) AS k LIMIT 1")[0]["k"] if cnt else []
            pairs = rows(s, f"MATCH (a)-[r:`{rel}`]->(b) RETURN DISTINCT labels(a) AS A, labels(b) AS B LIMIT 10")
            combo = [(p["A"], p["B"]) for p in pairs]
            print(f"  <{rel}> count={cnt} relKeys={keys}")
            print(f"      endpoint combos (A->B): {combo}")
            if cnt:
                one = rows(s, f"MATCH (a)-[r:`{rel}`]->(b) RETURN a AS A, properties(r) AS R, b AS B LIMIT 1")[0]
                a = {k: str(v)[:30] for k, v in dict(one["A"]).items()}
                b = {k: str(v)[:30] for k, v in dict(one["B"]).items()}
                rr = {k: str(v)[:60] for k, v in dict(one["R"]).items()}
                print(f"      sample: ({a}) -[{rel}: {rr}]-> ({b})")
        except Exception as e:
            print(f"  <{rel}> ERR {e}")

driver.close()
