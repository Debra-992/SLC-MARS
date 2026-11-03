# -*- coding: utf-8 -*-
"""只读 Cypher 守卫，镜像 SLC multi_agent/tool_operations.validate_cypher 的真实规则。

图谱/文献知识只允许单条只读查询；写子句、存储过程、多语句一律拒绝。
这是 text2cypher 链路的安全边界，绝不可放宽。
"""
from __future__ import annotations

import re

_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|REMOVE|SET|DROP|DETACH|LOAD|FOREACH|INSERT|ALTER|RENAME|"
    r"GRANT|DENY|REVOKE|TRANSACTIONS|TERMINATE|USE|CALL)\b",
    re.I,
)
_ALLOWED_START = re.compile(r"^(MATCH|OPTIONAL\s+MATCH|WITH|RETURN|UNWIND|EXPLAIN)\b", re.I)
_PROC = re.compile(r"\b(APOC|GDS|DBMS)\b", re.I)


class CypherRejected(PermissionError):
    """Cypher 未通过只读校验。"""


def validate_cypher(query: str) -> str:
    """校验并返回可执行的单条只读 Cypher，否则抛 CypherRejected/ValueError。"""
    if not isinstance(query, str) or not query.strip() or len(query) > 20000:
        raise ValueError("Empty or oversized query")
    scrubbed = query.strip().rstrip(";")
    if ";" in scrubbed:
        raise CypherRejected("Only a single statement is allowed")
    if _FORBIDDEN.search(scrubbed) or _PROC.search(scrubbed):
        raise CypherRejected("Only read-only Cypher without procedures is allowed")
    if not _ALLOWED_START.match(scrubbed):
        raise CypherRejected("Query must start with MATCH/OPTIONAL MATCH/WITH/RETURN/UNWIND/EXPLAIN")
    return scrubbed
