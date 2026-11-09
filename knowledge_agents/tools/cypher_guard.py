# -*- coding: utf-8 -*-
"""Read-only Cypher guard, mirroring the real rules of
SLC multi_agent/tool_operations.validate_cypher.

Graph and literature knowledge may only be fetched with a single read-only query; write
clauses, stored procedures and multi-statement text are always rejected. This is the
security boundary of the text2cypher path and must never be relaxed.
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
    """Cypher failed the read-only validation."""


def validate_cypher(query: str) -> str:
    """Validate and return an executable single read-only Cypher statement, otherwise raise
    CypherRejected/ValueError."""
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
