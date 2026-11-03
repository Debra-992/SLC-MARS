# -*- coding: utf-8 -*-
"""把 data/questions.jsonl 全量题目转成 MCP project_knowledge 的公开白名单文件。

slc_benchmark_context_mcp._public_item 读取 {"cases":[{"item": SanitizedQuestion,...}]}
并在加载时用 SanitizedQuestion 逐条校验（extra=forbid / constraints 仅白名单键 /
task_type 前缀须匹配 task_bucket）。本脚本在本地复刻同一套校验，任何一条不合法都会
导致服务端整个文件加载失败 -> 全部 qid rejected，所以先全部验过再落盘。

用法： python tools/gen_public_inputs.py  [输出路径]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "questions.jsonl"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT / "outputs" / "mcp-public-inputs-all647.json")

PUBLIC_CONSTRAINTS = {"cancer", "species", "cell_line", "treatment", "method", "family", "k"}
TASK_BUCKETS = ("topk", "pairwise", "membership")
ITEM_FIELDS = {"qid", "question", "task_type", "task_bucket", "constraints"}


def validate(item: dict) -> None:
    """镜像 SanitizedQuestion 的 model + validator 约束。"""
    if set(item) - ITEM_FIELDS:
        raise ValueError(f"extra fields: {sorted(set(item) - ITEM_FIELDS)}")
    if not item.get("qid") or not item.get("question") or not item.get("task_type"):
        raise ValueError("empty required field")
    bucket = next((b for b in TASK_BUCKETS if item["task_type"].startswith(b)), "other")
    if bucket != item.get("task_bucket"):
        raise ValueError(f"task_type {item['task_type']!r} != bucket {item.get('task_bucket')!r}")
    unknown = set(item.get("constraints", {})) - PUBLIC_CONSTRAINTS
    if unknown:
        raise ValueError(f"non-public constraints: {sorted(unknown)}")


def main() -> None:
    cases = []
    seen = set()
    with SRC.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            validate(item)
            if item["qid"] in seen:
                raise SystemExit(f"duplicate public qid: {item['qid']}")
            seen.add(item["qid"])
            cases.append({"item": item})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
