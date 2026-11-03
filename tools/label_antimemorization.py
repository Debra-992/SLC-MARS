# -*- coding: utf-8 -*-
"""防记忆标注工具：把数据集 A 的 40 题分成"可回忆"与"防记忆(题集B)"两类。

动机：baseline 审计发现 pure-llm/autogen/mdagents 均为零检索裸记忆答题，在DepMap
同源文献语料上可能"背过答案"。题集B = 模型记忆覆盖不到的题，用它评测才能体现
管线(图谱/统计证据)相对记忆先验的真实增量。

双探针（均零检索、与 baseline 同端点同温度）：
  probe1 = 复用 outputs/traces/answers_pure_llm.jsonl 的原题面作答（无 paraphrase）；
  probe2 = 新调用，【去掉原题面句式】只给实验约束+候选，用空白填空模板问
           "哪个基因是必需性差异的主要驱动"——干扰逐字记忆线索，考察真实知识。
判定（GT 只在本工具读取做对照，绝不进入任何答题路径）：
  stable = 两探都对（记忆稳固，剔除出B）；novel = 两探都错（纯盲区，B 的核心）；
  unstable = 一对一错（记忆不可靠，也算"记忆不可用"，入B并单列）。
  B := novel ∪ unstable。

env：SLC_PROBE2_TIMEOUT 单次调用超时（默认 120s）；凭证走 baselines.common.llm_settings()。
用法：$env:PYTHONPATH=<repo>; .\.venv\Scripts\python.exe tools\label_antimemorization.py
产物：outputs/traces/antimemorization_labels.json（逐题判定）+ dataset_b_qids.json（B 清单）
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_agents.baselines.common import (  # noqa: E402
    GT_PATH, INPUT_PATH, TRACES, extract_json, llm_settings)

MANIFEST = TRACES / os.environ.get("SLC_LABEL_MANIFEST", "sample_manifest.json")   # 数据集A 的当前 qid 清单（扩测前曾是 v10 定稿 40 题硬编码，2026-10-25 改读现行样本；可用 SLC_LABEL_MANIFEST 覆盖以切全量 489）
PURE_ROWS = TRACES / "answers_pure_llm.jsonl"
LABEL_OUT = TRACES / os.environ.get("SLC_LABEL_OUT", "antimemorization_labels.json")     # 默认同旧产物；全量跑用 SLC_LABEL_OUT 指向独立文件以免覆盖
BQIDS_OUT = TRACES / os.environ.get("SLC_BQIDS_OUT", "dataset_b_qids.json")              # 同上：全量 B 清单另存

# probe2 空白填空模板：不复述原题句式，只给实验设置 + 候选，干扰逐字记忆线索
BLANK_TEMPLATE = """Complete the assessment below using only established domain knowledge of {cancer} CRISPR screens. Do NOT guess from question phrasing.

Experimental setting: {setting}
Candidates: {cands}

The differential essentiality observed in this screen is primarily driven by ____ .

Output JSON only: {{"fill": "<one candidate gene>", "confidence": 0~1, "rationale": "<=25 words"}}"""

# membership 专用模板：谓语句必须原样保留（否则"填基因名"无法折算 true/false），
# 但剥掉原句的实验设定头，改写为断言审判 —— 同样起到干扰逐字记忆线索的作用。
MEMBERSHIP_TEMPLATE = """You are auditing a claimed research finding about {cancer} CRISPR screens, using only established domain knowledge. Do NOT guess from phrasing.

Claim under audit: "{predicate}"
Gene in question: {gene}

Is this claim correct, given what is known about {gene} and {cancer}?

Output JSON only: {{"fill": "true" or "false", "confidence": 0~1, "rationale": "<=25 words"}} (fill="true" means the claim holds, "false" means it does not)."""


def _setting(rec: dict) -> str:
    """从 constraints 重建实验设置文本（不含原题疑问句）。"""
    c = rec.get("constraints", {}) or {}
    parts = [f"{c.get('cancer', '')}", f"{c.get('species', '')}",
             f"{c.get('cell_line', '') or c.get('tissue', '')} as {c.get('sample_type', 'sample')}",
             f"{c.get('treatment', '')} treatment", f"{c.get('method', '')} method"]
    parts = [p.strip() for p in parts if p and p.strip()]
    q = rec.get("question", "")
    head = q.split("Which")[0].split("Is ")[0].split("Between")[0]
    return "; ".join(parts) if parts else head.strip().rstrip(".")


def _pick(row: dict | None, bucket: str):
    if not row:
        return None
    if bucket == "pairwise":
        return str(row.get("winner") or "").upper() or None
    a = row.get("answer")
    return None if a is None else bool(a)


def _probe2_correct(rec: dict, gt: dict, parsed: dict) -> bool | None:
    bucket = rec["bucket"]
    fill = str((parsed or {}).get("fill", "")).strip()
    if not fill:
        return None
    cands = [c.upper() for c in rec.get("candidates", [])]
    if bucket == "pairwise":
        f = fill.upper()
        # 允许 "SLC1A6 rather than SLC25A1" 之类的表述：取首个命中的候选
        f = next((c for c in cands if c in f), "")
        if not f:
            return None
        return f == str(gt.get("winner", "")).upper()
    m = fill.lower()
    if m in {"true", "false"}:
        return (m == "true") == bool(gt.get("answer"))
    # 兼容模型返回自由文本：先判否定（"not true" 含子串 true，顺序不能反）
    if re.search(r"\b(false|incorrect|no|not|wrong)\b", m):
        return False == bool(gt.get("answer"))
    if re.search(r"\b(true|correct|yes|holds)\b", m):
        return True == bool(gt.get("answer"))
    return None


def _membership_predicate(rec: dict) -> str:
    """取 membership 疑问句的核心谓语句（'Is SLC.. inhibit ...?' 尾句），
    重写为断言审判句式（框架改变 = 干扰逐字记忆，但语义必须原样保留）。"""
    q = rec.get("question", "")
    m = re.search(r"(Is\s+\S+\s+.*)$", q, re.S)
    core = (m.group(1) if m else q).strip()
    if core.endswith("?"):
        core = core[:-1]
    if core.lower().startswith("is "):
        core = "It is true that " + core[3:] + "."
    return core


async def probe2_once(cfg: dict, rec: dict) -> dict:
    timeout = float(os.environ.get("SLC_PROBE2_TIMEOUT", "60"))
    from openai import AsyncOpenAI
    cands = [c.upper() for c in rec.get("candidates", [])]
    if rec["bucket"] == "pairwise":
        prompt = BLANK_TEMPLATE.format(
            cancer=rec.get("constraints", {}).get("cancer", "tumor"),
            setting=_setting(rec), cands=", ".join(cands))
    else:
        prompt = MEMBERSHIP_TEMPLATE.format(
            cancer=rec.get("constraints", {}).get("cancer", "tumor"),
            predicate=_membership_predicate(rec), gene=cands[0])
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=timeout, max_retries=0)
    try:
        rsp = await asyncio.wait_for(
            client.chat.completions.create(model=cfg["model"],
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0.2),
            timeout=timeout)
        data = extract_json(rsp.choices[0].message.content or "")
        return {"ok": bool(data), "raw": (rsp.choices[0].message.content or "")[:300],
                "fill": (data or {}).get("fill"), "parsed": data}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass


async def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    a_qids = list(manifest["qids"])
    inputs = {r["qid"]: r for r in json.loads(INPUT_PATH.read_text(encoding="utf-8"))}
    gt_all = json.loads(GT_PATH.read_text(encoding="utf-8"))
    pure: dict[str, dict] = {}
    for line in PURE_ROWS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            pure[r["qid"]] = r            # 后写覆盖先写（同 baseline pending 口径）

    cfg = llm_settings()
    if not cfg["base_url"] or not cfg["api_key"]:
        raise SystemExit("llm_settings 缺 base_url/api_key，无法跑 probe2")

    # 断点续跑：同 template_version 且 probe2 判定有效（True/False）的行直接复用
    TEMPLATE_VERSION = "2"      # v2 = membership 改用断言审判模板（v1 的填空模板对 membership 不可折算）
    prev_labels: dict[str, dict] = {}
    prev_tv = None
    if LABEL_OUT.exists():
        try:
            old = json.loads(LABEL_OUT.read_text(encoding="utf-8"))
            prev_tv = str(old.get("template_version"))
            prev_labels = {l["qid"]: l for l in old.get("labels", [])}
        except Exception:  # noqa BLE001
            prev_tv = None

    # 并发 + 增量落盘（防整轮白跑）：sidecar partial，每完成一题即时 append，可 resume
    partial_path = LABEL_OUT.with_name(LABEL_OUT.stem + ".partial.jsonl")
    if partial_path.exists():
        for ln in partial_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                try:
                    d = json.loads(ln)
                    if d.get("qid"):
                        prev_labels[d["qid"]] = d          # partial 覆盖 LABEL_OUT（更新）
                except Exception:  # noqa BLE001
                    pass

    conc = max(1, int(os.environ.get("SLC_PROBE2_CONC", "6")))
    sem = asyncio.Semaphore(conc)
    wrlock = asyncio.Lock()
    done = {"n": 0}

    async def _one(idx: int, qid: str):
        rec = inputs.get(qid)
        gt = gt_all.get(qid)
        if rec is None or gt is None:
            return idx, {"qid": qid, "class": "missing_input"}, "missing_input", False
        old_l = prev_labels.get(qid)
        if (prev_tv == TEMPLATE_VERSION and old_l and old_l.get("class") != "missing_input"
                and old_l.get("probe2", {}).get("correct") in (True, False)
                and old_l.get("probe1", {}).get("correct") in (True, False)):
            return idx, old_l, old_l["class"], True
        p1 = _pick(pure.get(qid), rec["bucket"])
        c1 = None if p1 is None else (
            p1 == str(gt.get("winner", "")).upper() if rec["bucket"] == "pairwise"
            else p1 == bool(gt.get("answer")))
        async with sem:
            r2 = await probe2_once(cfg, rec)
        c2 = _probe2_correct(rec, gt, r2.get("parsed") or {}) if r2.get("ok") else None
        if c1 is None and c2 is None:
            cls = "probe_error"
        elif c1 and c2:
            cls = "stable"
        elif c1 or c2:
            cls = "unstable"
        else:
            cls = "novel"
        lab = {
            "qid": qid, "bucket": rec["bucket"], "gt": gt,
            "probe1": {"answer": p1, "correct": c1},
            "probe2": {"ok": r2.get("ok"), "answer": (r2.get("parsed") or {}).get("fill"),
                       "correct": c2, "error": r2.get("error")},
            "class": cls,
        }
        async with wrlock:
            with open(partial_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(lab, ensure_ascii=False) + "\n")
            done["n"] += 1
            print(f"[{done['n']:>3}/{len(a_qids)}] {cls:<12} probe1={c1} probe2={c2} {qid[:60]}", flush=True)
        return idx, lab, cls, False

    results = await asyncio.gather(*(_one(i, q) for i, q in enumerate(a_qids, 1)),
                                   return_exceptions=True)
    labels = []
    counters = {"stable": 0, "novel": 0, "unstable": 0, "probe_error": 0, "missing_input": 0}
    n_reused = 0
    for r in results:
        if isinstance(r, Exception):
            print(f"[gather-exc] {type(r).__name__}: {str(r)[:150]}", flush=True)
            continue
        idx, lab, cls, reused = r
        labels.append(lab)
        counters[cls] = counters.get(cls, 0) + 1
        if reused:
            n_reused += 1

    b_qids = [l["qid"] for l in labels if l["class"] in ("novel", "unstable")]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LABEL_OUT.write_text(json.dumps({
        "generated_at": now, "dataset_a_n": len(a_qids), "counters": counters,
        "template_version": TEMPLATE_VERSION, "reused_rows": n_reused,
        "probe_model": cfg["model"], "note": "GT 仅在此标注工具内读取对照；B := novel ∪ unstable",
        "labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8")
    BQIDS_OUT.write_text(json.dumps({
        "generated_at": now, "definition": "pure-LLM 双探针(原题面复用 + 空白填空 paraphrase)非双对",
        "counters": counters, "b_qids": b_qids}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n分类统计: {counters}\n题集B: {len(b_qids)}/{len(a_qids)} -> {BQIDS_OUT}")


if __name__ == "__main__":
    asyncio.run(main())
