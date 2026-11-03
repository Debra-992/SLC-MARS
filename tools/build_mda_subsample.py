# -*- coding: utf-8 -*-
"""生成 mdagents 评测子集清单：pw 抽 100 + mem 抽 100（mem 只从【已答成功】里抽=复用）。

固定 seed(默认5) + 先排序再采样 → 可复现。写出 outputs/traces/mdagents_subsample.json:
  {"seed":5,"pw":[...100],"mem":[...100],"qids":[...200]}
run_mdagents.py 用 SLC_MDA_QIDFILE 指向此文件，只跑清单内且未答的题（mem 已答→自动跳过=复用；pw 未答→新跑）。
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "outputs" / "traces"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=100)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    inputs = json.loads((TRACES / "baselines_input.json").read_text(encoding="utf-8"))
    bucket_of = {i["qid"]: i["bucket"] for i in inputs}
    pw_all = sorted(q for q, b in bucket_of.items() if b == "pairwise")
    mem_all = sorted(q for q, b in bucket_of.items() if b == "membership")

    # 已成功答过的 membership（mem 复用池）
    done_mem_ok: set[str] = set()
    p = TRACES / "answers_mdagents.jsonl"
    if p.exists():
        last: dict[str, dict] = {}
        for l in p.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if l:
                o = json.loads(l)
                last[o["qid"]] = o
        done_mem_ok = {q for q, r in last.items()
                       if r["bucket"] == "membership" and not (r.get("error") or "")}

    rng = random.Random(args.seed)
    pw_pick = rng.sample(pw_all, min(args.per, len(pw_all)))
    mem_pool = sorted(done_mem_ok)                     # 复用：仅从已答 mem 里抽
    mem_pick = rng.sample(mem_pool, min(args.per, len(mem_pool)))

    out = {"seed": args.seed, "per": args.per,
           "pw": sorted(pw_pick), "mem": sorted(mem_pick),
           "qids": sorted(set(pw_pick) | set(mem_pick)),
           "mem_reuse_pool_ok": len(done_mem_ok), "pw_total": len(pw_all), "mem_total": len(mem_all)}
    (TRACES / "mdagents_subsample.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"清单 -> mdagents_subsample.json | pw {len(out['pw'])} / mem {len(out['mem'])} "
          f"(复用池已答 mem={out['mem_reuse_pool_ok']}) / 合计 qids={len(out['qids'])}")
    print(f"pw 需新跑=100（现 0 答），mem 复用=100（池 {out['mem_reuse_pool_ok']}）")


if __name__ == "__main__":
    main()
