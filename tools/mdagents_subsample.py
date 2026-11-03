# -*- coding: utf-8 -*-
"""mdagents 独立子集编排器（detached）：在 100pw+100mem 子集上评测 mdagents。

- 只跑清单内未答题（pw 100 新跑；mem 100 已答=复用，pending 自动跳过）。
- 每轮用 mdagents-venv 跑 run_mdagents.py（env SLC_MDA_QIDFILE 限定子集 + 降档参数）。
- 进程退出后按清单算成功覆盖；仍有 error 且未耗尽轮次 → 补跑（只重跑 error 行）。
- 收敛（清单全覆盖 / 轮次耗尽 / 总时长到 CAP_H）后写 outputs/traces/mdagents_subsample_report.txt
  与哨兵 mdagents_subsample.DONE。绝不碰 489-orchestrator 的四个系统。

用法（主进程 detached 拉起）:
  Start-Process .venv\\Scripts\\python.exe -ArgumentList 'tools\\mdagents_subsample.py' -WindowStyle Hidden
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRACES = ROOT / "outputs" / "traces"
MANIFEST = TRACES / "mdagents_subsample.json"
MDA_PY = ROOT / "outputs" / "mdagents-venv" / "Scripts" / "python.exe"
RUNNER = "knowledge_agents\\baselines\\run_mdagents.py"
LOGF = TRACES / "mdagents_subsample.log"
REPORT = TRACES / "mdagents_subsample_report.txt"
DONE = TRACES / "mdagents_subsample.DONE"

MAXPASS = 3
POLL = 120
CAP_H = 16
# 降档：与之前全量一致（agents=3 + 2/2/2/2 + 高超时）
MDA_ENV = {
    "SLC_MDA_ROUNDS": "2", "SLC_MDA_TURNS": "2", "SLC_MDA_TEAMS": "2",
    "SLC_MDA_TEAM_MEMBERS": "2", "SLC_MDA_AGENTS": "3",
    "SLC_BASELINE_LLM_TIMEOUT": "180", "SLC_MDAGENTS_TIMEOUT": "1800",
    "PYTHONUTF8": "1", "GIT_PYTHON_REFRESH": "quiet",
}


def log(m: str) -> None:
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {m}"
    try:
        LOGF.open("a", encoding="utf-8").write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def rows() -> dict[str, dict]:
    p = TRACES / "answers_mdagents.jsonl"
    out: dict[str, dict] = {}
    if p.exists():
        for l in p.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if l:
                try:
                    o = json.loads(l); out[o["qid"]] = o
                except json.JSONDecodeError:
                    pass
    return out


def covered(mf: dict) -> tuple[int, int, int]:
    by = rows()
    scope = set(mf["qids"])
    ok = sum(1 for q in scope if q in by and not (by[q].get("error") or ""))
    err = sum(1 for q in scope if q in by and (by[q].get("error") or ""))
    return ok, err, len(scope)


def launch_once() -> None:
    env = os.environ.copy()
    env.update(MDA_ENV)
    env["SLC_MDA_QIDFILE"] = str(MANIFEST)
    with open(TRACES / "bl_mdagents_sub.log", "a", encoding="utf-8") as out:
        subprocess.run([str(MDA_PY), RUNNER], cwd=str(ROOT), env=env,
                       stdout=out, stderr=subprocess.STDOUT)


def build_report(mf: dict) -> str:
    from knowledge_agents.baselines.common import GT_PATH, load_inputs
    from knowledge_agents.baselines.merge_baselines import outcome_of
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    bucket_of = {i["qid"]: i["bucket"] for i in load_inputs()}
    by = rows()
    pct = lambda v: "%.1f%%" % (v * 100) if v is not None else "-"

    def stat(keep):
        c = w = a = 0
        for q in keep:
            r = by.get(q)
            if r is None or (r.get("error") or ""):
                a += 1; continue
            o = outcome_of(r, gt.get(q, {}), bucket_of[q])
            if o == "correct":
                c += 1
            elif o == "wrong":
                w += 1
            else:
                a += 1
        n = c + w
        return {"出": n, "对": c, "错": w, "缺/弃": a, "tot": len(keep),
                "acc已答": (c / n if n else None), "acc全": (c / len(keep) if keep else None)}

    pw = stat(mf["pw"]); mem = stat(mf["mem"]); allst = stat(mf["qids"])
    ok = [by[q] for q in by if q in set(mf["qids"]) and (by[q].get("tokens") or {}).get("total", 0) > 0]
    avgtok = round(sum(r["tokens"]["total"] for r in ok) / len(ok)) if ok else 0
    avgcall = round(sum(r["tokens"]["calls"] for r in ok) / len(ok), 1) if ok else 0
    L = [f"# mdagents 子集评测 (pw{len(mf['pw'])}/mem{len(mf['mem'])}, seed={mf.get('seed')}) "
         f"[{datetime.now():%Y-%m-%d %H:%M}]",
         "  mem 复用已答(未重跑), pw 新跑; 口径同 489 baseline(已答=剔除缺/弃/超时)"]
    for name, s in [("总体", allst), ("pairwise", pw), ("membership", mem)]:
        L.append(f"  {name:<10} 出={s['出']:<4}/{s['tot']:<4} 对={s['对']} 错={s['错']} 缺/弃={s['缺/弃']} | "
                 f"acc已答={pct(s['acc已答'])} acc全={pct(s['acc全'])}")
    L.append(f"  成本: 均 tok/题={avgtok} 均 calls/题={avgcall}")
    L.append(f"  锚点: 自研 v16@100-A 70.0% | pure_llm@489 已答64.1% | autogen@489 已答68.9%")
    return "\n".join(L)


def main() -> None:
    if not MANIFEST.exists():
        log("缺 mdagents_subsample.json，先跑 build_mda_subsample.py"); return
    mf = manifest()
    start = time.time()
    log(f"START mdagents 子集 scope={len(mf['qids'])} (pw{len(mf['pw'])}/mem{len(mf['mem'])}) MAXPASS={MAXPASS}")
    passes = 0
    while time.time() - start < CAP_H * 3600:
        try:
            ok, err, tot = covered(mf)
            if ok >= tot:
                log(f"FINALIZED cov={ok}/{tot}"); break
            if passes >= MAXPASS:
                log(f"轮次耗尽定格 cov={ok}/{tot} err={err}"); break
            passes += 1
            log(f"PASS #{passes} 覆盖={ok}/{tot} err={err} -> 启动 run_mdagents(子集)")
            launch_once()
            ok2, err2, tot2 = covered(mf)
            log(f"PASS #{passes} 结束 覆盖={ok2}/{tot2} err={err2}")
            time.sleep(POLL)
        except Exception as e:  # noqa BLE001
            log(f"LOOP-ERR(忽略续跑): {type(e).__name__}: {e}")
            time.sleep(POLL)
    REPORT.write_text(build_report(manifest()) + "\n", encoding="utf-8")
    DONE.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    log("REPORT -> outputs/traces/mdagents_subsample_report.txt")
    print(REPORT.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
