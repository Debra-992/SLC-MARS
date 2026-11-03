# -*- coding: utf-8 -*-
"""无人值守收尾编排器（detached 后台跑，不占对话）。

职责：
  1) 监控 pure_llm / lit_rag / autogen / metagpt 四个 489-全量 baseline runner；
     （mdagents 走独立 100pw+100mem 子集，由 tools/mdagents_subsample.py 自管，不在此列）
     某系统的 python 进程【退出后】若仍有 error 行（端点超时→abstain），用
     run_baselines_all500.ps1 重启该系统一次以补跑（pending() 只捞失败行）。每系统至多 MAXPASS 轮。
  2) 四系统全部收敛（成功覆盖到位或轮次耗尽）或总时长到 CAP_H 后，生成：
       - outputs/traces/baseline_final_report.txt：干净终值（489）+ 与 pipeline 同题(100-A) 对齐表。
       - outputs/traces/auto_finish.DONE 哨兵。
  绝不 -Kill、绝不 -Fresh、绝不碰 compare_eval(v20)；只对单系统幂等重启。

用法（由主进程 detached 拉起）:
  Start-Process .venv\\Scripts\\python.exe -ArgumentList 'tools\\auto_finish_baselines.py' -WindowStyle Hidden
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRACES = ROOT / "outputs" / "traces"
LAUNCHER = str(ROOT / "tools" / "run_baselines_all500.ps1")
LOGF = TRACES / "auto_finish.log"
REPORT = TRACES / "baseline_final_report.txt"
DONE = TRACES / "auto_finish.DONE"

SYSTEMS = ["pure_llm", "lit_rag", "autogen", "metagpt"]   # mdagents 已改为独立 100pw+100mem 子集评测(见 mdagents_subsample.py)，不再跨 489
MAXPASS = 3          # 每系统最多补跑轮数（专治瞬时超时；每轮只重跑 error 行）
POLL = 150           # 秒
CAP_H = 14           # 总时长上限（小时）
# 各系统重启附带参数（不改题集/不杀进程；只为口径一致：mdagents 保持 agents=3、autogen 关缓存高超时）
EXTRA = {
    "pure_llm": " -LlmTimeout 120",
    "lit_rag": " -LlmTimeout 120",
    "autogen": " -LlmTimeout 180",
    "mdagents": " -MdaRounds 2 -MdaTurns 2 -MdaTeams 2 -MdaTeamMembers 2 -MdaAgents 3 -LlmTimeout 180",
    "metagpt": "",   # 原生 MetaGPT 单 _aask，超时已烧进 runner(SLC_BASELINE_LLM_TIMEOUT=120)
}


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}"
    try:
        with LOGF.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def _ps(script: str) -> str:
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                       capture_output=True, text=True, timeout=90)
    return (r.stdout or "").strip()


def running(sysname: str) -> int:
    """匹配 run_<sysname> 的 python 进程数（含 uv 父子）。"""
    script = ("(Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
              "Where-Object { $_.CommandLine -match 'run_" + sysname + "' } | Measure-Object).Count")
    try:
        return int(_ps(script) or "0")
    except (ValueError, subprocess.SubprocessError):
        return -1


def read_rows(sysname: str) -> dict[str, dict]:
    p = TRACES / f"answers_{sysname}.jsonl"
    for _ in range(5):
        try:
            if not p.exists():
                return {}
            out: dict[str, dict] = {}
            for l in p.read_text(encoding="utf-8").splitlines():
                l = l.strip()
                if l:
                    try:
                        o = json.loads(l); out[o["qid"]] = o
                    except json.JSONDecodeError:
                        continue
            return out
        except (PermissionError, OSError):
            time.sleep(2)
    return {}


def relaunch(sysname: str) -> None:
    """发射即忘：run_baselines_all500.ps1 内部用 Start-Process 投递后台进程后立即返回，
    故不等待、不捕获输出（避免 powershell 冷启动/负载下阻塞→抛 TimeoutExpired 拖垮主循环）。
    真正的成败由下一轮 running()/read_rows() 验证。"""
    cmd = f"& '{LAUNCHER}' -Systems @('{sysname}'){EXTRA.get(sysname, '')}"
    try:
        subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:  # noqa BLE001
        log(f"relaunch {sysname} Popen 失败(忽略): {e}")
    time.sleep(5)   # 给 launcher 一点时间完成投递


def eval_inputs() -> list[dict]:
    from knowledge_agents.baselines.common import load_inputs
    return [i for i in load_inputs() if i["bucket"] in {"pairwise", "membership"}]


def build_report(inputs: list[dict]) -> str:
    from knowledge_agents.baselines.common import GT_PATH
    from knowledge_agents.baselines.merge_baselines import outcome_of
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    bucket_of = {i["qid"]: i["bucket"] for i in inputs}
    allq = [i["qid"] for i in inputs]

    def acc(by, keep):
        c = w = a = 0
        for q in keep:
            r = by.get(q)
            if r is None or (r.get("error") or ""):
                a += 1
                continue
            o = outcome_of(r, gt.get(q, {}), bucket_of[q])
            if o == "correct":
                c += 1
            elif o == "wrong":
                w += 1
            else:
                a += 1
        n = c + w + a
        return c, w, a, n, (c / n if n else -1), (c / (c + w) if (c + w) else -1)

    pw_q = [q for q in allq if bucket_of[q] == "pairwise"]
    mem_q = [q for q in allq if bucket_of[q] == "membership"]
    rows = {}
    for s in SYSTEMS:
        rows[s] = read_rows(s)
    pct = lambda v: "%.1f%%" % (v * 100) if v >= 0 else "-"

    L = []
    L.append(f"# Baseline 干净终值表  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]  489 评测集 (pw {len(pw_q)} / mem {len(mem_q)})")
    L.append("  %-9s %-4s %-6s %-8s %-8s %-8s %-7s" % ("sys", "出", "acc答", "acc全", "pw", "mem", "均tok"))
    for s in SYSTEMS:
        by = rows[s]; out = sum(1 for q in allq if q in by and not (by[q].get("error") or ""))
        c, w, a, n, accall, accans = acc(by, allq)
        _, _, _, npv, _, accp = acc(by, pw_q)
        _, _, _, nmv, _, accm = acc(by, mem_q)
        ok = [by[q] for q in by if (by[q].get("tokens") or {}).get("total", 0) > 0]
        avgtok = round(sum(r["tokens"]["total"] for r in ok) / len(ok)) if ok else 0
        L.append("  %-9s %-4d %-6s %-8s %-8s %-8s %-7d" % (
            s, out, pct(accans), pct(accall), pct(accp) if npv else "-", pct(accm) if nmv else "-", avgtok))

    # ---- 与 pipeline 同题对齐：100-A manifest ∩ 489 ----
    L.append("")
    try:
        manifest = json.loads((TRACES / "sample_manifest.json").read_text(encoding="utf-8"))
        a_qids = set(manifest["qids"]) & set(allq)
        from importlib import import_module
        fv = import_module("tools._final_v20")
        arms = {
            "v16-auto": fv.trace_by_version("pipeline_auto", "16"),
            "v20-auto": fv.trace_by_version("pipeline_auto", "20"),
        }
        L.append(f"# 与 pipeline 同题对齐（100-A ∩ 489 = {len(a_qids)} 题）")
        L.append("  %-10s %-6s %-8s" % ("arm", "n答", "acc"))
        for name in ("v16-auto", "v20-auto"):
            by = arms[name]
            cc = sum(1 for q in a_qids if by.get(q) == "correct")
            an = sum(1 for q in a_qids if by.get(q) in ("correct", "wrong"))
            L.append("  %-10s %-6d %-8s" % (name, an, pct(cc / an) if an else "-"))
        for s in SYSTEMS:
            by = rows[s]
            c, w, a, n, accall, accans = acc(by, list(a_qids))
            L.append("  %-10s %-6d %-8s" % (s, c + w, pct(accans) if (c + w) else "-"))
    except Exception as e:  # noqa BLE001
        L.append("  (对齐表生成失败: %s)" % e)
    return "\n".join(L)


def main() -> None:
    start = time.time()
    inputs = eval_inputs()
    allq = {i["qid"] for i in inputs}
    passes = {s: 0 for s in SYSTEMS}
    finalized = {s: False for s in SYSTEMS}
    log(f"START 收尾编排 systems={SYSTEMS} eval_n={len(allq)} MAXPASS={MAXPASS} CAP_H={CAP_H}")
    while time.time() - start < CAP_H * 3600:
      try:
        all_done = True
        for s in SYSTEMS:
            if finalized[s]:
                continue
            nrun = running(s)
            if nrun < 0:
                all_done = False
                continue
            if nrun > 0:               # 首轮/补跑仍在进行
                all_done = False
                continue
            # 进程已退出：判断是否收敛
            by = read_rows(s)
            err = sum(1 for q in by if (by[q].get("error") or ""))
            cov = sum(1 for q in by if q in allq and not (by[q].get("error") or ""))
            if cov >= len(allq):        # 全成功
                finalized[s] = True
                log(f"FINALIZED {s} cov={cov}/{len(allq)}")
            elif passes[s] < MAXPASS:   # 仍有 error/缺口且未耗尽轮次 → 补跑
                passes[s] += 1
                log(f"RELAUNCH #{passes[s]} {s} cov={cov}/{len(allq)} err={err}")
                relaunch(s)
                all_done = False
            else:                        # 轮次耗尽仍不完美 → 定格
                finalized[s] = True
                log(f"FINALIZED(capped) {s} cov={cov}/{len(allq)} err={err}")
        if all_done and all(finalized.values()):
            break
      except Exception as e:  # noqa BLE001 —— 多时长观测器：任何单次异常都不能杀死主循环
        log(f"LOOP-ERR(忽略续跑): {type(e).__name__}: {e}")
      time.sleep(POLL)
    rep = build_report(inputs)
    REPORT.write_text(rep + "\n", encoding="utf-8")
    DONE.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    log("REPORT written -> outputs/traces/baseline_final_report.txt")
    print(rep, flush=True)


if __name__ == "__main__":
    main()
