# -*- coding: utf-8 -*-
# run_final_master.ps1 —— v20@489 跑满后自动收尾：刷新管线报告 → 端点空闲跑 token 探针 → 出效率-精度总表。
# 全程 idle-gate（绝不与 v500 抢端点，避免超时污染）。停滞检测兜底 v500 意外早退。
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py   = "$root\.venv\Scripts\python.exe"
$env:PYTHONUTF8 = '1'
$log  = "$root\logs\final_master.log"
New-Item -ItemType Directory -Force -Path "$root\logs" | Out-Null

function Log($m) { $ts = Get-Date -Format 'MM-dd HH:mm:ss'; "$ts $m" | Out-File -FilePath $log -Append -Encoding utf8 }

function V20Count {
    (& $py -c "import json;from pathlib import Path;s=set();[s.add(json.loads(l)['qid']) for l in Path('outputs/traces/tracestore.jsonl').open(encoding='utf-8') if l.strip() and (json.loads(l).get('provenance') or {}).get('trace_version')=='20'];print(len(s))").Trim()
}

function Busy {
    @(Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
        Where-Object { $_.CommandLine -match 'compare_eval|run_pure_llm|run_lit_rag|run_autogen|run_metagpt|run_mdagents|label_antimemorization|_r2_oos' })
}

Log "==== 收尾链启动 ===="

# ---- Phase 1: 等 v20 跑满 489（停滞兜底：n 不动且无重活 -> 按现状出表）----
$t0 = Get-Date; $lastN = -1; $stall = 0; $n = 0
while ($true) {
    $n = [int](V20Count)
    if ($n -ge 489) { Log "v20 达标 $n/489"; break }
    if ($n -eq $lastN -and (Busy).Count -eq 0) { $stall++ } else { $stall = 0 }
    $lastN = $n
    if ($stall -ge 6) { Log "v20 停滞于 $n/489 且无重活 -> 按现状出表（可能部分/早退）"; break }
    if (((Get-Date) - $t0).TotalHours -gt 60) { Log "ABORT 等待超时 n=$n"; exit 1 }
    Log "等 v20 跑满: $n/489 (stall=$stall)"; Start-Sleep -Seconds 180
}

# ---- Phase 2: 端点空闲（连读 2 次 0）----
$idle = 0
while ($true) {
    if ((Busy).Count -eq 0) { $idle++ } else { $idle = 0 }
    if ($idle -ge 2) { break }
    Log "等端点空闲... 重活=$((Busy).Count)"; Start-Sleep -Seconds 60
}

# ---- Phase 3: 刷新管线全489报告（满489时 _final_pipeline_full489 自动去掉'未跑满'横幅）----
Log "刷新 pipeline_full489_report.txt"
& $py "tools\_final_pipeline_full489.py" 1>>"$log" 2>>"$log"

# ---- Phase 4: 管线 token 探针（需 slc_mars_env 的端点/DB 环境；默认温度=与 v500 同口径）----
Log "跑管线 token 探针 --n 50（idle 独占）"
try { . "$root\tools\slc_mars_env.ps1" } catch { Log "slc_mars_env dot-source 失败: $_" }
& $py "tools\_pipeline_token_probe.py" --n 50 --seed 5 1>>"$log" 2>>"$log"

# ---- Phase 5: 效率-精度总表 ----
Log "生成 efficiency_accuracy_master.txt"
& $py "tools\_final_master_table.py" 1>>"$log" 2>>"$log"

Log "==== 收尾链完成 ===="
