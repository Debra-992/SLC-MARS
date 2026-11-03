# -*- coding: utf-8 -*-
# run_token_probe.ps1 —— 独立补跑管线 token 探针（修收尾链同文件重定向冲突）。
# 探针输出写【独立日志】logs\token_probe.log，绝不与主链 log 抢句柄；跑完自动刷新效率-精度总表。
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py   = "$root\.venv\Scripts\python.exe"
$env:PYTHONUTF8 = '1'
$plog = "$root\logs\token_probe.log"
$rlog = "$root\logs\token_probe_run.log"   # 探针 stdout/stderr 独立文件，绝不与 Log(Out-File) 抢同一句柄
New-Item -ItemType Directory -Force -Path "$root\logs" | Out-Null
function Log($m) { $ts = Get-Date -Format 'MM-dd HH:mm:ss'; "$ts $m" | Out-File -FilePath $plog -Append -Encoding utf8 }

Log "==== 管线 token 探针补跑起 ===="
# 端点须空闲（绝不与别的重活抢，避免超时污染）
$busy = @(Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object { $_.CommandLine -match 'compare_eval|run_pure_llm|run_lit_rag|run_autogen|run_metagpt|run_mdagents|label_antimemorization|_pipeline_token_probe' -and $_.Id -ne $PID })
Log "起前重活进程数=$($busy.Count)"

try { . "$root\tools\slc_mars_env.ps1" } catch { Log "slc_mars_env dot-source 失败: $_" }
Log "env 就绪，跑探针 --n 50 --seed 5（探针 stdout/stderr 独立落本文件）"

# 关键：stdout 先开一次句柄(1>>$rlog)，stderr 用 2>&1 复制它——绝不用 2>>$rlog（双开同文件句柄会 IOException 秒退）
& $py "tools\_pipeline_token_probe.py" --n 50 --seed 5 1>>$rlog 2>&1

if (Test-Path "$root\outputs\traces\pipeline_token_probe.json") {
    Log "探针产物 OK，刷新效率-精度总表"
    & $py "tools\_final_master_table.py" 1>>$rlog 2>&1
    Log "==== 完成：主表已含管线 token/题 ===="
} else {
    Log "!! 探针未产出 json（可能端点抖动/全 ERR），主表保持待探针"
    Log "==== 结束（失败） ===="
}
