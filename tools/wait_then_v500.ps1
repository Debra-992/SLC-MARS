# 串行等待器：轮询直到池中无任何"重活"python 进程（当前 4 个 baseline 评测跑完），再自动接跑 v20 扩500。
# 不杀任何进程；起跑仍由 run_v20_n500.ps1 内的互斥二次把关，避免过booking超时灾难。
$ErrorActionPreference = 'Stop'
$ROOT = Split-Path -Parent $PSScriptRoot
$LOG = Join-Path $ROOT "outputs\traces\v20n500_chain.log"
function Log($m) { "[$((Get-Date).ToString('HH:mm:ss'))] $m" | Add-Content -Path $LOG -Encoding UTF8 }
$heavy = 'compare_eval|run_pure_llm|run_lit_rag|run_autogen|run_mdagents|label_antimemorization'

Log "等待器启动：等当前 baseline 重活清空后接跑 v20 扩500"
$idleRounds = 0
while ($true) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object { $_.CommandLine -match $heavy })
    if ($busy.Count -eq 0) {
        $idleRounds++
        if ($idleRounds -ge 2) { break }   # 连续两轮空 = 真排空（防瞬时抖动）
    } else {
        $idleRounds = 0
        Log "池中 $($busy.Count) 个重活进程，继续等待"
    }
    Start-Sleep -Seconds 300
}
Log "池已排空，60s 缓冲（等文件 flush）后接跑"
Start-Sleep -Seconds 60
& (Join-Path $PSScriptRoot "run_v20_n500.ps1")
