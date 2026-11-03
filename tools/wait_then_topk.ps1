# 串行等待器：等 v500 扩池跑完后自动接跑 topk 评测（pipeline + pure_llm + 对比表）
$ErrorActionPreference = 'Stop'
$ROOT = Split-Path -Parent $PSScriptRoot
$LOG = Join-Path $ROOT "outputs\traces\topk_eval_chain.log"
function Log($m) { "[$((Get-Date).ToString('HH:mm:ss'))] $m" | Add-Content -Path $LOG -Encoding UTF8 }

# 1) 等 v500 排空（只检查 compare_eval，不含其他 baseline）
Log "等 v500 compare_eval 清空..."
$idleRounds = 0
while ($true) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object { $_.CommandLine -match 'compare_eval' })
    if ($busy.Count -eq 0) {
        $idleRounds++
        if ($idleRounds -ge 2) { break }
    } else {
        $idleRounds = 0
        Log "v500 running: $($busy.Count) processes"
    }
    Start-Sleep -Seconds 120
}
Log "v500 drained, 60s flush buffer"
Start-Sleep -Seconds 60

# 2) pipeline topk eval（--hint ppr_order：候选按图谱序给 LLM，LLM 只做重排）
Log "=== START pipeline topk eval ==="
. (Join-Path $ROOT "tools\slc_mars_env.ps1")
$exit = 1
try {
    & (Join-Path $ROOT ".venv\Scripts\python.exe") (Join-Path $ROOT "tools\_topk_eval.py") --system pipeline --hint ppr_order --seeds 42,7,13 *>&1 | Out-File (Join-Path $ROOT "outputs\traces\topk_eval_pipeline.txt") -Encoding UTF8
    $exit = $LASTEXITCODE
} catch { Log "pipeline crash: $_" }
Log "pipeline exit=$exit"

# 3) pure_llm topk eval（--hint none：整池乱序，剥离图谱顺序，才是真·裸 LLM）
Log "=== START pure_llm topk eval ==="
. (Join-Path $ROOT "tools\slc_mars_env.ps1")
$exit = 1
try {
    & (Join-Path $ROOT ".venv\Scripts\python.exe") (Join-Path $ROOT "tools\_topk_eval.py") --system pure_llm --hint none --seeds 42,7,13 *>&1 | Out-File (Join-Path $ROOT "outputs\traces\topk_eval_pure_llm.txt") -Encoding UTF8
    $exit = $LASTEXITCODE
} catch { Log "pure_llm crash: $_" }
Log "pure_llm exit=$exit"

# 4) comparison summary
Log "=== comparison table ==="
. (Join-Path $ROOT "tools\slc_mars_env.ps1")
& (Join-Path $ROOT ".venv\Scripts\python.exe") (Join-Path $ROOT "tools\_topk_compare.py") *>&1 | Out-File (Join-Path $ROOT "outputs\traces\topk_eval_summary.txt") -Encoding UTF8
Log "DONE - check outputs/traces/topk_eval_summary.txt"