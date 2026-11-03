# v20 + baselines 结束后串行跑 topk 评测（pipeline + pure_llm）
# 零 overbooking：等池排空再逐个系统跑
$ErrorActionPreference = 'Stop'
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$T = Join-Path $ROOT 'outputs\traces'
$LOG = Join-Path $T 'topk_eval_chain.log'
function Log($m) { "[$((Get-Date).ToString('HH:mm:ss'))] $m" | Add-Content -Path $LOG -Encoding UTF8 }
$py = Join-Path $ROOT '.venv\Scripts\python.exe'
$heavy = 'compare_eval|run_pure_llm|run_lit_rag|run_autogen|run_mdagents|label_antimemorization|_topk_eval'

Log 'topk eval chain: waiting for pool to drain'

# 1) wait for drain
$idleRounds = 0
while ($true) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object { $_.CommandLine -match $heavy })
    if ($busy.Count -eq 0) {
        $idleRounds++
        if ($idleRounds -ge 2) { break }
    } else {
        $idleRounds = 0
        Log "pool busy: $($busy.Count) processes"
    }
    Start-Sleep -Seconds 120
}
Log 'pool drained, 60s flush buffer'
Start-Sleep -Seconds 60

. (Join-Path $ROOT 'tools\slc_mars_env.ps1')

# 2) pipeline topk eval（--hint ppr_order：候选按图谱序给 LLM，LLM 只做重排）
Log '=== pipeline topk eval start ==='
& $py (Join-Path $PSScriptRoot '_topk_eval.py') --system pipeline --hint ppr_order --seeds 42,7,13 *>&1 | Out-File (Join-Path $T 'topk_eval_pipeline.txt') -Encoding UTF8
Log "pipeline topk done exit=$LASTEXITCODE"

# 3) pure_llm topk eval（--hint none：整池乱序，剥离图谱顺序，才是真·裸 LLM）
Log '=== pure_llm topk eval start ==='
& $py (Join-Path $PSScriptRoot '_topk_eval.py') --system pure_llm --hint none --seeds 42,7,13 *>&1 | Out-File (Join-Path $T 'topk_eval_pure_llm.txt') -Encoding UTF8
Log "pure_llm topk done exit=$LASTEXITCODE"

# 4) comparison summary
Log '=== topk comparison summary ==='
& $py (Join-Path $PSScriptRoot '_topk_compare.py') *>&1 | Out-File (Join-Path $T 'topk_eval_summary.txt') -Encoding UTF8
Log 'topk eval chain ALL DONE'