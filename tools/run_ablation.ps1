# 消融实验启动器：串行跑两路消融（先知识、后计算），全量 n=100。
$ErrorActionPreference = 'Stop'
$ROOT = Split-Path -Parent $PSScriptRoot
$LOG = Join-Path $ROOT "outputs\traces\ablation_chain.log"
function Log($m) { "[$((Get-Date).ToString('HH:mm:ss'))] $m" | Add-Content -Path $LOG -Encoding UTF8 }

# 串行互斥：不与 compare_eval 或其他重活争端点
$heavy = 'compare_eval|_ablation'
$busy = @(Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object { $_.CommandLine -match $heavy })
if ($busy.Count -gt 0) {
    Log "ABORT: $($busy.Count) heavy python processes detected. Manual kill required."
    Write-Host "ABORT: heavy processes running - check $LOG"
    exit 1
}

# 1) 消融知识智能体 (ablate_knowledge)
Log "=== START ablate_knowledge (n=100) ==="
. (Join-Path $ROOT "tools\slc_mars_env.ps1")
try {
    & (Join-Path $ROOT ".venv\Scripts\python.exe") (Join-Path $ROOT "tools\_ablation.py") --mode ablate_knowledge --n 100 *>&1 | Out-File (Join-Path $ROOT "outputs\traces\ablation_knowledge.log") -Encoding UTF8
    $exit = $LASTEXITCODE
} catch { Log "crash: $_"; $exit = -1 }
Log "ablate_knowledge exit=$exit"

# 2) 消融计算智能体 (ablate_compute)
Log "=== START ablate_compute (n=100) ==="
. (Join-Path $ROOT "tools\slc_mars_env.ps1")
try {
    & (Join-Path $ROOT ".venv\Scripts\python.exe") (Join-Path $ROOT "tools\_ablation.py") --mode ablate_compute --n 100 *>&1 | Out-File (Join-Path $ROOT "outputs\traces\ablation_compute.log") -Encoding UTF8
    $exit = $LASTEXITCODE
} catch { Log "crash: $_"; $exit = -1 }
Log "ablate_compute exit=$exit"
Log "ALL DONE"
Write-Host "消融实验完成 - 检查 outputs/traces/ablation_*.log"