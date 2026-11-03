# v16 启动器：三通道表决 + 交叉批评 + 竞速（TRACE_VERSION=16 全量重算 40 题）。
# 用法：powershell -File tools\run_v16.ps1（Start-Process 分离进程，日志见 outputs\traces\v16_*.log）
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\slc_mars_env.ps1"
$env:SLC_TRIAD_VOTE       = '1'   # v15/v16 三通道表决
$env:SLC_TRIAD_CRITIQUE   = '1'   # v16 冲突态并行交叉批评
$env:SLC_PRIOR_RACE       = '2'   # v16 L1/L2 双发竞速（首个有效票）
$env:SLC_PRIOR_TIMEOUT    = '240'
$env:SLC_JUDGE_TIMEOUT    = '240'
$env:SLC_CRITIQUE_TIMEOUT = '180'
$py = "$PSScriptRoot\..\.venv\Scripts\python.exe"
$log = "$PSScriptRoot\..\outputs\traces\v16_run.log"
$err = "$PSScriptRoot\..\outputs\traces\v16_err.log"
Start-Process -FilePath $py -ArgumentList "-m","knowledge_agents.compare_eval","--n","40","--seed","5" `
    -RedirectStandardOutput $log -RedirectStandardError $err -NoNewWindow
Write-Host "v16 launched: TRIAD=1 CRITIQUE=1 RACE=2 log=$log"
