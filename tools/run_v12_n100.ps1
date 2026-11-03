# v12 冠军定版 × 扩测 n=100（seed=5，同题池 489 可扩；39 题缓存复用、~61 题新算）。
# 门控 = slc_mars_env.ps1 默认（PRIOR=1 DEBATE=1 CHIEF=1，TRIAD/CRITIQUE/RACE 全关 = v12）。
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\slc_mars_env.ps1"
# 显式锁定冠军门控（防外部环境残留）
$env:SLC_PRIOR_ENSEMBLE  = '1'
$env:SLC_ANSWER_ENSEMBLE = '1'
$env:SLC_TRIAD_VOTE      = '0'
$env:SLC_TRIAD_CRITIQUE  = '0'
$env:SLC_PRIOR_RACE      = '1'
$py = "$PSScriptRoot\..\.venv\Scripts\python.exe"
$log = "$PSScriptRoot\..\outputs\traces\v12n100_run.log"
$err = "$PSScriptRoot\..\outputs\traces\v12n100_err.log"
Start-Process -FilePath $py -ArgumentList "-m","knowledge_agents.compare_eval","--n","100","--seed","5" `
    -RedirectStandardOutput $log -RedirectStandardError $err -NoNewWindow
Write-Host "v12 n=100 launched: log=$log"
