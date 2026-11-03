# v16@n=100 接力器：等扩测下游链（含 mdagents + merge）完全结束后，自动串跑 v16 三通道表决
# 于同一 100 题（seed=5），再产出 v16 vs v12 vs mdagents 三方终表。
# 严格串行设计源自 v15 教训：绝不同时抢占 LLM 端点，避免长尾超时污染弃权率。
# 用法：powershell -File tools\run_v16_n100.ps1（由 Bash 以分离进程拉起；进度见 outputs\traces\v16n100_chain.log）
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\slc_mars_env.ps1"
$T     = "$PSScriptRoot\..\outputs\traces"
$py    = "$PSScriptRoot\..\.venv\Scripts\python.exe"
$chain = "$T\expansion_chain.log"
$blog  = "$T\v16n100_chain.log"

function Note($m) { "[$(Get-Date -Format 'HH:mm:ss')] $m" | Add-Content $blog -Encoding utf8 }

Note "waiting for expansion chain (mdagents + merge) to finish..."
while ($true) {
    $done = (Test-Path $chain) -and ((Get-Content $chain -Encoding utf8 -ErrorAction SilentlyContinue) -match 'CHAIN DONE').Count -gt 0
    $busy = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
             Where-Object { $_.CommandLine -match 'compare_eval|run_mdagents|run_autogen|merge_baselines|run_pure_llm|run_lit_rag|label_antimemorization' }).Count -gt 0
    if ($done -and -not $busy) { break }
    Start-Sleep -Seconds 120
}
Note "chain done; launching v16 n=100 (TRIAD=1 CRITIQUE=1 RACE=2 TRACE_VERSION=16)"

# v16 门控（在 slc_mars_env 冠军默认之上开表决三件套 + 版本覆盖，v12 行不受影响、并存于 tracestore）
$env:SLC_TRACE_VERSION   = '16'
$env:SLC_TRIAD_VOTE      = '1'
$env:SLC_TRIAD_CRITIQUE  = '1'
$env:SLC_PRIOR_RACE      = '2'
$env:SLC_PRIOR_TIMEOUT   = '240'
$env:SLC_JUDGE_TIMEOUT   = '240'
$env:SLC_CRITIQUE_TIMEOUT = '180'

& $py -m knowledge_agents.compare_eval --n 100 --seed 5 *>&1 |
    Add-Content "$T\v16n100_run.log" -Encoding utf8
Note "v16 run exited; writing final 3-way table"

& $py "$PSScriptRoot\_final_n100.py" *>&1 | Add-Content "$T\v16n100_final.log" -Encoding utf8
Note "V16 N100 CHAIN DONE"
