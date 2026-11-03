# -*- coding: utf-8 -*-
# v17 判决臂：L2 票源去同源化（SLC_PRIOR_RETRIEVE=1）@ n=100，串行独占 + 内网预检。
# 预检失败即拒绝起跑（宁等不污染： Mongo 断时 pack 恒退化 = 白跑一次 6h 的 v16 复测）。
$ErrorActionPreference = 'Stop'
$ROOT = Split-Path -Parent $PSScriptRoot   # tools 的上一层 = 仓库根（注意：不是 $MyInvocation 写法）
Set-Location $ROOT
$T = Join-Path $ROOT "outputs\traces"
$LOG = Join-Path $T "v17_chain.log"
function Log($m) { "[$((Get-Date).ToString('HH:mm:ss'))] $m" | Add-Content -Path $LOG -Encoding UTF8 }
$py = Join-Path $ROOT ".venv\Scripts\python.exe"

# 1) 串行互斥：任何重活在场就不起跑（compare_eval/五基线/label）
$busy = Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object {
    $_.CommandLine -match 'compare_eval|run_pure_llm|run_lit_rag|run_autogen|run_mdagents|label_antimemorization' }
if ($busy) { Log "ABORT: 检出 $($busy.Count) 个在跑重活进程，串行保护"; exit 1 }

. (Join-Path $ROOT "tools\slc_mars_env.ps1")

# 2) 内网预检（v17 特有依赖：papers 检索包）——mongo 可达才允许起跑
$mongoOk = & $py -c "import os;from pymongo import MongoClient;c=MongoClient(os.environ['MONGO_URI'],serverSelectionTimeoutMS=8000);print(c.admin.command('ping'));print(c[os.environ.get('MONGO_DB_NAME','SLCdb')].papers.estimated_document_count())" 2>&1
$mongoOk = ($mongoOk -join ' ')
if ($mongoOk -notmatch "'ok':\s*1") { Log "ABORT: Mongo 预检失败 -> v17 会退化成 v16 复测，拒跑。详情: $mongoOk"; exit 1 }
Log "Mongo 预检 OK: $mongoOk"

Log "v17 n=100 起跑 (TRIAD=1 CRITIQUE=1 RACE=2 RETRIEVE=1 TRACE_VERSION=17, 全 100 题重算)"
$env:SLC_TRACE_VERSION    = '17'
$env:SLC_TRIAD_VOTE       = '1'
$env:SLC_TRIAD_CRITIQUE   = '1'
$env:SLC_PRIOR_RACE       = '2'
$env:SLC_PRIOR_RETRIEVE   = '1'
$env:SLC_PRIOR_TIMEOUT    = '240'
$env:SLC_JUDGE_TIMEOUT    = '240'
$env:SLC_CRITIQUE_TIMEOUT = '180'
& $py -m knowledge_agents.compare_eval --n 100 --seed 5 *>&1 | Add-Content "$T\v17n100_run.log" -Encoding UTF8
Log "compare_eval 完成 exit=$LASTEXITCODE"

& $py "$PSScriptRoot\_final_v17.py" *>&1 | Add-Content "$T\v17n100_final.log" -Encoding UTF8
Log "V17 CHAIN DONE"
