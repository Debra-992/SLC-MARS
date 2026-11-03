# v18 判决臂：membership 负证据通道（SLC_NEG_EVID=1）@ n=100，串行独占 + 内网预检。
# 选择性弃权 THETA=0（本臂只判负证据；θ 工作点由 _risk_cover.py 在 conf 分布上离线模拟）。
$ErrorActionPreference = 'Stop'
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$T = Join-Path $ROOT "outputs\traces"
$LOG = Join-Path $T "v18_chain.log"
function Log($m) { "[$((Get-Date).ToString('HH:mm:ss'))] $m" | Add-Content -Path $LOG -Encoding UTF8 }
$py = Join-Path $ROOT ".venv\Scripts\python.exe"

# 1) 串行互斥
$busy = Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object {
    $_.CommandLine -match 'compare_eval|run_pure_llm|run_lit_rag|run_autogen|run_mdagents|label_antimemorization' }
if ($busy) { Log "ABORT: 检出 $($busy.Count) 个在跑重活进程，串行保护"; exit 1 }

. (Join-Path $ROOT "tools\slc_mars_env.ps1")

# 2) 内网预检（负证据通道依赖 papers 库；断内网时通道静默退化 = 白跑一场 v16 复测）
$mongoOk = & $py -c "import os;from pymongo import MongoClient;c=MongoClient(os.environ['MONGO_URI'],serverSelectionTimeoutMS=8000);print(c.admin.command('ping'));print(c[os.environ.get('MONGO_DB_NAME','SLCdb')].papers.estimated_document_count())" 2>&1
$mongoOk = ($mongoOk -join ' ')
if ($mongoOk -notmatch "'ok':\s*1") { Log "ABORT: Mongo 预检失败，拒跑。详情: $mongoOk"; exit 1 }
Log "Mongo 预检 OK: $mongoOk"

Log "v18 n=100 起跑 (TRIAD=1 CRITIQUE=1 RACE=2 NEG_EVID=1 THETA=0 TRACE_VERSION=18)"
$env:SLC_TRACE_VERSION    = '18'
$env:SLC_TRIAD_VOTE       = '1'
$env:SLC_TRIAD_CRITIQUE   = '1'
$env:SLC_PRIOR_RACE       = '2'
$env:SLC_NEG_EVID         = '1'
$env:SLC_ABSTAIN_THETA    = '0'
$env:SLC_PRIOR_TIMEOUT    = '240'
$env:SLC_JUDGE_TIMEOUT    = '240'
$env:SLC_CRITIQUE_TIMEOUT = '180'
& $py -m knowledge_agents.compare_eval --n 100 --seed 5 *>&1 | Add-Content "$T\v18n100_run.log" -Encoding UTF8
Log "compare_eval 完成 exit=$LASTEXITCODE"

& $py "$PSScriptRoot\_final_v18.py" *>&1 | Add-Content "$T\v18n100_final.log" -Encoding UTF8
& $py "$PSScriptRoot\_risk_cover.py" 16 18 *>&1 | Add-Content "$T\v18n100_final.log" -Encoding UTF8
Log "V18 CHAIN DONE"
