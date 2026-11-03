# v20 扩 500：冠军(v16+R2 兜底)在全 clean 池(pairwise+membership=489) 增量重算，串行独占 + 内网预检。
# TRACE_VERSION 保持 20 => 复用已跑的 100 缓存，只补 ~389 新题。跑完自动出 R2 out-of-sample 迁移判决。
$ErrorActionPreference = 'Stop'
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$T = Join-Path $ROOT "outputs\traces"
$LOG = Join-Path $T "v20n500_chain.log"
function Log($m) { "[$((Get-Date).ToString('HH:mm:ss'))] $m" | Add-Content -Path $LOG -Encoding UTF8 }
$py = Join-Path $ROOT ".venv\Scripts\python.exe"

# 1) 串行互斥
$busy = Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object {
    $_.CommandLine -match 'compare_eval|run_pure_llm|run_lit_rag|run_autogen|run_mdagents|label_antimemorization' }
if ($busy) { Log "ABORT: 检出 $($busy.Count) 个在跑重活进程，串行保护"; exit 1 }

. (Join-Path $ROOT "tools\slc_mars_env.ps1")

# 2) 内网预检（证据链依赖 Mongo；断内网会整体退化 = 白跑）
$mongoOk = & $py -c "import os;from pymongo import MongoClient;c=MongoClient(os.environ['MONGO_URI'],serverSelectionTimeoutMS=8000);print(c.admin.command('ping'));print(c[os.environ.get('MONGO_DB_NAME','SLCdb')].papers.estimated_document_count())" 2>&1
$mongoOk = ($mongoOk -join ' ')
if ($mongoOk -notmatch "'ok':\s*1") { Log "ABORT: Mongo 预检失败，拒跑。详情: $mongoOk"; exit 1 }
Log "Mongo 预检 OK: $mongoOk"

Log "v20 扩500 起跑 (TRIAD=1 CRITIQUE=1 RACE=2 MEM_R2=1 TRACE_VERSION=20 复用100缓存, --n 500)"
$env:SLC_TRACE_VERSION    = '20'
$env:SLC_TRIAD_VOTE       = '1'
$env:SLC_TRIAD_CRITIQUE   = '1'
$env:SLC_PRIOR_RACE       = '2'
$env:SLC_MEM_R2           = '1'
$env:SLC_PRIOR_TIMEOUT    = '240'
$env:SLC_JUDGE_TIMEOUT    = '240'
$env:SLC_CRITIQUE_TIMEOUT = '180'
& $py -m knowledge_agents.compare_eval --n 500 --seed 5 *>&1 | Add-Content "$T\v20n500_run.log" -Encoding UTF8
Log "compare_eval --n 500 完成 exit=$LASTEXITCODE"

& $py "$PSScriptRoot\_r2_oos.py" *>&1 | Out-File "$T\v20n500_oos.log" -Encoding UTF8
Log "V20 500 CHAIN DONE (R2 OOS -> v20n500_oos.log)"
