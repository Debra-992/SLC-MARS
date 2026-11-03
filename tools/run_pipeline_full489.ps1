# 管线 v16 / v20 全 489 判决链：等 R2 温度0 A/B 报告落地且端点空 -> v16(R2off) -> v20(R2on) -> 分层分析。
# 排在所有工作最后（label -> R2 A/B -> 本链）。两臂 --n 500 取满 489（与基线同集合），版本 16f489/20f489 分开存不覆盖旧 100-A(v16/v20) 缓存。
# 默认温度（生产口径，非 temp0），只 SLC_MEM_R2 一个变量差异 v16↔v20。
$ErrorActionPreference = 'Continue'
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$T = Join-Path $ROOT "outputs\traces"
$py = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOG = Join-Path $T "pipeline_full489_chain.log"
function Log($m) { "[$((Get-Date).ToString('MM-dd HH:mm:ss'))] $m" | Add-Content -Path $LOG -Encoding UTF8 }

# 1) 排队：先等 R2 A/B 报告出现（=上游全清），再等无重活连续两次
$MAXWAIT_H = 72; $w0 = Get-Date; $idle = 0
while ($true) {
    $r2done = Test-Path (Join-Path $T "r2_t0_ab_report.txt")
    $busy = @(Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object {
        $_.CommandLine -match 'compare_eval|run_pure_llm|run_lit_rag|run_autogen|run_mdagents|run_metagpt|label_antimemorization|baselines_by_dataset' })
    if ($r2done -and $busy.Count -eq 0) { $idle++ } else { $idle = 0 }
    if ($idle -ge 2) { break }
    if (((Get-Date) - $w0).TotalHours -gt $MAXWAIT_H) { Log "ABORT: 等待超时 ${MAXWAIT_H}h（r2done=$r2done 重活=$($busy.Count)）"; exit 1 }
    Log "排队中：r2done=$r2done 重活=$($busy.Count) idle=$idle，3 分钟后复探"; Start-Sleep -Seconds 180
}
Log "上游已清，起跑管线全 489"

# 2) 备份现有 100-A 清单（compare_eval 每次运行会覆盖 sample_manifest.json；保留旧定义供 _final_r2_t0/_final_v20）
if (-not (Test-Path (Join-Path $T "sample_manifest_100A.json")) -and (Test-Path (Join-Path $T "sample_manifest.json"))) {
    Copy-Item (Join-Path $T "sample_manifest.json") (Join-Path $T "sample_manifest_100A.json")
    Log "已备份 100-A 清单 -> sample_manifest_100A.json"
}

# 3) Mongo 预检（Kira 证据链依赖）
$mongoOk = & $py -c "import os;from pymongo import MongoClient;c=MongoClient(os.environ['MONGO_URI'],serverSelectionTimeoutMS=8000);print(c.admin.command('ping'))" 2>&1
if (($mongoOk -join ' ') -notmatch "MONGO_URI") {
    . (Join-Path $ROOT "tools\slc_mars_env.ps1")
    $mongoOk = & $py -c "import os;from pymongo import MongoClient;c=MongoClient(os.environ['MONGO_URI'],serverSelectionTimeoutMS=8000);print(c.admin.command('ping'))" 2>&1
}
Log "Mongo 预检: $(($mongoOk -join ' '))"

# 4) 冠军基座（slc_mars_env 已含 TRIAD/CRITIQUE/RACE；此处只切 R2 开关与版本）
. (Join-Path $ROOT "tools\slc_mars_env.ps1")
$env:SLC_PRIOR_TIMEOUT  = '240'; $env:SLC_JUDGE_TIMEOUT = '240'; $env:SLC_CRITIQUE_TIMEOUT = '180'

# 5) arm v16：R2 OFF（版本 16f489），全 489
Log "arm v16 R2=off TRACE_VERSION=16f489 起（全 489 题）"
$env:SLC_MEM_R2 = '0'; $env:SLC_TRACE_VERSION = '16f489'
& $py -m knowledge_agents.compare_eval --n 500 --seed 5 *>&1 | Add-Content "$T\pl16_full489.log" -Encoding UTF8
Log "arm v16 完成 exit=$LASTEXITCODE"

# 6) arm v20：R2 ON（版本 20f489），全 489
Log "arm v20 R2=on TRACE_VERSION=20f489 起（全 489 题）"
$env:SLC_MEM_R2 = '1'; $env:SLC_TRACE_VERSION = '20f489'
& $py -m knowledge_agents.compare_eval --n 500 --seed 5 *>&1 | Add-Content "$T\pl20_full489.log" -Encoding UTF8
Log "arm v20 完成 exit=$LASTEXITCODE"

# 7) 分析：A(全部489)/B(防记忆489) × mem/pw × heuristic/auto + v16↔v20 配对
& $py "$PSScriptRoot\_final_pipeline_full489.py" *>&1 | Tee-Object -FilePath "$T\pipeline_full489_report.txt" | Add-Content $LOG -Encoding UTF8
Log "PIPELINE FULL-489 CHAIN DONE"
