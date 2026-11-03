# 温度0 R2 开关 A/B 判决链：等 baseline 端点跑空 -> armA(R2=off) -> armB(R2=on) -> 按 A(全部)/B(防记忆) 分析。
# 两臂同冠军基座(slc_mars_env)+ 同 SLC_FORCE_TEMP0=1，唯一变量 = SLC_MEM_R2。版本 a0/b0 分开存，互不覆盖，也不动 v16/v20 缓存。
# 注：Stop 会让起跑瞬间撞 v500/Mongo 预检的非终止错误直接静默退出（上次崩因），改 Continue + 预检前先注入 slc_mars_env。
$ErrorActionPreference = 'Continue'
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$T = Join-Path $ROOT "outputs\traces"
$py = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOG = Join-Path $T "r2_t0_ab_chain.log"
function Log($m) { "[$((Get-Date).ToString('MM-dd HH:mm:ss'))] $m" | Add-Content -Path $LOG -Encoding UTF8 }

# 1) 排队：等所有重活(compare_eval/五基线/label)离场，连续两次探测都空闲才起跑（=端点跑空）
$MAXWAIT_H = 30
$w0 = Get-Date; $idle = 0
while ($true) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object {
        $_.CommandLine -match 'compare_eval|run_pure_llm|run_lit_rag|run_autogen|run_mdagents|run_metagpt|label_antimemorization' })
    if ($busy.Count -eq 0) { $idle++ } else { $idle = 0 }
    if ($idle -ge 2) { break }
    if (((Get-Date) - $w0).TotalHours -gt $MAXWAIT_H) { Log "ABORT: 等待端点空闲超时 ${MAXWAIT_H}h"; exit 1 }
    Log "排队中：检出 $($busy.Count) 个重活进程，2 分钟后复探"; Start-Sleep -Seconds 120
}
Log "端点已空闲，起跑温度0 R2 A/B"

# 2) 冠军基座先注入（含 MONGO_URI），再做内网预检（避免未设环境变量→KeyError 撞 Stop）
. (Join-Path $ROOT "tools\slc_mars_env.ps1")
$mongoOk = & $py -c "import os;from pymongo import MongoClient;c=MongoClient(os.environ['MONGO_URI'],serverSelectionTimeoutMS=8000);print(c.admin.command('ping'))" 2>&1
Log "Mongo 预检: $(($mongoOk -join ' '))"

# 3) 全局温度0（基座已在上面注入）
$env:SLC_FORCE_TEMP0    = '1'
$env:SLC_TRIAD_VOTE     = '1'
$env:SLC_TRIAD_CRITIQUE = '1'
$env:SLC_PRIOR_RACE     = '2'
$env:SLC_PRIOR_TIMEOUT  = '240'
$env:SLC_JUDGE_TIMEOUT  = '240'
$env:SLC_CRITIQUE_TIMEOUT = '180'

# 4) armA：R2 OFF（版本 a0）
Log "armA R2=off TRACE_VERSION=a0 起（全 100 题 temp0 重算）"
$env:SLC_MEM_R2 = '0'; $env:SLC_TRACE_VERSION = 'a0'
& $py -m knowledge_agents.compare_eval --n 100 --seed 5 *>&1 | Add-Content "$T\r2t0_armA.log" -Encoding UTF8
Log "armA 完成 exit=$LASTEXITCODE"

# 5) armB：R2 ON（版本 b0）
Log "armB R2=on TRACE_VERSION=b0 起（全 100 题 temp0 重算）"
$env:SLC_MEM_R2 = '1'; $env:SLC_TRACE_VERSION = 'b0'
& $py -m knowledge_agents.compare_eval --n 100 --seed 5 *>&1 | Add-Content "$T\r2t0_armB.log" -Encoding UTF8
Log "armB 完成 exit=$LASTEXITCODE"

# 6) 分析：按 A(全部)/B(防记忆) 分列 + membership 配对 fix/break
& $py "$PSScriptRoot\_final_r2_t0.py" *>&1 | Tee-Object -FilePath "$T\r2_t0_ab_report.txt" | Add-Content $LOG -Encoding UTF8
Log "R2 温度0 A/B 链 DONE"
