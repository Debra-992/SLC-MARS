# 全量干净题（489 = clean 500 排除 11 道 topk）并发跑 4 个独立 baseline。
# 与正在跑的 v20(compare_eval) 并发：用户确认端点余量充足。
#   覆盖：pure_llm / lit_rag（主 .venv）、autogen（autogen-venv）、mdagents（mdagents-venv）。
#   不含 metaGPT-native(管线内特化) —— 它由 compare_eval 写入 tracestore，属 pipeline 步。
#   但 run_metagpt(原生 stock MetaGPT Role/Action 纯 LLM)已作独立 5 号 runner，走同一 489 题集。
# 断点续跑：各 runner 用 answers_<system>.jsonl 按 qid 去重，只补未成功题；4 个系统各写各文件，无写入竞争。
param(
    # mdagents 辩论限流：留空 = 官方默认(5/5/3/3/5)，忠实可对比；传值 = 降档省钱提速
    [int]$MdaRounds = 0, [int]$MdaTurns = 0, [int]$MdaTeams = 0,
    [int]$MdaTeamMembers = 0, [int]$MdaAgents = 0,
    [int]$LlmTimeout = 0,          # >0 时设 SLC_BASELINE_LLM_TIMEOUT（洗 pure_llm/lit_rag 的 60s 瞬时超时）
    [switch]$Kill,                 # 先杀掉在跑的旧 baseline 进程再起
    [switch]$Fresh,                # 清空 answers_*.jsonl 从头重跑（获取全量 token 统计；会丢弃已算答案）
    [string[]]$Systems = @('pure_llm','lit_rag','autogen','mdagents')   # 只（重）启选中系统；用于单独重跑 autogen/mdagents 而不扰动其余
)
$ErrorActionPreference = 'Stop'
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$T = Join-Path $ROOT "outputs\traces"
$LOG = Join-Path $T "baselines_all500_chain.log"
function Log($m) { "[$((Get-Date).ToString('HH:mm:ss'))] $m" | Add-Content -Path $LOG -Encoding UTF8 }

# Windows 中文路径 + rpy2/子进程 UTF-8 一致性；GitPython 懒导入非致命（MetaGPT 侧）
$env:PYTHONUTF8 = '1'
$env:GIT_PYTHON_REFRESH = 'quiet'
# baseline 端点自源 config2.yaml（common.llm_settings），无需路由环境；仅 lit_rag 需 Mongo。

# 从 run_route_b.ps1 提取 Mongo 连接串（避免密钥在本文件重复落地）
$rb = Get-Content (Join-Path $ROOT "run_route_b.ps1") -Raw
if ($rb -match '(?m)^\s*\$env:MONGO_URI\s*=\s*"([^"]+)"') { $env:MONGO_URI = $Matches[1] } else { throw "run_route_b.ps1 未找到 MONGO_URI" }
if ($rb -match '(?m)^\s*\$env:MONGO_DB_NAME\s*=\s*"([^"]+)"') { $env:MONGO_DB_NAME = $Matches[1] } else { $env:MONGO_DB_NAME = 'SLCdb' }

$py  = Join-Path $ROOT ".venv\Scripts\python.exe"
$apy = Join-Path $ROOT "outputs\autogen-venv\Scripts\python.exe"
$mpy = Join-Path $ROOT "outputs\mdagents-venv\Scripts\python.exe"
foreach ($p in @($py, $apy, $mpy)) { if (-not (Test-Path $p)) { throw "缺解释器: $p" } }
if (-not (Test-Path (Join-Path $ROOT "outputs\MDAgents"))) { throw "缺 MDAgents 仓库克隆" }

# 系统 -> (解释器, 参数) 定义表；$Systems 选中的才 kill/fresh/launch
$DEF = [ordered]@{
    'pure_llm' = @{ exe = $py;  args = @('-m','knowledge_agents.baselines.run_pure_llm','--conc','4') }
    'lit_rag'  = @{ exe = $py;  args = @('-m','knowledge_agents.baselines.run_lit_rag','--conc','4') }
    'autogen'  = @{ exe = $apy; args = @('knowledge_agents\baselines\run_autogen.py') }
    'mdagents' = @{ exe = $mpy; args = @('knowledge_agents\baselines\run_mdagents.py') }
    'metagpt'  = @{ exe = $py;  args = @('-m','knowledge_agents.baselines.run_metagpt') }
}
foreach ($s in $Systems) { if (-not $DEF.Contains($s)) { throw "未知系统: $s（可选 pure_llm/lit_rag/autogen/mdagents/metagpt）" } }

function Start-Bl($name, $exe, $argList) {
    $out = Join-Path $T "bl_$name.log"
    $err = Join-Path $T "bl_$name.err"
    Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory $ROOT `
        -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err
    Log "START $name -> bl_$name.log"
}

Log "=== baselines 全量 489 并发起跑 (pure_llm/lit_rag/autogen/mdagents) ==="

# 可选杀旧：仅杀 4 个 baseline runner，绝不碰 compare_eval(v20)
if ($Kill) {
    $old = Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object {
        $_.CommandLine -match (($Systems | ForEach-Object { 'run_' + $_ }) -join '|') }
    foreach ($o in $old) { Stop-Process -Id $o.ProcessId -Force -ErrorAction SilentlyContinue; Log "KILL pid=$($o.ProcessId)" }
}
# 可选清空：从头重跑以获取全量 tokens（会丢弃已算好的答案行，慎用）
if ($Fresh) {
    foreach ($s in $Systems) {
        $f = Join-Path $T "answers_$s.jsonl"
        if (Test-Path $f) { Remove-Item $f -Force; Log "FRESH wiped answers_$s.jsonl" }
    }
}

# mdagents 辩论限流 env（0/未设 = 保持官方默认）
if ($MdaRounds      -gt 0) { $env:SLC_MDA_ROUNDS       = "$MdaRounds" }
if ($MdaTurns       -gt 0) { $env:SLC_MDA_TURNS        = "$MdaTurns" }
if ($MdaTeams       -gt 0) { $env:SLC_MDA_TEAMS        = "$MdaTeams" }
if ($MdaTeamMembers -gt 0) { $env:SLC_MDA_TEAM_MEMBERS = "$MdaTeamMembers" }
if ($MdaAgents      -gt 0) { $env:SLC_MDA_AGENTS       = "$MdaAgents" }
if ($LlmTimeout     -gt 0) { $env:SLC_BASELINE_LLM_TIMEOUT = "$LlmTimeout" }
Log ("mda 限流: rounds={0} turns={1} teams={2} members={3} agents={4} | llm_timeout={5}" -f `
    ($(if($MdaRounds){$MdaRounds}else{'5*'})), ($(if($MdaTurns){$MdaTurns}else{'5*'})), `
    ($(if($MdaTeams){$MdaTeams}else{'3*'})), ($(if($MdaTeamMembers){$MdaTeamMembers}else{'3*'})), `
    ($(if($MdaAgents){$MdaAgents}else{'5*'})), ($(if($LlmTimeout){$LlmTimeout}else{'default'})))

foreach ($s in $Systems) { $d = $DEF[$s]; Start-Bl $s $d.exe $d.args }
Log ("已投递 {0} 个后台进程: {1}；进度看 answers_<system>.jsonl 行数增长与 bl_*.log" -f $Systems.Count, ($Systems -join ','))
