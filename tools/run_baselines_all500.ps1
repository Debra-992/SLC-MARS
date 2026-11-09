# Every clean question (489 = the 500-question clean set minus 11 top-k items) through the four
# independent baselines, launched concurrently.  This script is meant to run side by side with the
# pipeline evaluation (compare_eval), which is a separate process.
#   Covered: pure_llm / lit_rag (main .venv), autogen (autogen-venv), mdagents (mdagents-venv).
#   metaGPT-native is not one of them: it is specialised inside the pipeline and compare_eval
#   writes it into the trace store, so it belongs to the pipeline step.  run_metagpt (a stock
#   MetaGPT Role/Action, pure LLM) does run as a fifth runner over the same 489-question set.
# Resume-safe: each runner de-duplicates by qid inside its own answers_<system>.jsonl and only
# retries the questions that have not succeeded yet; the systems write separate files, so there is
# no write contention.
param(
    # mdagents debate rate limits: empty = the official defaults (5/5/3/3/5), which stay faithful
    # and comparable; passing values dials them down to save cost and wall-clock time
    [int]$MdaRounds = 0, [int]$MdaTurns = 0, [int]$MdaTeams = 0,
    [int]$MdaTeamMembers = 0, [int]$MdaAgents = 0,
    [int]$LlmTimeout = 0,          # >0: sets SLC_BASELINE_LLM_TIMEOUT (covers the transient 60 s timeouts of pure_llm/lit_rag)
    [switch]$Kill,                 # kill the baseline processes still running before starting
    [switch]$Fresh,                # wipe answers_*.jsonl and re-run from scratch (collects full token statistics; discards the answers already computed)
    [string[]]$Systems = @('pure_llm','lit_rag','autogen','mdagents')   # (re)start only the selected systems; used to re-run autogen/mdagents alone without touching the rest
)
$ErrorActionPreference = 'Stop'
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$T = Join-Path $ROOT "outputs\traces"
$LOG = Join-Path $T "baselines_all500_chain.log"
function Log($m) { "[$((Get-Date).ToString('HH:mm:ss'))] $m" | Add-Content -Path $LOG -Encoding UTF8 }

# UTF-8 consistency for Windows paths containing non-ASCII characters and for rpy2/subprocesses;
# GitPython is imported lazily by MetaGPT and its failure must not be fatal here.
$env:PYTHONUTF8 = '1'
$env:GIT_PYTHON_REFRESH = 'quiet'
# The baselines resolve the LLM endpoint themselves (common.llm_settings: SLC_LLM_* first, then
# config2.yaml), so no full launcher environment is needed here; only lit_rag additionally needs
# MongoDB.  Credentials are never stored in this file: a value already exported into the environment
# wins, otherwise the untracked `secrets.local.ps1` at the repository root is read through the same
# loader the pipeline launcher uses.  (This replaced an earlier step that scraped MONGO_URI out of
# run_route_b.ps1; that file no longer carries a literal connection string, so the scrape threw.)
. (Join-Path $PSScriptRoot "load_secrets.ps1")
Import-SlcSecrets -Root $ROOT -Quiet | Out-Null
if (-not $env:MONGO_DB_NAME) { $env:MONGO_DB_NAME = 'SLCdb' }
if (-not $env:MONGO_URI) {
    Write-Warning 'MONGO_URI is not configured: skipping the lit_rag baseline (it can only record errors without a literature database). The other runners are unaffected.'
    $Systems = @($Systems | Where-Object { $_ -ne 'lit_rag' })
}

$py  = Join-Path $ROOT ".venv\Scripts\python.exe"
$apy = Join-Path $ROOT "outputs\autogen-venv\Scripts\python.exe"
$mpy = Join-Path $ROOT "outputs\mdagents-venv\Scripts\python.exe"
foreach ($p in @($py, $apy, $mpy)) { if (-not (Test-Path $p)) { throw "missing interpreter: $p" } }
if (-not (Test-Path (Join-Path $ROOT "outputs\MDAgents"))) { throw "the MDAgents repository clone is missing" }

# system -> (interpreter, arguments); only the systems selected in $Systems are killed / wiped / launched
$DEF = [ordered]@{
    'pure_llm' = @{ exe = $py;  args = @('-m','knowledge_agents.baselines.run_pure_llm','--conc','4') }
    'lit_rag'  = @{ exe = $py;  args = @('-m','knowledge_agents.baselines.run_lit_rag','--conc','4') }
    'autogen'  = @{ exe = $apy; args = @('knowledge_agents\baselines\run_autogen.py') }
    'mdagents' = @{ exe = $mpy; args = @('knowledge_agents\baselines\run_mdagents.py') }
    'metagpt'  = @{ exe = $py;  args = @('-m','knowledge_agents.baselines.run_metagpt') }
}
foreach ($s in $Systems) { if (-not $DEF.Contains($s)) { throw "unknown system: $s (choose from pure_llm/lit_rag/autogen/mdagents/metagpt)" } }

function Start-Bl($name, $exe, $argList) {
    $out = Join-Path $T "bl_$name.log"
    $err = Join-Path $T "bl_$name.err"
    Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory $ROOT `
        -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err
    Log "START $name -> bl_$name.log"
}

Log "=== baselines: full 489-question run started in parallel (pure_llm/lit_rag/autogen/mdagents) ==="

# Optional kill of stale processes: the four baseline runners only, never the pipeline evaluation.
if ($Kill) {
    $old = Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object {
        $_.CommandLine -match (($Systems | ForEach-Object { 'run_' + $_ }) -join '|') }
    foreach ($o in $old) { Stop-Process -Id $o.ProcessId -Force -ErrorAction SilentlyContinue; Log "KILL pid=$($o.ProcessId)" }
}
# Optional wipe: re-run from scratch to collect full token counts (this discards the answer rows
# already computed, so use it with care)
if ($Fresh) {
    foreach ($s in $Systems) {
        $f = Join-Path $T "answers_$s.jsonl"
        if (Test-Path $f) { Remove-Item $f -Force; Log "FRESH wiped answers_$s.jsonl" }
    }
}

# mdagents debate rate limits (0 / unset = keep the official defaults)
if ($MdaRounds      -gt 0) { $env:SLC_MDA_ROUNDS       = "$MdaRounds" }
if ($MdaTurns       -gt 0) { $env:SLC_MDA_TURNS        = "$MdaTurns" }
if ($MdaTeams       -gt 0) { $env:SLC_MDA_TEAMS        = "$MdaTeams" }
if ($MdaTeamMembers -gt 0) { $env:SLC_MDA_TEAM_MEMBERS = "$MdaTeamMembers" }
if ($MdaAgents      -gt 0) { $env:SLC_MDA_AGENTS       = "$MdaAgents" }
if ($LlmTimeout     -gt 0) { $env:SLC_BASELINE_LLM_TIMEOUT = "$LlmTimeout" }
Log ("mda rate limits: rounds={0} turns={1} teams={2} members={3} agents={4} | llm_timeout={5}" -f `
    ($(if($MdaRounds){$MdaRounds}else{'5*'})), ($(if($MdaTurns){$MdaTurns}else{'5*'})), `
    ($(if($MdaTeams){$MdaTeams}else{'3*'})), ($(if($MdaTeamMembers){$MdaTeamMembers}else{'3*'})), `
    ($(if($MdaAgents){$MdaAgents}else{'5*'})), ($(if($LlmTimeout){$LlmTimeout}else{'default'})))

foreach ($s in $Systems) { $d = $DEF[$s]; Start-Bl $s $d.exe $d.args }
Log ("launched {0} background processes: {1}; watch progress in the growth of the answers_<system>.jsonl line counts and in bl_*.log" -f $Systems.Count, ($Systems -join ','))
