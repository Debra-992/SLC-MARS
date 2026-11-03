# 恢复续跑（判决优先序）：mdagents 补 19 题 -> merge(v12×基线终表) -> v16@n=100 -> 三方终表
# -> 清扫 pure_llm/lit_rag/autogen error 行 + 重建 B -> 重出两张终表。
# 全程串行（v15 教训：绝不并发抢端点）；每步断点续跑，日志追加到既有 expansion_*.log。
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\slc_mars_env.ps1"
$T     = "$PSScriptRoot\..\outputs\traces"
$py    = "$PSScriptRoot\..\.venv\Scripts\python.exe"
$agpy  = "$PSScriptRoot\..\outputs\autogen-venv\Scripts\python.exe"
$mdpy  = "$PSScriptRoot\..\outputs\mdagents-venv\Scripts\python.exe"
$chain = "$T\expansion_chain.log"
function Log($m) { "[$(Get-Date -Format 'MM-dd HH:mm:ss')] $m" | Add-Content $chain -Encoding utf8 }

Log "RESUME(判决优先序): run_mdagents 补断点(剩 17+2error)"
& $mdpy "$PSScriptRoot\..\knowledge_agents\baselines\run_mdagents.py" *>&1 | Add-Content "$T\expansion_mdagents.log" -Encoding utf8

Log "merge_baselines -> v12 x 基线 n=100 终表(第一版)"
& $py -m knowledge_agents.baselines.merge_baselines *>&1 | Add-Content "$T\expansion_merge.log" -Encoding utf8

Log "v16 n=100 (TRIAD=1 CRITIQUE=1 RACE=2 TRACE_VERSION=16, 全 100 题重算)"
$env:SLC_TRACE_VERSION   = '16'
$env:SLC_TRIAD_VOTE      = '1'
$env:SLC_TRIAD_CRITIQUE  = '1'
$env:SLC_PRIOR_RACE      = '2'
$env:SLC_PRIOR_TIMEOUT   = '240'
$env:SLC_JUDGE_TIMEOUT   = '240'
$env:SLC_CRITIQUE_TIMEOUT = '180'
& $py -m knowledge_agents.compare_eval --n 100 --seed 5 *>&1 | Add-Content "$T\v16n100_run.log" -Encoding utf8

Log "三方终表 _final_n100 (v16 vs v12 vs mdagents)"
& $py "$PSScriptRoot\_final_n100.py" *>&1 | Add-Content "$T\v16n100_final.log" -Encoding utf8
$env:SLC_TRACE_VERSION = '12'   # 还原冠军默认，防后续步骤误用 16

Log "清扫基线 error 行: pure_llm -> lit_rag -> label -> autogen"
& $py -m knowledge_agents.baselines.run_pure_llm *>&1 | Add-Content "$T\expansion_purellm.log" -Encoding utf8
& $py -m knowledge_agents.baselines.run_lit_rag  *>&1 | Add-Content "$T\expansion_litrag.log"  -Encoding utf8
$env:PYTHONPATH = "$PSScriptRoot\.."
& $py "$PSScriptRoot\label_antimemorization.py"  *>&1 | Add-Content "$T\expansion_label.log"   -Encoding utf8
& $agpy "$PSScriptRoot\..\knowledge_agents\baselines\run_autogen.py" *>&1 | Add-Content "$T\expansion_autogen.log" -Encoding utf8

Log "重出终表(基线补齐版): merge + _final_n100"
& $py -m knowledge_agents.baselines.merge_baselines *>&1 | Add-Content "$T\expansion_merge.log" -Encoding utf8
& $py "$PSScriptRoot\_final_n100.py" *>&1 | Add-Content "$T\v16n100_final.log" -Encoding utf8
Log "RESUME DONE"
