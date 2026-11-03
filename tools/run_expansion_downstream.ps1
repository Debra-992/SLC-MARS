# 扩测下游链：等 v12 n=100 管线进程退出后，按依赖顺序补齐基线并重建 B 集、出合并表。
# 各步增量可断点续跑（answers_*.jsonl 按 qid 跳过成功行；label 按 template_version 复用）。
# 用法：powershell -File tools\run_expansion_downstream.ps1（由 Bash 以分离进程拉起）
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\slc_mars_env.ps1"
$T = "$PSScriptRoot\..\outputs\traces"
$mainpy = "$PSScriptRoot\..\.venv\Scripts\python.exe"
$agpy   = "$PSScriptRoot\..\outputs\autogen-venv\Scripts\python.exe"
$mdpy   = "$PSScriptRoot\..\outputs\mdagents-venv\Scripts\python.exe"

"[$(Get-Date -Format 'HH:mm:ss')] waiting for compare_eval (v12 n=100) to finish..." |
    Out-File "$T\expansion_chain.log" -Encoding utf8
while ((Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'compare_eval' }).Count -gt 0) {
    Start-Sleep -Seconds 60
}
"[$(Get-Date -Format 'HH:mm:ss')] pipeline done; export_inputs" | Add-Content "$T\expansion_chain.log" -Encoding utf8
& $mainpy -m knowledge_agents.baselines.export_inputs  *>&1 | Add-Content "$T\expansion_export.log" -Encoding utf8

"[$(Get-Date -Format 'HH:mm:ss')] run_pure_llm (probe1 + 表)" | Add-Content "$T\expansion_chain.log" -Encoding utf8
& $mainpy -m knowledge_agents.baselines.run_pure_llm   *>&1 | Add-Content "$T\expansion_purellm.log" -Encoding utf8

"[$(Get-Date -Format 'HH:mm:ss')] run_lit_rag" | Add-Content "$T\expansion_chain.log" -Encoding utf8
& $mainpy -m knowledge_agents.baselines.run_lit_rag    *>&1 | Add-Content "$T\expansion_litrag.log" -Encoding utf8

"[$(Get-Date -Format 'HH:mm:ss')] label_antimemorization (重建 B 集 on 100 题)" | Add-Content "$T\expansion_chain.log" -Encoding utf8
$env:PYTHONPATH = "$PSScriptRoot\.."
& $mainpy "$PSScriptRoot\label_antimemorization.py"    *>&1 | Add-Content "$T\expansion_label.log" -Encoding utf8

"[$(Get-Date -Format 'HH:mm:ss')] run_autogen" | Add-Content "$T\expansion_chain.log" -Encoding utf8
& $agpy "$PSScriptRoot\..\knowledge_agents\baselines\run_autogen.py"  *>&1 | Add-Content "$T\expansion_autogen.log" -Encoding utf8

"[$(Get-Date -Format 'HH:mm:ss')] run_mdagents (最重：61 新题 x 多轮协作)" | Add-Content "$T\expansion_chain.log" -Encoding utf8
& $mdpy "$PSScriptRoot\..\knowledge_agents\baselines\run_mdagents.py" *>&1 | Add-Content "$T\expansion_mdagents.log" -Encoding utf8

"[$(Get-Date -Format 'HH:mm:ss')] merge_baselines -> 终表" | Add-Content "$T\expansion_chain.log" -Encoding utf8
& $mainpy -m knowledge_agents.baselines.merge_baselines *>&1 | Add-Content "$T\expansion_merge.log" -Encoding utf8
"[$(Get-Date -Format 'HH:mm:ss')] CHAIN DONE" | Add-Content "$T\expansion_chain.log" -Encoding utf8
