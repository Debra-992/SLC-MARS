# 全量 489 双探针防记忆标注链（空闲独占版）：等基线定版+端点空 -> 独占循环标注 -> 分层分析。
# 教训：与基线抢同一把 api key 时 probe2 大量 60s 超时（PASS1 里 345/489 超时），B 被基础设施噪声污染。
#       故必须等 key 空闲再跑；独占时放宽超时到 150s、并发降到 4，循环补跑直到 None 归零。
# 输出独立文件，绝不覆盖 pipeline 在用的 dataset_b_qids.json / antimemorization_labels.json。
$ErrorActionPreference = 'Continue'
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$T = Join-Path $ROOT "outputs\traces"
$py = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOG = Join-Path $T "label_all489.log"
function Log($m) { "[$((Get-Date).ToString('MM-dd HH:mm:ss'))] $m" | Add-Content -Path $LOG -Encoding UTF8 }

$env:PYTHONUTF8 = '1'; $env:GIT_PYTHON_REFRESH = 'quiet'
$env:SLC_LABEL_MANIFEST = 'manifest_all489.json'
$env:SLC_LABEL_OUT      = 'antimemorization_labels_all489.json'
$env:SLC_BQIDS_OUT      = 'dataset_b_qids_all489.json'
$env:SLC_PROBE2_CONC    = '4'
$env:SLC_PROBE2_TIMEOUT = '150'

# 1) 排队：只等"除慢 mdagents 外"无重活连续两次（端点已大半空闲，mdagents 低并发可与标注共存）
$MAXWAIT_H = 6; $w0 = Get-Date; $idle = 0
while ($true) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object {
        $_.CommandLine -match 'compare_eval|run_pure_llm|run_lit_rag|run_autogen|run_metagpt|label_antimemorization' })
    if ($busy.Count -eq 0) { $idle++ } else { $idle = 0 }
    if ($idle -ge 2) { break }
    if (((Get-Date) - $w0).TotalHours -gt $MAXWAIT_H) { Log "ABORT: 等待空闲超时 ${MAXWAIT_H}h（重活不含mdagents=$($busy.Count)）"; exit 1 }
    Log "排队中：重活(不含mdagents)=$($busy.Count) idle=$idle，60 秒后复探"; Start-Sleep -Seconds 60
}
Log "端点基本空闲（仅剩/不剩 mdagents），开始独占标注（CONC=4 TIMEOUT=150）"

# 2) 循环补跑：每趟靠 resume 只重探仍 None 的行，直到 None<=5 或最多 5 趟
for ($p = 1; $p -le 5; $p++) {
    Log "标注第 $p 趟起"
    & $py "$PSScriptRoot\label_antimemorization.py" *>&1 | Add-Content "$T\label_all489_run.log" -Encoding UTF8
    $rem = & $py -c "import json;d=json.load(open('outputs/traces/antimemorization_labels_all489.json',encoding='utf-8'));print(sum(1 for l in d['labels'] if l.get('probe2',{}).get('correct') is None and l.get('class')!='missing_input'))" 2>$null
    Log "第 $p 趟完成，仍 None=$rem"
    if ($rem -ne $null -and [int]$rem -le 5) { Log "超时基本清零，停止补跑"; break }
    Start-Sleep -Seconds 20
}

# 3) 分层分析（读干净 B + 各基线当前答案）
Log "分析器 baselines_by_dataset 起"
& $py "$PSScriptRoot\baselines_by_dataset.py" *>&1 | Add-Content "$T\baselines_by_dataset_run.log" -Encoding UTF8
Log "IDLE-LABEL+BENCH CHAIN DONE"
