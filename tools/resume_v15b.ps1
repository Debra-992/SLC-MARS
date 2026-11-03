# Resume v15 triad run after network pause (16:00). Reuses the 11 clean cached v15
# rows and recomputes only the remaining questions (no --no-cache on purpose).
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)   # avoid hardcoded CJK path (encoding-fragile)
. "$PSScriptRoot\slc_mars_env.ps1"
$env:SLC_TRIAD_VOTE   = '1'
$env:SLC_PRIOR_TIMEOUT = '240'
$env:SLC_JUDGE_TIMEOUT = '240'
.\.venv\Scripts\python.exe -c "import os; print('CHECK triad=',os.environ.get('SLC_TRIAD_VOTE'),'pt=',os.environ.get('SLC_PRIOR_TIMEOUT'),'jt=',os.environ.get('SLC_JUDGE_TIMEOUT'),'chief=',os.environ.get('SLC_LLM_CHIEF'),'prior=',os.environ.get('SLC_PRIOR_AGENT'),'mongo=',len(os.environ.get('MONGO_URI',''))>10)"
Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList '-m','knowledge_agents.compare_eval','--n','40','--seed','5' -RedirectStandardOutput 'outputs\traces\v15c_run.log' -RedirectStandardError 'outputs\traces\v15c_run.err.log' -WindowStyle Hidden
Write-Host 'v15 resume launched -> outputs\traces\v15c_run.log'
