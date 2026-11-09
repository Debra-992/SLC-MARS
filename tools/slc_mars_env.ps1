# SLC-MARS evaluation environment: the released configuration, pinned here in full.
# Usage: dot-source `. tools\slc_mars_env.ps1` inside the SAME shell block in which you then
#        start the process (Start-Process does not inherit the environment of a later block).
# This file is the reproduction entry point; it agrees with run_route_b.ps1, and every gate below
# is fixed to the value that produced the reported numbers.
$ErrorActionPreference = "Continue"
# Derive the repository root from this script's own location, so that no absolute path containing
# non-ASCII characters has to be hard-coded (such paths break under a child shell's encoding).
$ws = Split-Path $PSScriptRoot -Parent
Set-Location $ws

# ---- 1. Credentials and machine-local paths: nothing is stored in this file ----
# Variables already present in the environment win; otherwise the untracked `secrets.local.ps1`
# at the repository root is dot-sourced (copy secrets.example.ps1 to create it).  That file
# injects the LLM endpoint as SLC_LLM_*, the Neo4j / MongoDB credentials as NEO4J_* / MONGO_* and
# the local handoff-bundle root as SLC_ROOT.
. "$PSScriptRoot\load_secrets.ps1"
$slcSecretsSource = Import-SlcSecrets -Root $ws -Quiet

# ---- 1b. Handoff-bundle root (the one value that differs per machine; set it in secrets.local.ps1)
if (-not $env:SLC_ROOT) {
    $env:SLC_ROOT = 'D:\PATH\TO\SLC_fresh'      # placeholder: set the real path in secrets.local.ps1 or in the environment
}
if (-not (Test-Path $env:SLC_ROOT)) {
    Write-Warning "SLC_ROOT does not exist: $($env:SLC_ROOT) -- the local cache of Kira and the compute tools will be unavailable"
}

# ---- 2. Interpreter / cache roots / tool timeouts ----
$env:SLC_KMTIMER_ROOT      = $env:SLC_ROOT + '\outputs\tool-cache\KMTIMER'
$env:SLC_TOOLS_CACHE_ROOT  = "$ws\outputs\slc-tools-cache"
$env:SLC_MCP_PYTHON        = "$ws\outputs\mcp-venv\Scripts\python.exe"
$env:SLC_MCP_PUBLIC_INPUTS = "$ws\outputs\mcp-public-inputs-all647.json"
$env:SLC_MCP_TOOL_TIMEOUT  = '45'
$env:SLC_TOOLS_OFFLINE     = '0'
$env:SLC_MCP_ALLOW_DOWNLOADS = '0'
$env:SLC_MCP_ALLOW_BROWSER_NETWORK = '0'
$env:SLC_TCGA_SAMPLES_PER_GROUP = '6'
$env:SLC_MCP_R_TIMEOUT     = '900'

# ---- 3b. R + DESeq2 / GDC cache (machine-local paths, not credentials) ----
$env:R_HOME = 'C:\Program Files\R\R-4.6.1'
$env:R_LIBS_USER = "$ws\outputs\R-libs"
$env:SLC_GDCDATA_ROOT = "$ws\outputs\slc-tools-cache\GDCdata"
$env:SLC_DE_TIMEOUT = '900'; $env:SLC_DE_CACHE = '1'; $env:SLC_DE_TOPN = '400'

# ---- 4. Runtime ----
$env:PYTHONUTF8 = '1'; $env:GIT_PYTHON_REFRESH = 'quiet'
$env:SLC_DATASET = 'clean'
$env:SLC_TRACE_VERSION = '20'          # trace cache key of the released run; bump it only when the decision/evidence
                                       # structure changes, since that invalidates the cached traces

# ---- 5. Gates of the released configuration (how to roll one back: see README.md) ----
$env:SLC_PRIOR_AGENT   = '1'           # prior agent: a third, memory-based evidence channel that runs in parallel with Kira and Cael
$env:SLC_DEBATE        = '1'           # expert debate: one extra parallel vote round on ambiguous cases only; it never overrides an evidence channel
$env:SLC_LLM_CHIEF     = '1'           # LLM chief judge: knowledge and compute evidence serve as verification material, the verdict is the LLM's
$env:SLC_PAIRWISE_MODEL = "$ws\outputs\traces\pairwise_graph_model.json"  # logistic model on pairwise graph features (frozen artifact)
$env:SLC_TRIAD_VOTE    = '1'           # triad vote: the evidence channel plus two pure-LLM votes
$env:SLC_TRIAD_CRITIQUE = '1'          # on a split vote, one round of parallel cross-critique first (heterogeneous role personas)
$env:SLC_PRIOR_RACE    = '2'           # fire the prior channel twice and keep the first reply (absorbs endpoint long-tail timeouts)
$env:SLC_MEM_R2        = '1'           # R2: re-rank membership evidence by essentiality -- the one step the released configuration adds

# ---- 6. Gates for the ablations that did not pay off ----
# Left as set, they reproduce the released numbers byte for byte; flipping one re-runs the
# corresponding negative result, which is why the switches stay documented here.
$env:SLC_PRIOR_ENSEMBLE  = '1'         # >1: ensemble of several prior channels (negative)
$env:SLC_ANSWER_ENSEMBLE = '1'         # >1: ensembling at the answer layer (negative)
$env:SLC_PRIOR_RETRIEVE  = '0'         # 1: literature retrieval as an extra vote source for the prior (not adopted)
$env:SLC_NEG_EVID        = '0'         # 1: explicit negative-evidence channel for membership (negative)
$env:SLC_ABSTAIN_THETA   = '0'         # >0: selective abstention below a calibrated threshold (negative)

$env:Path = "$env:R_HOME\bin\x64;$env:R_HOME\bin;" + $env:Path

Write-Host ("SLC-MARS released env: TRACE={0} PRIOR={1} DEBATE={2} CHIEF={3} PWM={4} TRIAD={5} CRITIQUE={6} RACE={7} MEM_R2={8} | MONGO_set={9} creds={10}" -f `
    $env:SLC_TRACE_VERSION, $env:SLC_PRIOR_AGENT, $env:SLC_DEBATE, $env:SLC_LLM_CHIEF, `
    ($null -ne $env:SLC_PAIRWISE_MODEL), $env:SLC_TRIAD_VOTE, $env:SLC_TRIAD_CRITIQUE, `
    $env:SLC_PRIOR_RACE, $env:SLC_MEM_R2, ($env:MONGO_URI.Length -gt 10), $slcSecretsSource)
