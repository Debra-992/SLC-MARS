# SLC-MARS local secrets template.
#
# Copy this file to `secrets.local.ps1` and fill in real values.  `secrets.local.ps1`
# is listed in .gitignore and must never be committed, zipped into a handoff, or
# pasted into a screenshot/log.
#
#   Copy-Item secrets.example.ps1 secrets.local.ps1
#   notepad secrets.local.ps1
#
# If the variables below are already present in the environment (e.g. set by your
# PowerShell profile or a CI secret store), `secrets.local.ps1` is not needed at all:
# run_route_b.ps1 and tools/slc_mars_env.ps1 prefer an already-set variable over the
# file (see the "-not [string]::IsNullOrEmpty" guards there).
#
# Keep this file ASCII-only: PowerShell 5.1 mis-decodes BOM-less UTF-8 with CJK.

# --- LLM endpoint (OpenAI-compatible) -----------------------------------------------
$env:SLC_LLM_BASE_URL = 'https://REPLACE-me.example/compatible-mode/v1'
$env:SLC_LLM_API_KEY  = 'REPLACE-WITH-YOUR-API-KEY'
$env:SLC_LLM_MODEL    = 'REPLACE-WITH-MODEL-NAME'

# --- Neo4j knowledge graph (Kira: project_knowledge / kg_* / PPR) -------------------
$env:NEO4J_URI      = 'bolt://REPLACE-HOST:7687'
$env:NEO4J_USER     = 'neo4j'
$env:NEO4J_PASSWORD = 'REPLACE'

# --- MongoDB literature collection (Kira literature branch / LitRAG baseline) ------
$env:MONGO_URI               = 'mongodb://REPLACE_USER:REPLACE_TOKEN@REPLACE-HOST:27017/SLCdb?authSource=SLCdb'
$env:MONGO_DB_NAME           = 'SLCdb'
$env:MONGO_COLLECTION_PAPERS = 'papers'

# --- Machine-local paths (not credentials, but per-checkout) -------------------------
# Root of the extracted SLC_fresh handoff bundle; KM/TIMER plots and the DESeq2/GDC tool
# caches live underneath it.  Both launchers read this and fall back to a harmless
# placeholder, so never hard-code a personal path into tracked files again.
$env:SLC_ROOT = 'D:\PATH\TO\SLC_fresh'
