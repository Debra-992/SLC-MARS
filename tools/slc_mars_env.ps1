# SLC-MARS 定版（frozen v20）评测环境。
# 用法：在同一个 shell 块里 `. tools\slc_mars_env.ps1` 后再启动进程（Start-Process 不继承后续块的环境）。
# 与 run_route_b.ps1 同口径；本文件即复现入口，所有门控已固化为 v20 定版值。
$ErrorActionPreference = "Continue"
# 用脚本自身位置推导仓库根，避免硬编码 CJK 路径在子 shell 编码下错乱
$ws = Split-Path $PSScriptRoot -Parent
Set-Location $ws

# ---- 1. 凭据与本机路径：一律不在本文件里，先解析 ----
# 优先用已在环境中的变量；否则 dot-source 仓库根的 secrets.local.ps1（未纳入版本管理，
# 由 secrets.example.ps1 复制生成）。LLM api_key 同样由该处注入 SLC_LLM_* 三个变量，
# Neo4j / MongoDB 凭据由 NEO4J_* / MONGO_* 注入，SLC_fresh 交接包根由 SLC_ROOT 注入。
. "$PSScriptRoot\_load_secrets.ps1"
$slcSecretsSource = Import-SlcSecrets -Root $ws -Quiet

# ---- 1b. SLC_fresh 交接包根（唯一需要按本机修改的一行，写在 secrets.local.ps1 里）----
if (-not $env:SLC_ROOT) {
    $env:SLC_ROOT = 'D:\PATH\TO\SLC_fresh'      # 占位：在 secrets.local.ps1 或环境里设真值
}
if (-not (Test-Path $env:SLC_ROOT)) {
    Write-Warning "SLC_ROOT 不存在：$($env:SLC_ROOT) —— Kira/计算工具的本地缓存会不可用"
}

# ---- 2. 解释器 / 缓存根 / 工具超时 ----
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

# ---- 3b. R + DESeq2 / GDC 缓存（本机路径，非密钥）----
$env:R_HOME = 'C:\Program Files\R\R-4.6.1'
$env:R_LIBS_USER = "$ws\outputs\R-libs"
$env:SLC_GDCDATA_ROOT = "$ws\outputs\slc-tools-cache\GDCdata"
$env:SLC_DE_TIMEOUT = '900'; $env:SLC_DE_CACHE = '1'; $env:SLC_DE_TOPN = '400'

# ---- 4. 运行时 ----
$env:PYTHONUTF8 = '1'; $env:GIT_PYTHON_REFRESH = 'quiet'
$env:SLC_DATASET = 'clean'
$env:SLC_TRACE_VERSION = '20'          # 定版缓存键；改判定/证据结构才 +1（会让旧 trace 缓存失效）

# ---- 5. SLC-MARS v20 定版门控（回退方法见 README「配置回退」）----
$env:SLC_PRIOR_AGENT   = '1'           # Prior 智能体：与 Kira∥Cael 并行的第三路记忆先验证据通道
$env:SLC_DEBATE        = '1'           # 专家辩论（仅在歧义处开一轮并行表决，不推翻证据通道）
$env:SLC_LLM_CHIEF     = '1'           # LLM 主审：知识/计算作为校验材料，最终判定交给 LLM 法官
$env:SLC_PAIRWISE_MODEL = "$ws\outputs\traces\pairwise_graph_model.json"  # pairwise 图特征 logistic 工件
$env:SLC_TRIAD_VOTE    = '1'           # 三通道表决（证据 ∥ 纯LLM×2）
$env:SLC_TRIAD_CRITIQUE = '1'          # 冲突态先走一轮并行交叉批评（角色异构 persona）
$env:SLC_PRIOR_RACE    = '2'           # prior 通道双发竞速，治端点长尾超时
$env:SLC_MEM_R2        = '1'           # R2：membership 必需性证据重排（v20 相对 v16 的唯一增量）

# ---- 6. 已否定实验臂的门控（保持关闭 = v20 逐字节；置 1 即复现对应阴性结果）----
$env:SLC_PRIOR_ENSEMBLE  = '1'         # >1 → v13 先验通道集成（阴性）
$env:SLC_ANSWER_ENSEMBLE = '1'         # >1 → v14 答案层集成（阴性）
$env:SLC_PRIOR_RETRIEVE  = '0'         # 1 → v17 L2 检索票源（未采纳）
$env:SLC_NEG_EVID        = '0'         # 1 → v18 membership 负证据通道（阴性）
$env:SLC_ABSTAIN_THETA   = '0'         # >0 → v18 选择性弃权（阴性）

$env:Path = "$env:R_HOME\bin\x64;$env:R_HOME\bin;" + $env:Path

Write-Host ("SLC-MARS v20 env: TRACE={0} PRIOR={1} DEBATE={2} CHIEF={3} PWM={4} TRIAD={5} CRITIQUE={6} RACE={7} MEM_R2={8} | MONGO_set={9} creds={10}" -f `
    $env:SLC_TRACE_VERSION, $env:SLC_PRIOR_AGENT, $env:SLC_DEBATE, $env:SLC_LLM_CHIEF, `
    ($null -ne $env:SLC_PAIRWISE_MODEL), $env:SLC_TRIAD_VOTE, $env:SLC_TRIAD_CRITIQUE, `
    $env:SLC_PRIOR_RACE, $env:SLC_MEM_R2, ($env:MONGO_URI.Length -gt 10), $slcSecretsSource)
