# SLC-MARS · 知识供给三智能体架构（对接 SLC multi_agent 真实 8 工具）

> **定版 = v20**（frozen 2026-11-02）。复现入口 `tools/slc_mars_env.ps1`，
> 定版决策流/结果见本文「定版部分」第十六~二十节；总览与快速上手见仓库根 `README.md`。
>
> 依赖 MetaGPT（`<PROJECT_ROOT>\MetaGPT`）。代码骨架位于 `knowledge_agents\`。
> 本设计把 SLC `multi_agent` 的 8 个真实工具按语义划归两个**纯知识/数据供给侧**智能体：
> **知识检索智能体（Kira）** 供给背景知识，**计算智能体（Cael）** 供给候选数值知识，
> **决策智能体（Vega）** 只消费已产出的证据做判定。供给侧互不依赖、可独立运行，
> 下游只依赖统一 `schema.py` 契约。

## 一、工具 → 智能体 映射

| 智能体 | 对接的真实工具 | 源码 | 产出 |
|---|---|---|---|
| **知识检索 `KnowledgeQueryRole`** | `project_knowledge` | slc_benchmark_context_mcp.py | Neo4j 三路证据 + Mongo 文献 |
| | `kg_schema` → `kg_cypher`（text2cypher） | slc_kg_mcp.py | LLM 生成只读 Cypher 补背景 |
| | `kg_gene_neighbors` | slc_kg_mcp.py | 候选缺覆盖时补邻居 |
| **计算 `ComputeRole`** | `project_graph_computation`（PPR） | ppr_weighted.py | 疾病种子加权邻近度打分 |
| | `timer_pancancer_expression` | timer.py | 泛癌肿瘤/正常表达 |
| | `km_survival` | km_plotter.py | 生存分析统计原文 |
| | `slc_differential_expression` | TCGA.py / DESeq2.py | DESeq2 差异表达 |

## 二、总体架构

```
                     下游 Analyst（watch 两类 cause_by）
                              ▲                 ▲
             BackgroundKnowledge            ComputedKnowledge
              (knowledge_type=               (knowledge_type=
                background_kg)                 candidate_data)
                     │                 │
        ┌────────────┴───┐       ┌─────┴──────────────┐
        │ KnowledgeQuery  │       │    ComputeRole      │
        │     Role        │       │  (按需路由 4 类计算) │
        └───────┬────────┘       └─────┬──────────────┘
   KnowledgeRetrieval Action        Computation Action
        │                                │
   RetrievalTools                   ComputationTools
  (project_knowledge/               (graph_computation/
   kg_schema/neighbors/cypher)       timer/km/deseq2)
        └──────────────┬───────────────┘
                slc_backend.Backend
        ┌──────────────┴───────────────┐
   RealMcpBackend                 UnavailableBackend
   (mcp.ClientSession.call_tool    (离线兜底：一律未知，
    调真实 4 个 stdio MCP server)     验证不伪造)
```

## 三、文件与分层

| 层 | 文件 | 职责 |
|---|---|---|
| 契约 | `schema.py` | `ToolStatus` / `BackgroundKnowledge` / `ComputedKnowledge` 及子结构、`parse_knowledge()` |
| Action | `actions/knowledge_retrieval_action.py` | project_knowledge + text2cypher（覆盖不足才触发） |
| Action | `actions/computation_action.py` | 按题面语义路由 PPR/TIMER/KM/DESeq2 |
| Role | `roles/knowledge_query_role.py` | `KnowledgeQueryRole`（Kira）装配 RetrievalTools |
| Role | `roles/compute_role.py` | `ComputeRole`（Cael）装配 ComputationTools，`set_task()` 注入路由参数 |
| Tool | `tools/retrieval_tools.py` | 4 个检索工具适配，只读守卫 |
| Tool | `tools/computation_tools.py` | 4 个计算工具适配，保留 evidence_scope/interpretation |
| Tool | `tools/cypher_guard.py` | 镜像真实 `validate_cypher`（拒绝写/过程/多语句） |
| Tool | `tools/slc_backend.py` | RealMcpBackend（真调）/ UnavailableBackend（兜底），`get_default_backend()` 自动选择 |
| Demo | `demo.py` | 两角色独立演示 |

## 四、证据完整性不变量（照搬真实工具，绝不放宽）

1. **失败/不支持/无数据 == 未知**，映射到 `ToolStatus.{unavailable,download_required,unsupported_cancer,no_seed,rejected,error}`，绝不写成 `ok` 或空成功。
2. **不伪造数值**：TIMER `plot_only` 不臆测数值；KM 缺 HR 不编造；DESeq2 缺计数返回 `download_required`。
3. **只读边界**：text2cypher 的 Cypher 经 `cypher_guard` 校验，写子句/存储过程/多语句直接 `rejected`。
4. **来源独立性显式标注**：`source_independence=NOT_ESTABLISHED`，`public_web_used=False` 如实回传。
5. **按需路由**：计算工具由确定性规则触发（非模型自主 function calling），未路由记 `skipped_tools`，不当阴性。

## 五、如何运行

### 5.1 环境准备（一次性）

```powershell
# 主 venv：跑 MetaGPT 智能体 + MCP 客户端（Python 3.11）
uv venv .venv --python 3.11
.venv\Scripts\python.exe -m pip install -e .\MetaGPT --no-deps
.venv\Scripts\python.exe -m pip install mcp==1.27.2 pydantic==2.* tenacity json5 \
  qianfan dashscope==1.19.3 scikit-learn python-docx Pillow chardet \
  pandas==2.1.1 "numpy<2" "pyarrow<18" tree-sitter tree-sitter-python \
  gitignore-parser grep-ast pygithub nbformat nbclient ipython ipykernel ipywidgets
# （上面这一串是 SLC-Mars 环境里 metagpt 实际会拉起的最小依赖集）

# 工具 venv：SLC MCP server 用（Python 3.11 + requirements-tools）
uv venv outputs\mcp-venv --python 3.11
outputs\mcp-venv\Scripts\python.exe -m ensurepip --upgrade
outputs\mcp-venv\Scripts\python.exe -m pip install `
  -r "<SLC_fresh>\requirements-tools.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5.2 运行

```powershell
# 离线（未装 mcp / 无 SLC_ROOT）：全走 UnavailableBackend，验证不伪造
python -m knowledge_agents.demo

# 路线 B（真实 MCP，本仓库推荐）：直接调启动脚本
.\run_route_b.ps1
```

`run_route_b.ps1` 里必须设的 4 个 env（缺一都会导致工具在子进程里退回 unavailable）：

| 变量 | 值 | 作用 |
|---|---|---|
| `SLC_ROOT` | SLC_fresh 包根 | `RealMcpBackend` 定位 mcp_servers 目录 |
| `SLC_MCP_PYTHON` | `outputs\mcp-venv\Scripts\python.exe` | spawn 每个 stdio server 的解释器 |
| `SLC_KMTIMER_ROOT` | `$SLC_ROOT\outputs\tool-cache\KMTIMER` | TIMER/KM 读缓存的根 |
| `SLC_TOOLS_CACHE_ROOT` | 工程内可写目录 | `_tools_worker.py` 的 cwd + temp |
| `SLC_TOOLS_OFFLINE` | `1` | 只允许读缓存，缺缓存返回 download_required |

### 5.3 已验证的真实输出（离线，走 MCP）

```
[候选知识] status=ok
  TIMER: status=ok scope=plot_only        boxplot=...boxplot_SLC25A1.png
  KM    : status=ok scope=plot_and_stats  stats_head='P value: 0.0032'
  计算候选完成：可用工具 2 项（ppr=not_routed, timer=ok, km=ok, de=download_required）；
              未路由/未知：['ppr:not_routed']。未知不代表阴性。
[背景知识] status=rejected   # qid 不在 SLC_MCP_PUBLIC_INPUTS 白名单，属预期未知
```

## 五·附、常见坑（路线 B 实测踩过的两个）

1. **`mcp.ClientSession` 的 `StdioServerParameters.env` 默认不是继承**：
   若不给 `env=dict(os.environ)`，SLC server 子进程里 `SLC_KMTIMER_ROOT` 全丢，
   `_tools_worker.py` 会拿不到缓存根，`safe_error` 一律回 `unavailable`。
   已在 `slc_backend.py` 里显式传。
2. **`SLC_TOOLS_CACHE_ROOT` 必须显式指到工程内可写目录**：
   默认落到 `%TEMP%\slc-tools`，Windows 沙盒 + `install_read_guard` 组合会拦下
   `mkdir`/`chdir`；一旦 setup 阶段抛 `FileNotFoundError`，就被归类成 `unavailable`
   而不是明确的错误。

## 六、与既有 SLC 五角色管线的衔接

本两智能体等价于管线里的 `Knowledge → Tool` 两段。可直接把 `KnowledgeQueryRole` /
`ComputeRole` 产出的两类 Message 作为 `Analyst` 的上下文（`watch` 两个 `cause_by`），
后续再接 Reviewer / Adjudicator，即构成完整的 MetaGPT 版五角色交接。

## 七、后续 TODO

1. `kg_gene_neighbors` 已封装未接入 Action 分支，按需补齐"邻居补覆盖"链路。
2. RealMcpBackend 每次新建 stdio 会话；可复用网关会话 + 同参缓存（对齐真实工具缓存策略）。
3. 增加 `Analyst` 汇合角色示例，演示两路知识合并消费。

## 八、决策智能体 Vega（三智能体：Kira ∥ Cael → Vega）

在两路供给侧之上加入 **决策角色 `DecisionRole`（Vega）**，只消费 Kira/Cael 已产出的
`BackgroundKnowledge` + `ComputedKnowledge`，**绝不重新取数**。判定内核 `actions/decision_action.py`
支持三档 `SLC_DECISION_MODE`：`heuristic`（确定性打分，免 LLM）｜`llm`｜`auto`（有直接证据走启发式、
证据不足才跨癌种类推）。核心不变量沿用第四节：**未知≠阴性，证据不足即弃答，绝不伪造**。

Vega 把每题的证据聚成三条**同权**通道（`_collect_scores` 返回 `(strength, pol, buf, basis)`）：

| 通道 | 含义 | 取值 | 来源 |
|---|---|---|---|
| `strength` 量级 | 关联/差异表达强度（哪个候选证据更"响"） | ≥0 | KG 路径 / PPR / DE / TIMER |
| `pol` 因果方向 | 时间因果图的促/抑极性（定答案方向） | {-1,0,+1} | RelaEvent 超边方向 ⊗ 题目动词 |
| `buf` 旁系缓冲 | 功能冗余削弱"必需/阳性"（见第九节） | [-1,0] | 目标癌种 DESeq2 面板的同亚家族共表达 |

融合公式（`W_DIRECTION=0.6`、`W_BUFFER=0.8`、`MIN_SCORE_TO_ANSWER=0.5`）：
- **pairwise**：`combined = strength + W_DIRECTION·pol + W_BUFFER·buf`，取最大者为 winner；三者皆不足则弃答。
- **membership**：净极性 `net = pol + W_BUFFER·buf`；`net>0`→阳性、`net<0`→阴性；`net==0`（既无因果方向
  又无缓冲证据）→ **诚实弃答**。置信 `conf = 0.3 + 0.12·|net| + 0.1·log1p(strength)`。

> 关键设计：`buf` 是**单边证据**——只能把判断往"更不必需/阴性"压，**绝不因"无旁系"就臆断基因为必需**。

## 九、① 旁系缓冲（paralog buffering）

**机理**：SLC 基因按亚家族（符号前缀 `SLC<编号>`，如 SLC7/SLC25/SLC12）成簇，同亚家族成员常功能重叠。
若某基因在**目标癌种**里有多个同亚家族旁系被活跃共表达，则敲除它更易被旁系"缓冲"，CRISPR
loss-of-function 越不产生显著必需性表型 → 提示 membership 偏阴性。这直接补"关联/表达 ≠ CRISPR 必需性"
的构念缺口。

**实现** `tools/paralog.py`（数据源 = 该癌种 DESeq2 `top_genes`；拿不到面板即未知、通道关闭）：
- `subfamily()`：`SLC25A32 → SLC25`；非 SLC 命名 → None（不参与冗余估计）。
- `is_functional_symbol()`：剔除伪基因/反义/内含子转录本（`SLC25A14P1`/`SLC22A18AS`/`SLC7A6OS`/`SLC7A2-IT1`/
  `SLC25A21-AS1`），避免大家族仅凭成员数虚增缓冲。
- `panel_from_de(de)`：把 DE 表转成 `{基因: 表达权重}`，权重 = `min(|log2FC|, LNCAP)/LNCAP`（`LNCAP=5`）——
  **既看是否显著、也看差异表达幅度**，强失调的旁系更能实际缓冲。
- `redundancy(gene, panel, k_sat)`：`redundancy = min(1, Σ_旁系 表达权重 / k_sat)`，随共表达强度自然分级。
- `computation_action.py` 在 DE 之后算 `redundancy_map(候选, 面板)` → 写入 `ComputedKnowledge.paralog_redundancy /
  paralog_partners`；无面板记 `skipped=paralog:no_expression_panel`。

## 十、⑤ 保形弃答（conformal calibrated abstention）

用手调的 `MIN_SCORE_TO_ANSWER` 定弃答阈值是**任意的**；本机制改用 **Conformal Risk Control**（Bates 等 2021）
在不相交校准集上标定一个置信阈值，并给出风险-覆盖曲线，使操作点由统计保证而非拍脑袋决定。

- 数学内核 `conformal.py`（纯函数）：非一致性分数 `s = 1 - confidence`；选中作答 ⟺ `s ≤ τ`；
  有限样本"加一"校正经验风险 `L̂(τ) = (Σ loss·1[s≤τ] + 1)/(N+1)`；
  `τ̂ = sup{τ : L̂(τ) ≤ α}`（无满足项则 `τ̂=None` → 全弃答，是正确保守行为）。保证
  `E[loss·1[s≤τ̂}] ≤ α` ⇒ **已答错误率 ≤ α/覆盖率**。
- 校准 + 报告 `conformal_report.py`：按 `sha1(seed|qid)` 把题切成**不相交**的校准/测试集，在校准集求 τ̂、
  在测试集报覆盖率/已答错误率/ECE，并导出阈值文件。
- 在途钩子 `decision_action.py`：`run()` 每条返回都过 `_conformal_filter(ans, bucket)`；当环境变量
  `SLC_CONFORMAL_FILE` 指向的 json 里 `thresholds[bucket]` 存在且 `confidence < 阈值` → **改判弃答**
  （并在 `Answer.conformal_threshold` 记录、rationale 前缀 `[保形弃答 ...]`）。

> ⚠️ 注意：阈值文件若来自 `pipeline_heuristic` 的校准，只应喂给 heuristic 决策；且**校准输入必须是未加保形的
> 原始置信度**（`calib_dump.py` 会显式清除 `SLC_CONFORMAL_FILE`，避免自我实现/循环论证）。
> 校准集很小时，有限样本下界 `1/(N+1)` 会超过 α → τ̂=None（正确地"全弃答"）；要得到可部署的紧阈值需**做大校准池**。

## 十一、复现与运行（①⑤ 相关）

```powershell
# (可选) 预热更多癌种的 DESeq2 缓存：旁系缓冲只在有 DE 面板的癌种生效
.\run_route_b.ps1 -Mode prewarm                          # 下载 ON，逐癌种跑一次并落盘缓存

# ⑤-1 生成大规模保形校准池（只跑 heuristic，免 LLM，工具/检索为主）
.\run_route_b.ps1 -Mode calib -Eval 250 -Seed 5          # -> outputs/traces/calibstore.jsonl

# ⑤-2 在校准池上标定阈值 + 风险-覆盖 + ECE，导出部署阈值
$env:PYTHONPATH="<PROJECT_ROOT>"
.\.venv\Scripts\python.exe -m knowledge_agents.conformal_report `
    --traces outputs\traces\calibstore.jsonl --version 8 --system pipeline_heuristic --alpha 0.10
#   -> outputs/traces/conformal_report.md + conformal_thresholds.json（α 太小时不产出阈值=正确保守）

# 公平三系统对照（heuristic / auto / 原生 MetaGPT 基线，共享同一份证据）
.\run_route_b.ps1 -Mode compare -Eval 40 -Seed 5         # -> outputs/traces/compare_table_n40_seed5.md

# 部署在途保形：把导出的阈值挂回决策层再答一遍
$env:SLC_CONFORMAL_FILE = "<PROJECT_ROOT>\outputs\traces\conformal_thresholds.json"
```

**①⑤ 环境变量旋钮**

| 变量 | 默认 | 作用 |
|---|---|---|
| `SLC_PARALOG_KSAT` | `4` | 旁系冗余饱和常数（Σ 表达权重到达此值即 redundancy=1） |
| `SLC_CONFORMAL_FILE` | 空 | 指向 `{thresholds:{bucket:conf阈值}}` json；置空=不启用在途保形 |
| `SLC_PAIRWISE_MODEL` | 空 | 指向 pairwise 图特征 logistic 工件 json；置空=pairwise 走纯 v9 规则（v10 可逆开关） |
| `SLC_DECISION_MODE` | `auto` | `heuristic`｜`llm`｜`auto` |
| `SLC_DE_TOPN` | `400` | DESeq2 返回宽面板（决定旁系可见度） |
| `TRACE_VERSION`（env `SLC_TRACE_VERSION`） | `20` | 判定/证据结构变更即 +1，令旧缓存失效（定版=20） |

**离线自检**（免 DB/LLM/工具，29 项）：
```powershell
$env:PYTHONPATH="<PROJECT_ROOT>"
.\.venv\Scripts\python.exe tools\test_paralog_conformal.py    # 旁系缓冲 + 保形数学 + 在途钩子
```

---

## 十二、membership 改进（①修正 + ②③④可行性 + ⑤学习式校准 + ⑥）

针对最弱的 membership 桶，按优先序逐条落地（v9，n=250 clean 校准池，seed5，qids 与
启发式判定同分布）：

### ① 去非对称 + 修 `net==0` 浮点陷阱（已落地，v9）
`decision_action._heuristic` membership 分支重写：阴/阳**极性只由因果方向 `pol` 决定**；
旁系缓冲 `buf` 只能在已有方向时**削弱净强度**，绝不能在 `pol=0` 时单独把基因臆断为阴性
（unknown≠negative）。旧的 `net=pol+W_BUFFER*buf; if net==0` 会在 `pol=0` 时仅凭缓冲
（`net=0.8*buf<0`）批量制造假阴，且浮点恰好等于 0 的判定漏网。新逻辑：`if pol==0 → 弃答`；
`if abs(net)<1e-9 → 弃答`。离线 29 项全过（含“仅缓冲→弃答，不再伪造阴性”）。

效果（v9 启发式，同 250 池）：membership 覆盖从旧 56%→**40.3%**（48/119），把缓冲-only
的臆断答案诚实地收回；但**已答准确仅 41.7%**（≈53% 少数类 dummy）——说明 polarity 驱动的
残余信号仍不足以判必需性。保形在 α=0.1 下对 membership **无可认证阈值**（其校准半池仅 22 已答）。

### ②③④ CRISPR 阳性通道 / 极性重定向 / 细胞系匹配 —— 均被数据可得性与防泄漏封锁（不可行）
经源码核查，本项目的“真值”本身来自 CRISPR（MAGeCK Low/High-Growth）筛选，而公开工具**刻意
不供给**同类信号以防泄漏：
- `slc_benchmark_context_mcp.project_knowledge` 硬编码 `matched_screen_measurements_available=False`；
- `tool_runtime.BLOCKED_PARTS` 明确屏蔽 `onco_crisprdb_slc`、`benchmark_791`（即唯一真实 CRISPR
  必需性数据＝被封锁的真值源，用作特征＝作弊）；
- 图谱仅 `SLCGene -[:IS_SOURCE]-> RelaEvent <-[:IS_TARGET]- Disease`，`rel_AB/rel_BC∈{PROMOTION,SUPPRESSION}`
  是**基因→疾病的关联极性**，**无 knockout/viability 边**（③无数据可重定向）；
- 表达只有 TIMER/TCGA **bulk 肿瘤/癌旁**，**无 CCLE 细胞系**面板（④无法做细胞系级匹配）。
结论：②③④ 在“不泄漏 CRISPR 真值 + 只用现有 8 工具”的硬约束下**不可实现**，如实登记而非臆造。

### ⑤ 免泄漏 logistic 校准 + 保形（决定性改进，`tools/train_membership_logit.py`）
既然手工常数（`W_BUFFER`、`0.3+0.12|net|`）把弱信号压成接近随机，改让 **500 clean 真值**去学
“证据→必需性”的权重。`calib_dump._features` 从工具原始数值抽出**免泄漏特征向量**（基因→疾病
促/抑计数 `n_promote/n_suppress/n_paths/n_cancer_hit`、PPR、TIMER/KM 可用位、目标基因 DE
`log2FC`、面板规模、旁系冗余），标签＝`ground_truth.answer`。严格防泄漏：
- 训练/评测按 **qid 哈希不相交**切分（同题标签只落一侧）；`C` 只在训练半池内 CV；概率经
  `CalibratedClassifierCV` 校准以充当置信度；
- 与启发式在**同一 test 子集**对比，并给多数类 dummy 作下限。

结果（N=119 membership，essential base=52.9%）：
- **重复 10×5-fold CV：AUC=0.730 ± 0.017（min 0.701）**，acc@0.5 68–71% —— 免泄漏特征里
  确有可用信号，是手工规则**没用满**，而非数据无信号。
- 单折不相交切分（train81/test38）：logistic 全答 60.5% 准确（>50% dummy）；叠加保形：

  | 判据 | 覆盖 | 已答准确 | 净贡献 |
  |---|---|---|---|
  | 多数类 dummy | 100% | 50.0% | 50% |
  | ①手工启发式 | 47% | 61.1% | **28.9%** |
  | ⑤ logistic @0.5 全答 | 100% | 60.5% | **60.5%** |
  | ⑤ logistic＋保形 α=0.20 | 89.5% | 64.7% | **57.9%** |
  | ⑤ logistic＋保形 α=0.10 | 21.1% | 75.0% | 15.8%（高精度可弃答）|

  学习式在覆盖与准确率上**同时**优于启发式与 dummy，净贡献近乎翻倍。
- **learned 权重**显示 `de_log2FC(+1.16)`、`paralog_redundancy(+0.79)`、`n_promote(+0.48)`、
  `n_paths(+0.40)` 为主导。注意 `paralog_redundancy` 的**学习符号为正**，与启发式“缓冲降必需性”的先验
  **反号**——这正是 ⑥ 无法靠调 `k_sat` 救回的原因（先验本身方向错了），而 ⑤ 通过数据自动纠正。

复现：
```powershell
# 1) 生成带特征的 v9 校准池（须在能被 spawn MCP 检索服务器的权限下运行；沙箱会拒绝 project_knowledge）
.\.venv\Scripts\python.exe -m knowledge_agents.calib_dump --n 250 --seed 5
# 2) membership 学习式校准 + 保形评测
.\.venv\Scripts\python.exe tools\train_membership_logit.py --seed 5 --train-frac 0.6 --alphas 0.10,0.20,0.30
```

### ⑥ 旁系饱和/门控调优 —— 被 ⑤ 取代
⑤ 的权重分析表明旁系冗余与必需性在本数据中呈**正**相关，与 ①/启发式所设“缓冲削弱必需性”的先验相悖；
仅调 `SLC_PARALOG_KSAT`/饱和无法修正错向先验。故 ⑥ 不再单独手调，交由 ⑤ 的数据驱动权重处理（若要把 ⑤
接入 Vega 判定层，需 bump `TRACE_VERSION→10` 并落一份序列化模型工件，属架构级改动，另行确认）。

> 运维要点：`project_knowledge`/`project_graph_computation` 由 `.venv` 派生 MCP stdio 服务器；
> 若运行环境（如受限沙箱）**拒绝派生该服务器**，Kira 会整批返回 `status=unavailable`
> （`kg.evidence=0 → pol=0`），使启发式 membership 全弃答、pairwise 覆盖骤降——此为**环境**问题，
> 非判定逻辑问题。校验：`evidence_view.kg_evidence_n` 均值应为个位数而非 0。

---

## 十三、图谱 SLC→Pathway→Disease 链的拆细榨取（membership 已封顶 / pairwise 大增）

回答"图谱里看得到 SLC-pathway-disease 关系，用得上吗"：用得上，但要**分任务**看，且此前只榨了
复合极性 `gene_disease_effect = rel_AB⊗rel_BC` 一段。本轮把这条链**拆细**并持久化原始路径行，量化其增量。

**改动（不改判定逻辑，`TRACE_VERSION` 仍为 9）**：`calib_dump._features` 增加拆细图特征
（`n_ab_promote/suppress`、`n_bc_promote/suppress`、`n_promote_matched/suppress_matched`（命中本癌种疾病的
促/抑拆分）、`n_distinct_pathways(_matched)` 中介通路多样性）；`calib_dump._kg_paths` 落盘每题
**全部候选基因**的 `gene_disease_paths` 原始行（gene/pathway/rel_AB/rel_BC/effect/disease）+ `cancer`，
使之后任何图特征增强可离线秒算、**无需重跑池**。pairwise 的 per-gene 图特征即从 `kg_paths` 离线重建。

**membership —— 拆细不涨（5 seed 稳健）**：以 `features` 拟合 logistic，qid 不相交切分，重复 10×5 折 CV-AUC：

| seed | base(复合极性+组学) | base + graph(拆细) |
|---|---|---|
| 5 | 0.758±0.012 | 0.742±0.016 |
| 7 | 0.751±0.013 | 0.741±0.016 |
| 11 | 0.747±0.012 | 0.732±0.011 |
| 23 | 0.623±0.025 | 0.622±0.025 |
| 99 | 0.756±0.015 | 0.736±0.019 |

拆细**从不提升**（0 至 −0.02）：复合极性已把这条链里与"必需性"相关的方向信号**吃满**，再拆 `rel_AB/rel_BC`
与通路多样性在 n≈81 训练点上噪声多于信号（轻度稀释）。学到的拆细权重（`n_bc_promote +0.36`、
`n_ab_promote +0.25`、`n_distinct_pathways +0.23`）说明它们**有信号但不可外推**。结论：图谱链对
membership 的价值 = base 里那点复合极性，已榨干，无更多可取。

**pairwise —— 拆细 + 学习式 pointwise 归约 是决定性增益（5 seed 稳健）**：把"二选一"归约为逐基因打分
（同题 winner 记正例、loser 记负例；per-gene 特征**只用** `kg_paths` 图特征），test 上两基因 argmax 判 winner：

| seed | logistic winner-argmax（全图特征） |
|---|---|
| 5 | 74.0% (37/50) |
| 7 | 75.9% (44/58) |
| 11 | 79.6% (39/49) |
| 23 | 77.3% (34/44) |
| 99 | 75.6% (34/45) |

held-out 稳定 **≈74–80%（均值 ~76.5%）**，远超 dummy(50%) 与手工启发式(seed5 同 test：答题 43/50、
已答准确 60.5%、净贡献 52.0%)。且这是**仅用图谱链**（不含 PPR/TIMER/KM/DE）做到的——即"SLC-pathway-disease
关系"在 pairwise 排序上还有**大量此前未榨的价值**，本轮用学习式图特征兑现了出来。

复现（免重跑，读现有 calibstore）：
```powershell
.\.venv\Scripts\python.exe tools\train_membership_logit.py            # membership base-vs-graph 消融 + 保形
.\.venv\Scripts\python.exe tools\train_membership_logit.py --pairwise # pairwise 图特征 logistic（pointwise）
```

> 若要接入 Vega 判定层（把 pairwise 图特征 logistic 作为排序打分、membership 保留复合极性），
> 属"手工规则 → 数据驱动"的架构级改动，需落序列化模型工件并 bump `TRACE_VERSION→10`，另行确认。

---

## 十四、v10：pairwise 图特征 logistic 接入 Vega 判定层 + 三方法公平对比重刷

§十三 的兑现：pairwise 的启发式二选一**替换**为训练出来的图特征 logistic（`SLC_PAIRWISE_MODEL` 门控，
membership 保持 v9 路径不动）。`TRACE_VERSION 9→10`（决策结构变更，旧 trace 缓存全部失效重算）。

**构件（单一来源，杜绝训练/服务偏移）**

| 件 | 文件 | 说明 |
|---|---|---|
| 特征定义 | `knowledge_agents/graph_features.py` | `PAIR_COLS`(12 项拆细图特征) + `gene_graph_feat` + `feat_vec` + `collect_path_rows`；calib 落盘、离线训练、序列化拟合、在线判定**四方共用**同一实现 |
| 拟合/序列化 | `tools/fit_pairwise_model.py` | 在 qid 哈希 TRAIN 切分（seed5, frac0.6；162 基因行/81 题）上 GridSearchCV→标准化 logistic，导出 JSON 工件（features/mu/sd/coef/intercept/trained_qids/holdout）→ `outputs/traces/pairwise_graph_model.json` |
| 在线推理 | `decision_action._pairwise_model/_score_pairwise_graph` | **免 sklearn**：只算 `(x−μ)/σ→coef·z+b→sigmoid`，按 argmax 出 winner，confidence=赢家概率 |
| 回退不变量 | 同上 + `_heuristic` pairwise 分支 | env 未设 / 工件坏 / **两候选皆无 gene_disease_paths 图证据** → 诚实回退 v9 规则，绝不伪造模型排序 |

一致性验证：`fit_pairwise_model.py` 的 holdout argmax = **74.0% (37/50)**，与 `train_membership_logit.py --pairwise`
(seed5) **逐题一致**（同一共享特征模块、同一 qid 切分）——无训练/服务偏移。离线测试 `tools/test_pairwise_graph_model.py`
7/7 通过；`test_paralog_conformal` / `test_direction_logic` 全绿。单测在文件头把会影响判定路径的
运行时门控（`SLC_MEM_R2`、`SLC_PAIRWISE_MODEL` 工件等）钉回默认值，因此在"先 dot-source 过
`tools/slc_mars_env.ps1`"的 shell 里也给出同样结果——否则会拿定版 v20 的重排行为去比裸启发式的断言，得到假失败。

**三方法公平对比重刷（n=40, seed5, clean，pairwise 臂=部署的 v10 管线，新算 40/复用 0）**

pairwise (n=22)：v10 图模型臂 对12/错8/弃2，已答 60.0%、全体 54.5%（v1·n100 旧启发式为 65.8%/53.2%）；
auto 50.0%；原生 MetaGPT 已答 91.7% 但只答一半（覆盖 54.5%），全体 50.0%。
membership (n=18)：与 v9 相同路径（heuristic 16.7% / auto 27.8% / baseline 22.2%全体），本轮**未动**。

**诚实拆分（防泄漏：工件带 `trained_qids`，`tools/analyze_v10_split.py` 按 qid 是否见过拆开）**

| pairwise·v10 图模型臂 | n | 对 | 错 | 弃 | 准确率(已答) | 准确率(全体) |
|---|---|---|---|---|---|---|
| 训练内（qids∈工件81） | 14 | 6 | 7 | 1 | 46.2% | 42.9% |
| **留出（未见过的题）** | **8** | **6** | **1** | **1** | **85.7%** | **75.0%** |

留出 75% 与离线 trainer 的 74–80% 带宽吻合；训练内反而偏低（43%）——C=0.01 强收缩、模型近乎不记忆，
训练内无乐观红利，**54.5% 的全体数是保守混合而非泄漏膨胀**。注意 n 小（留出仅 8 题），单 seed 波动大。

> 复现：`$env:SLC_PAIRWISE_MODEL=<repo>\outputs\traces\pairwise_graph_model.json; .\run_route_b.ps1 -Mode compare -Eval 40 -Seed 5`，
> 拆分报告 `.\.venv\Scripts\python.exe tools\analyze_v10_split.py`。撤销接入 = 清空该 env（回到纯 v9 行为）。

## 十五、baseline 扩展：五对比系统同场公平评测（pure-LLM / 文献RAG / AutoGen / MDAgents / 原生MetaGPT）

v10 只有 1 个 baseline（原生 MetaGPT）。本轮把对照组扩到 **5 个**，与管线双臂共 **7 系统**、
同一 40 题（=v10 sample_manifest 的 seed5 clean 样本）、同一 qwen3.8-flash 端点、同一 `_score` 口径。

**组件与隔离（`knowledge_agents/baselines/`）**

| 文件 | 系统 | 运行环境 | 要点 |
|---|---|---|---|
| `common.py` | 全部 | 任意 py3.11+ | 共享件：BASELINE_PROMPT（逐字复制自 baseline_metagpt，禁 abstain 强出答案）、`pending()`（只跳过无 error 行=断点续跑+失败重试）、`normalize()`（winner 不在候选/缺字段→诚实 abstain）、`llm_settings()`（端点三键解析：**`SLC_LLM_*` 环境变量优先**，其次仓库 `config2.yaml`；yaml 里未替换的占位符按未配置处理，绝不把占位符当密钥发出去） |
| `export_inputs.py` | — | `.venv` | 导出 **同题输入两文件严格分离**：`baselines_input.json`（runner 可读，无 GT）/ `baselines_gt.json`（仅 merge 评分读，防泄漏） |
| `run_pure_llm.py` | ①纯LLM | `.venv`（openai SDK） | AsyncOpenAI 并发，超时会诚实记 error 并在重跑时自动重试 |
| `run_lit_rag.py` + `mongo_rag.py` | ②文献RAG | `.venv`（pymongo） | 题面 SLC 基因 Mongo 正则粗召回（按基因缓存）→ 疾病词(title×3+abstract×1)+基因频次重排 top5 → 摘录拼进与①同式的 prompt；**检索失败不静默退化成纯LLM，如实 abstain**；凭证只从 `MONGO_URI` 环境变量 |
| `run_autogen.py` | ③AutoGen | `outputs/autogen-venv`（py3.11+pyautogen 0.2.35） | AssistantAgent+UserProxyAgent(NEVER, 不执行代码)，题面=①的同一 prompt；末条消息常是裸 TERMINATE → **倒扫 assistant 历史取最近可解析 JSON** |
| `run_mdagents.py` | ④MDAgents | `outputs/mdagents-venv`（py3.13.5）+ 克隆 `outputs/MDAgents` | **不改官方源码**，纯 monkeypatch：`OpenAI` 子类强制 qwen 端点/模型名+300s 超时、`google.generativeai` 桩、stdout 重配 UTF-8（官方代码 cprint 医生 emoji，GBK 控制台会炸）；adaptive 路由（basic/intermediate/advanced）原样保留；`_decision_text` 递归展平（intermediate 返回 `{'majority':{0.0:text}}`，str() 会变单引号 repr 毁掉 JSON） |
| `run_metagpt.py` | ⑤MetaGPT | `.venv` | 框架原生五角色交接、**standalone**（只用框架自带知识，无本管线证据）→ `answers_metagpt.jsonl`。与下表历史快照里的 `metaGPT-native` **不是同一个系统**：后者（tracestore `systems.baseline`）是把本管线已产出的同一份证据直接喂给 MetaGPT。定版论文 Table 1 的 MetaGPT 行取 standalone；两行不可互换引用 |
| `merge_baselines.py` | 汇总 | `.venv` | 管线两臂（heuristic / auto）从 tracestore 按 `trace_version` 直取 outcome + 五个 `answers_<system>.jsonl` 现算（qid 追加历史取最后一条；error⇒abstain），另附 `metaGPT-native(shared-evidence)` 诊断臂一行 → `outputs/traces/baselines_table_master.md` |

**公平性口径**：五个 baseline 全部"仅用该 LLM 自身知识（②多给检索文献）"，无知识图谱/DESeq2/保形工件；
③④框架自带的医学多智能体协作逻辑原样运行，只换端点不换提示语义；失败一律诚实 abstain，不伪造答案。

**零检索审计（回应"pure-llm 高分是不是偷偷检索"）**：①单次 completion 不传 tools/functions；
③pyautogen AssistantAgent 未注册任何 tool、UserProxyAgent `code_execution_config=False`；
④官方 `Agent.chat()` 唯一出口是 chatcompletions，runner 传 `dataset='custom'`+空 examplers 关掉其题库通道；
两个框架 venv 均无 duckduckgo/googlesearch/tavily/bs4 等检索库。**baseline 高分的真实来源 = 禁 abstain ×
DepMap/CRISPR 文献在 LLM 训练语料内（GT 与模型记忆同源）× 本样本 negative 桶先验红利，不是检索。**

**踩坑存档**：pyautogen 全版本要求 py<3.13（系统 3.14 装不上→用 uv 的 3.11 基座建 venv）；MDAgents utils.py 的
f-string 含反斜杠要求 py≥3.12；openai 1.14.2 与 httpx≥0.28 不兼容（`proxies` kwarg）→ 钉 httpx<0.28；
AsyncOpenAI 无 `max_connections` 参数 → 并发上限要走 `httpx.AsyncClient(limits=...)`。

**七系统同表（n=40 同一题集 · seed5 · clean · 全部 40/40 零 error 落盘）**

> ⚠️ 下表是 **v10 时代 n=40 的历史快照**，仅用于说明对照组件与口径；
> 定版全量结果（n=489 / B=265）见第二十节与 `outputs/traces/baselines_table_master.md`。
> 下表中的 `metaGPT-native` 是**共享证据**臂（tracestore `systems.baseline`），
> 与第二十节/论文 Table 1 的 MetaGPT 行（standalone `answers_metagpt.jsonl`）**不是同一个系统**。

pairwise（n=22）：

| 系统 | ✓/✗/弃 | acc(全体) | acc(已答) |
|---|---|---|---|
| pipeline-heuristic（v10 图模型臂） | 12/8/2 | 54.5% | 60.0% |
| pipeline-auto | 11/9/2 | 50.0% | 55.0% |
| metaGPT-native | 11/1/10 | 50.0% | 91.7% |
| pure-llm | 19/3/0 | **86.4%** | 86.4% |
| lit-rag | 12/10/0 | 54.5% | 54.5% |
| autogen | 19/3/0 | **86.4%** | 86.4% |
| mdagents | 19/3/0 | **86.4%** | 86.4% |

membership（n=18）：

| 系统 | ✓/✗/弃 | acc(全体) | acc(已答) |
|---|---|---|---|
| pipeline-heuristic | 3/5/10 | 16.7% | 37.5% |
| pipeline-auto | 5/6/7 | 27.8% | 45.5% |
| metaGPT-native | 4/0/14 | 22.2% | 100.0% |
| pure-llm | 10/8/0 | 55.6% | 55.6% |
| lit-rag | 9/9/0 | 50.0% | 50.0% |
| autogen | 12/6/0 | 66.7% | 66.7% |
| mdagents | 12/6/0 | 66.7% | 66.7% |

**诚实解读（这一轮 baseline 扩展暴露的关键事实）**
1. **禁 abstain 后，纯 LLM 先验在本样本上全面压过图谱管线**：pairwise 上 pure-llm/autogen/mdagents 三者
   **同为 19/3（86.4% 全体）** > v10 图模型臂 54.5%。此前"原生 MetaGPT 已答 91.7%"的高准确率被 45% 弃答稀释；
   换成强制作答的同一提示，弃答红利消失，裸模型直接吃满（22 题中 17 题为 negative 桶，GT winner 与"文献中心度"
   这一 LLM 先验高度合拍）。管线的差异点在**可校准的弃答与证据可溯源**，不在这份样本的裸命中率。
2. **文献 RAG 帮倒忙**：lit-rag pairwise 54.5% 明显低于 pure-llm 86.4%——摘录把模型带向"文献里出现多=更必需"
   的表面统计，且长上下文拖慢端点制造超时。membership 上 50.0% 也低于 pure-llm 55.6%。
3. **membership 桶强制作答的 baseline 全体分数占优**（pure-llm 55.6%/autogen=mdagents 66.7% vs 管线 16.7~27.8%），
   因为管线端有保形弃答（大量 abstain 拉低全体 acc）而 baseline 端被提示禁止弃答。本 18 题里 10 题 GT=true，
   **"全部答 true" 就有 55.6% 地板分——pure-llm 恰好踩在地板上**（10/18），autogen/mdagents 66.7% 略高于地板；
   两口径不可直接混比：**已答口径**下管线 auto 45.5% vs autogen/mdagents 66.7%。
4. **多智能体框架相对裸模型零增益、成本数量级**：autogen（双角色无工具空转）与 pure-llm 逐题同分（19/3）；
   mdagents 自适应路由 + 5 专家辩论/3 团队 MDT 的最终分布也与 autogen 完全同分（pairwise 19/3、membership 12/6），
   但单题耗时 ~4–17min（vs 纯LLM ~30s）。在本任务形态下，框架协作复杂度没有兑换成准确率。

> 复现：见上表组件行；四 runner 均可断点续跑（pending 只认无 error 行），七系统表一键重算：
> `.\.venv\Scripts\python.exe -m knowledge_agents.baselines.merge_baselines`。
> mdagents 防御性补丁（官方源码零改动）：`_safe_parse_hierarchy`（hierarchy 不合 "A > B" 模板时平铺，
> 该树仅展示用）与 `_RobustGroup.interact`（MDT 名单解析为空时该组退化单专家）——只兜崩溃、不加能力。


---

# 定版部分（SLC-MARS v20，frozen 2026-11-02）

> 本节以上为**供给侧与判定层的机制说明**（工具映射、契约、证据不变量、旁系缓冲、保形弃答、
> 图特征、baseline 对照组件）——它们描述的是"系统怎么搭"，与版本无关，仍然有效。
> 本节以下把 v11–v18 逐版实验编年压缩为**已否决方案表**，并给出 v20 定版决策流与复现命令。

## 十六、v20 定版决策流（当前默认配置）

v20 = **v16 基座 + R2 membership 必需性重排**。判定入口 `actions/decision_action.py`，
上游为三路并行供给（`answer_pipeline.supply`，同一 `asyncio.gather`）：

```
   ┌─ Kira（知识检索：Neo4j 三路证据 + text2cypher）┐
   ├─ Cael（计算：PPR / TIMER / KM / DESeq2 / 旁系冗余）┤ → 证据视图 EvidenceView
   ├─ L1（Prior 智能体：题面=纯 LLM 零证据，文献记忆先验）┤
   └─ L2（第二路纯 LLM：异构 persona + 温度 0.7 独立重答）┘
                          │
                          ▼
            Vega 融合层 `_fuse` / `_fuse_triad`
   ┌──────────────────────────────────────────────────────────────┐
   │ E == L1 同向      → 2-0 直通（agree 注记，零额外 LLM 调用）      │
   │ 冲突              → 交叉批评轮（L1/L2 互看论点修订终票）          │
   │                    → 三票 + 事实审料呈最终智能体 `adjudicate_multi` │
   │ 证据弃答          → 证据增强 LLM 主审（只给观测数值、不给结论）     │
   │ 法官缺席          → 确定性多数票；无多数 → prior 优先于证据通道     │
   │ 全弃权            → 维持诚实弃答（绝不臆造）                      │
   └──────────────────────────────────────────────────────────────┘
```

**pairwise 分支**：走 v10 的图特征 logistic 工件（`SLC_PAIRWISE_MODEL`，
`graph_features.PAIR_COLS` 12 项 SLC→Pathway→Disease 拆细特征），
按 argmax 出 winner；两候选皆无图证据 / 工件缺失 → 诚实回退"方向+量级+旁系"规则。

**membership 分支 = R2（v20 唯一增量）**：把判定构念从"促/抑极性方向"改为
"**CRISPR loss-of-function 必需性**"。自相矛盾的促/抑文献极性不再被当作定阴阳的证据；
改为五项证据重排（系数由 v16@n=53 membership 5-fold CV 拟合，样本外 acc 66.8% > 地板 60.4%）：

| 特征 | 含义 | 方向 | 权重 |
|---|---|---|---|
| `conflict` | 同基因既有促又有抑（极性矛盾） | → 必需 | +1.732673 |
| `dir_present` | 有任一向因果方向证据 | → 非必需 | −1.733301 |
| `solo` | 旁系孤儿（同亚家族无共表达伙伴） | → 必需 | +2.634866 |
| `buffered` | 旁系缓冲（同亚家族活跃共表达） | → 非必需 | −0.619402 |
| `mag = tanh(strength)` | 关联/差异表达量级 | → 必需 | +1.934390 |

`P(必需) = sigmoid(z − 0.408812)`，`>0.5` 判必需；**R2 不弃权**，只改 membership 判定，
pairwise 与门控关闭时的 v16 路径完全不受影响。溯源标注 `[R2 必需性重排·v20]`。

**保形与完整性**：prior / 辩论 / 表决 / R2 产生的答案**跳过保形过滤**
（保形阈值按证据通道 conf 语义校准，票决与先验 conf 套用无统计保证），并在
`decision_mode`/`basis`/`rationale` 里显式标注来源，绝不伪装成图谱/统计证据。

## 十七、环境变量旋钮（定版值见 `tools/slc_mars_env.ps1`）

> **凭据不进仓库**（2026-11-03 迁移）。`SLC_LLM_BASE_URL` / `SLC_LLM_API_KEY` / `SLC_LLM_MODEL`、
> `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`、`MONGO_URI` / `MONGO_DB_NAME` /
> `MONGO_COLLECTION_PAPERS` 统一由 `tools/_load_secrets.ps1::Import-SlcSecrets` 解析：
> **已导出到环境的变量优先**，否则 dot-source 仓库根那份未纳入版本管理的 `secrets.local.ps1`
> （模板 `secrets.example.ps1`；`.gitignore` 已挡住）。仓库 `config2.yaml` 只剩占位符，
> `llm_settings()` 会把没替换的占位符当"未配置"，宁可报错也不会把占位符当密钥发出去。
> 机器本地的 `$env:SLC_ROOT`（SLC_fresh 交接包根）也走同一个文件：它不是密钥，但曾以硬编码形式
> 躺在 `run_route_b.ps1` / `slc_mars_env.ps1` 里，带着个人目录名，2026-11-03 一并挪出被跟踪文件。
> MetaGPT 只肯读配置文件，但 `run_metagpt.py` 会把 `SLC_LLM_*` 就地覆盖进框架，
> 因此 `~/.metagpt/config2.yaml` 不需要写密钥——`tools/sync_metagpt_config.ps1` 生成无密钥版本。
> 历史上这些值曾明文躺在 `run_route_b.ps1` / `slc_mars_env.ps1` / `config2.yaml` 里，用过的密钥应视为已泄露并轮换。

| 变量 | 定版值 | 作用 |
|---|---|---|
| `SLC_TRACE_VERSION` | `20` | 判定/证据结构变更才 +1，会让旧 trace 缓存失效 |
| `SLC_PRIOR_AGENT` | `1` | Prior 智能体第三路证据通道 |
| `SLC_DEBATE` | `1` | 专家辩论（并行一票，仅确认不首席） |
| `SLC_LLM_CHIEF` | `1` | LLM 主审：知识/计算作审料，判定交 LLM 法官 |
| `SLC_TRIAD_VOTE` | `1` | 三通道表决（证据 ∥ 纯 LLM ×2） |
| `SLC_TRIAD_CRITIQUE` | `1` | 冲突态并行交叉批评（角色异构 persona） |
| `SLC_PRIOR_RACE` | `2` | prior 通道双发竞速，治端点长尾超时 |
| `SLC_MEM_R2` | `1` | **v20 增量**：membership 必需性证据重排 |
| `SLC_PAIRWISE_MODEL` | `outputs/traces/pairwise_graph_model.json` | pairwise 图特征 logistic 工件 |
| `SLC_PRIOR_ENSEMBLE` / `SLC_ANSWER_ENSEMBLE` | `1` | >1 复现 v13 / v14 集成（均阴性） |
| `SLC_PRIOR_RETRIEVE` | `0` | 1 复现 v17 L2 检索票源（未采纳） |
| `SLC_NEG_EVID` / `SLC_ABSTAIN_THETA` | `0` | 复现 v18 负证据 / 选择性弃权（阴性） |
| `SLC_CONFORMAL_FILE` | 空 | 指向保形阈值 json 即在途启用保形弃答 |
| `SLC_PARALOG_KSAT` | `4` | 旁系冗余饱和常数 |
| `SLC_DE_TOPN` | `400` | DESeq2 宽面板规模（决定旁系可见度） |

## 十八、已否决方案（阴性结果登记，代码保留可复现）

| 版本 | 尝试 | 结论 | 复现 |
|---|---|---|---|
| v13 | 记忆先验通道并行采样 N 次 + 众数聚合 | **阴性**：A 75.0→72.5 / B 68.4→63.2。集成中间通道把"该弃权"的难题强行凑成低共识硬答，agree 增量是"错误的共识" | `SLC_PRIOR_ENSEMBLE=3` |
| v14 | 端到端完整判定 N 次独立采样 + 答案层众数表决 | **阴性**：A 72.5 / B 57.9，heuristic 臂暴跌（B 63.2→26.3）。N 次抽样共享同一确定性证据 backbone，并非独立同分布推理样本 | `SLC_ANSWER_ENSEMBLE=3` |
| v15 | 三通道表决初版（证据 ∥ 纯 LLM ×2，第二票升温） | **机制阳性 / 运行阴性**：同题子集 +9.1pp，但每题 +2 次串行 LLM 调用致 17.5% 超时缺席 | `SLC_TRIAD_VOTE=1` + 串行版 |
| v16 | 表决层工程修复（L2 移入 supply 并行、角色 persona、交叉批评、双发竞速） | **阳性，成为基座**：缺席 0/40；n=100 扩测全体 70.0% > autogen 69.0 / mdagents 63.0 | 定版组成 |
| v17 | 表决第二票挂 Mongo 文献紧凑摘录（去同源化） | 未采纳（未见跨子集增益） | `SLC_PRIOR_RETRIEVE=1` |
| v18 | membership 负证据通道 + 置信选择性弃权 | **阴性 @n=100**：部分切片的领先是采样工件 | `SLC_NEG_EVID=1` / `SLC_ABSTAIN_THETA>0` |
| — | 文献 RAG（baseline 侧） | **有害**：堆原文使 pairwise 86.4%→54.5%，长上下文拖慢端点 | `run_lit_rag.py` |
| — | MDAgents 式全量串行专家辩论 | **零准确率增益、成本数量级**（单题 4–17min vs 纯 LLM 30s） | `run_mdagents.py` |

固化的两条规律：
1. **集成只在"独立完整推理路径 + 聚合点=决策点"时有效**；系统决策被确定性 backbone 主导时，
   多数表决会把正确答案一起平均掉（v13/v14 双证）。
2. **机制增益必须在全量样本上验标题口径**：n=40 的"净零"是量具不足——v12 的 75% 经 n=100 扩测
   证实为小样本运气（真值 47%），而 v16 的收益恰在"从诚实弃权里救回答案"这一 n=40 中不存在的场景。

## 十九、定版复现

```powershell
# 0) 环境与后端就绪（见第五节 5.1 + R/Neo4j/Mongo）
# 1) 管线三臂全量重算（n=489 clean 池，seed=5；TRACE_VERSION=20 命中缓存则秒回）
.\.venv\Scripts\python.exe -m knowledge_agents.compare_eval --n 500 --seed 5

# 2) 基线同题输入导出 + 五基线（各自 venv；均可断点续跑）
.\.venv\Scripts\python.exe -m knowledge_agents.baselines.export_inputs
.\.venv\Scripts\python.exe tools\run_baselines_all500.ps1

# 3) 防记忆题集 B 标注（双探针；GT 只在标注工具内读，绝不进答题路径）
.\.venv\Scripts\python.exe tools\label_antimemorization.py

# 4) 七系统主表 + TopK/消融终表 + 图
.\.venv\Scripts\python.exe -m knowledge_agents.baselines.merge_baselines   # -> outputs/traces/baselines_table_master.md
.\.venv\Scripts\python.exe tools\_v20_master_report.py                      # 三任务性能 & token
.\.venv\Scripts\python.exe tools\_v20_big_table.py                          # TopK 双候选池
.\.venv\Scripts\python.exe tools\_v20_main_figure.py                        # -> paper/figures/

# 5) 离线自检（免 DB / LLM / 工具）：tools/test_*.py 全跑，当前 12 个
Get-ChildItem tools\test_*.py | ForEach-Object {
    .\.venv\Scripts\python.exe -W error::SyntaxWarning $_
    if ($LASTEXITCODE) { "FAILED: $($_.Name)" }
}
#   其中 tools\test_no_secrets.py 是凭据守卫：源码/配置/文档里再出现明文密钥、口令或
#   带鉴权信息的 URI 就判失败，并确认本机 secrets.local.ps1 仍被 .gitignore 挡住。
```

## 二十、定版结果（v20，全量 clean 池）

数据集 A = 全部 489 题（membership 230 / pairwise 259）；
数据集 B = 防记忆子集 265 题（membership 138 / pairwise 127）。
精度口径为**全体**（未作答与答错均计错）；二选一任务的机会基线 = 50%。

| 系统 | A·Membership | A·Pairwise | B·Membership | B·Pairwise |
|---|---:|---:|---:|---:|
| pure-LLM | 56.1% | 70.7% | 27.5% | 40.2% |
| LitRAG | 50.9% | 64.1% | 31.2% | 49.6% |
| AutoGen | 62.6% | 68.3% | 43.5% | 48.0% |
| MDAgents | 33.9%† | 27.8%† | 34.1%† | 21.3%† |
| MetaGPT | 60.9% | 71.4% | 42.8% | 51.2% |
| **SLC-MARS heuristic** | **67.8%** | **73.7%** | **56.5%** | 52.0% |
| SLC-MARS auto | 67.4% | 73.0% | 52.2% | **52.8%** |

†MDAgents 为不完整快照（覆盖约 49%/51%），缺失按计错处理。

TopK 排序（11 道 CRISPR 排序题，NDCG@5 / hit@5）：图候选池上 SLC-MARS（Kira 逐基因）
**0.3838 / 0.909** 为最优；换到 LLM 自生成候选池后所有系统 NDCG@5 塌到 0.12–0.15，
说明**图谱候选池对排序任务是决定性的**。

智能体消融（三任务）：去掉 Kira 总体 73.0%→59.0%、去掉 Cael →65.0%，两智能体互补；
TopK 上"仅知识智能体"反而最强（0.3838）。

诚实性说明：管线覆盖率 100% 是 v20 定版行为（R2 不弃权、表决救回弃权题），
历史版本的保形弃权仍在 `conformal.py`/`_conformal_filter` 内可用（第十七节 `SLC_CONFORMAL_FILE`）。
