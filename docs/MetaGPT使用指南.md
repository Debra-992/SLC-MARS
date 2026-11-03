# MetaGPT 结构分析与使用指南

> 仓库已拉取至 `<PROJECT_ROOT>\MetaGPT`（来源：https://github.com/FoundationAgents/MetaGPT，浅克隆 master 分支）

---

## 一、MetaGPT 是什么

MetaGPT 是一个 **多智能体（Multi-Agent）协作框架**，核心理念为：

```
Code = SOP(Team)
```

即把软件公司的标准作业流程（SOP，Standard Operating Procedures）物化并应用到由 LLM 组成的团队上。它**输入一行需求**，输出**用户故事、竞品分析、需求文档、数据结构、API 设计、代码、文档**等完整软件交付物。内部模拟了一个软件公司：产品经理 → 架构师 → 项目经理 → 工程师 → QA 工程师，各司其职、按流程协作。

- 📄 官方文档：https://docs.deepwisdom.ai/main/zh/
- 🐍 Python 版本要求：**≥ 3.9 且 < 3.12**
- ⚖️ 协议：MIT

---

## 二、仓库整体结构

```
MetaGPT/
├── metagpt/              # 核心 Python 包（框架本体）
├── config/               # 配置示例（config2.example.yaml 等）
├── examples/             # 官方示例（DI、辩论、AFlow、SPO、狼人杀等）
├── docs/                 # 文档与多语言 README（含中文版 README_CN.md）
├── tests/                # 单元测试
├── requirements.txt      # 依赖清单
├── setup.py              # 安装脚本（定义 metagpt CLI 入口）
└── Dockerfile            # Docker 部署
```

### 核心包 `metagpt/` 模块详解

| 模块 | 职责 |
|---|---|
| `roles/` | **角色层**。`Role` 基类（name/profile/goal 等属性 + ReAct 循环），内置角色：`ProductManager`、`Architect`、`ProjectManager`、`Engineer`、`QAEngineer`、`Searcher`、`Researcher`、`Teacher` 等；`roles/di/` 为 Data Interpreter 系列（`DataInterpreter`、`RoleZero`、`SWEAgent`、`TeamLeader`） |
| `actions/` | **动作层**。`Action` 基类 + 40+ 具体动作：`WritePRD`（写需求文档）、`WriteDesign`（架构设计）、`WriteTasks`（任务拆分）、`WriteCode`（写代码）、`RunCode`（执行代码）、`CodeReview`、`PrepareDocuments` 等。每个 Action 定义 `run()` 逻辑与提示词 |
| `team.py` / `software_company.py` | **团队/公司层**。`Team` 负责雇角色（`hire`）、注入预算（`invest`）、运行环境；`SoftwareCompany` 是其上层封装，提供 `generate_repo()` 一行生成代码仓库 |
| `environment/` | **环境层**。多角色间的消息路由（"微信群"式即时通讯），角色通过订阅（subscription）接收感兴趣的消息 |
| `memory/` + `context.py` | **记忆层**。角色的短期记忆（Message 历史）与长期记忆检索 |
| `provider/` | **LLM 提供方层**。统一适配 OpenAI、Azure、Anthropic、Ollama、Gemini、Bedrock、智谱、星火、千帆、DashScope、OpenRouter 等十余种 API |
| `llm.py` | `LLM` 类，直接调模型的入口（`aask` / `acompletion` 等） |
| `tools/` | **工具层**。搜索引擎（Bing/Google/DuckDuckGo/SerpApi）、浏览器（Playwright/Selenium）、文生图、TTS、`tool_registry` 工具注册表等 |
| `rag/` + `document_store/` | **检索增强层**。FAISS/Chroma/Milvus 向量库、知识库、文档加载与检索 |
| `exp_pool/` | **经验池**。复用历史任务经验（BM25 或 Chroma 检索 + LLM 重排） |
| `strategy/` | **规划策略**。`Planner`、ToT（Tree of Thoughts）等 |
| `skills/` | **技能**。RoleZero 可加载的技能目录机制 |
| `ext/` | **扩展应用**。St. Stanford Town（小镇模拟）、狼人杀、安卓助手、Chainlit UI 等 |
| `configs/` | 配置的数据模型定义（`Config`、`LLMConfig`、`EmbeddingConfig` 等） |
| `base/` | 抽象基类：`BaseRole`、`BaseEnv`、序列化基类 |
| `startup.py` | CLI 入口（`metagpt "你的需求"`） |
| `schema.py` | 核心数据结构：`Message`、`Document`、`RoleState` 等 |

### 架构分层示意

```
        ┌─────────────────────────────────────────┐
        │  CLI / generate_repo() / 自定义代码       │  入口层
        ├─────────────────────────────────────────┤
        │  Team / SoftwareCompany / Environment    │  团队层（消息路由、预算控制）
        ├─────────────────────────────────────────┤
        │  Role（ProductManager → Architect →      │  角色层（ReAct 循环：
        │       Engineer → QA ...）                │  observe→think→act）
        ├─────────────────────────────────────────┤
        │  Action（WritePRD / WriteCode / ...）    │  动作层（具体提示词+逻辑）
        ├─────────────────────────────────────────┤
        │  LLM Provider / Tools / RAG / Memory     │  能力层（模型、工具、检索、记忆）
        └─────────────────────────────────────────┘
```

数据流：需求作为 `Message` 发布到环境 → 角色按订阅关系收到消息 → `Role.run()` 思考并执行 `Action` → Action 产出文档/代码再发布回环境 → 下游角色消费，逐步产出 PRD → 架构设计 → 任务列表 → 代码。

---

## 三、安装与配置

### 1. 安装

```powershell
# 方式 A：pip 稳定版（推荐）
pip install --upgrade metagpt

# 方式 B：从本地拉取的源码安装（开发模式）
cd "<PROJECT_ROOT>\MetaGPT"
pip install --upgrade -e .
```

> 使用前请先安装 [Node.js](https://nodejs.org/en/download) 和 [pnpm](https://pnpm.io/installation#using-npm)（部分功能如 mermaid 渲染依赖）。

### 2. 初始化配置

```powershell
metagpt --init-config   # 生成 ~\.metagpt\config2.yaml
```

编辑 `~\.metagpt\config2.yaml`（模板见仓库内 `config\config2.example.yaml`）：

```yaml
llm:
  api_type: "openai"            # 也支持 azure / ollama / anthropic / gemini / zhipuai 等
  base_url: "https://api.openai.com/v1"   # 可填中转地址
  api_key: "YOUR_API_KEY"
  model: "gpt-4-turbo"

# 可选：为不同角色单独指定模型（省钱的常用做法）
roles:
  - role: "ProductManager"
    llm: {api_type: "openai", model: "gpt-4-turbo", api_key: "...", base_url: "..."}
  - role: "Engineer"
    llm: {api_type: "ollama", model: "qwen2:7b", base_url: "http://localhost:11434/v1"}

search:                          # 使用搜索类角色时配置
  api_type: "bing"
```

> 国内模型（通义、智谱、星火、DeepSeek 等）大多兼容 openai 协议，直接改 `base_url` + `api_key` + `model` 即可。

### 3. 环境变量方式（最低成本验证）

```powershell
$env:OPENAI_API_KEY = "sk-xxx"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"   # 可选
$env:LLM_MODEL = "gpt-4o"                            # 可选
```

---

## 四、基本使用

### 1. CLI（一行需求生成项目）

```powershell
metagpt "创建一个 2048 小游戏"     # 自动在 ./workspace 下生成完整代码仓库
```

### 2. 库调用（软件公司模式）

```python
from metagpt.software_company import generate_repo

repo = generate_repo("创建一个 2048 小游戏")
print(repo)  # 输出仓库结构与所有文件
```

### 3. 直接调用 LLM

```python
import asyncio
from metagpt.llm import LLM

async def main():
    llm = LLM()
    ans = await llm.aask("用一句话解释什么是多智能体系统", stream=False)
    print(ans)

asyncio.run(main())
```

### 4. Data Interpreter（通用任务执行智能体，最常用的单角色）

```python
import asyncio
from metagpt.roles.di.data_interpreter import DataInterpreter

async def main():
    di = DataInterpreter()
    # 自动规划 → 写代码 → 执行 → 反思修正，直至完成
    await di.run("对 sklearn 的 Iris 数据集做可视化分析，并输出一张图")

asyncio.run(main())
```

### 5. 现成单角色

```python
import asyncio
from metagpt.roles import Searcher        # 搜索+总结

async def main():
    searcher = Searcher()
    await searcher.run("2025 年主流开源 LLM Agent 框架有哪些？")

asyncio.run(main())
```

---

## 五、进阶：自定义角色与多智能体团队

### 1. 自定义 Action + Role

```python
import asyncio
import re
import subprocess

from metagpt.actions import Action
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message


class SimpleWriteCode(Action):
    PROMPT_TEMPLATE: str = """
    Write a python function that can {instruction} and provide two runnable test cases.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """
    name: str = "SimpleWriteCode"

    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
        rsp = await self._aask(prompt)          # 调用绑定的 LLM
        return SimpleWriteCode.parse_code(rsp)  # 提取 ```python ... ``` 中的代码

    @staticmethod
    def parse_code(rsp):
        match = re.search(r"```python(.*)```", rsp, re.DOTALL)
        return match.group(1) if match else rsp


class SimpleRunCode(Action):
    name: str = "SimpleRunCode"

    async def run(self, code_text: str):
        result = subprocess.run(["python", "-c", code_text], capture_output=True, text=True)
        return result.stdout


class RunnableCoder(Role):
    name: str = "Alice"
    profile: str = "RunnableCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleWriteCode, SimpleRunCode])               # 绑定动作序列
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)   # 按顺序执行动作

    async def _act(self) -> Message:
        todo = self.rc.todo  # BY_ORDER 模式下依次为 SimpleWriteCode → SimpleRunCode
        msg = self.get_memories(k=1)[0]
        result = await todo.run(msg.content)
        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        self.rc.memory.add(msg)
        return msg


asyncio.run(RunnableCoder().run("write a function that calculates the product of a list"))
```

关键点：
- `Action` 子类：声明 `name` + 实现 `run()`，用 `self._aask(prompt)` 调 LLM，`PROMPT_TEMPLATE` 承载提示词；
- `Role` 子类：`name/profile/goal` 属性 + `set_actions([...])` 绑定动作 + 重写 `_act()` 决定如何消费记忆和消息；
- ReAct 模式两种：`BY_ORDER`（按顺序跑完动作，流程可控）与 `THINK_AND_ACT`（每轮 observe→think→选动作，适合复杂任务，默认）；
- 多角色协作时用 `watch()` 声明角色关心哪些上游消息（按 `Message.role` / `cause_by` 过滤）。

### 2. 组建多智能体团队

```python
import asyncio
from metagpt.team import Team
from metagpt.roles import ProductManager, Architect, Engineer, ProjectManager, QAEngineer

async def main():
    company = Team()
    company.hire([ProductManager(), Architect(), ProjectManager(), Engineer()])
    company.invest(10.0)          # 预算上限（美元），超限抛 NoMoneyException
    company.run_stage(3)          # 运行 3 轮（轮数 ≈ 角色数 即可让流程走完）
    # 等价异步写法: await company.run(n_round=3, with_message="开发一个todo应用")

asyncio.run(main())
```

### 3. 序列化调试（保存/恢复角色状态）

```python
from metagpt.roles import ProductManager

role = ProductManager()
role.dump("/tmp/pm.json")       # 落盘
role2 = ProductManager()
role2.load("/tmp/pm.json")      # 恢复，用于调试长流程
```

---

## 六、官方示例导览（`examples/` 目录）

| 示例 | 说明 |
|---|---|
| `hello_world.py` | LLM 最基础调用（aask / aask_batch / acompletion / 流式） |
| `use_off_the_shelf_agent.py` | 开箱即用单角色 |
| `build_customized_agent.py` | 自定义单 Action + 单 Role（入门模板）⭐ |
| `build_customized_multi_agents.py` | 自定义多智能体（如 "鹅兔传" 写故事团队）⭐ |
| `debate.py` / `debate_simple.py` | 双角色辩论（师生辩论，RoleZero 互看记忆） |
| `di/` | Data Interpreter 系列：数据分析、爬股票、生成 PPT/ PDF 等 ⭐ |
| `search_enhanced_qa.py` | 搜索增强问答 |
| `invoice_ocr.py` | 发票 OCR 助手 |
| `write_novel.py` / `write_tutorial.py` | 写小说 / 写教程 |
| `aflow/` | AFlow：自动生成 Agentic Workflow（ICLR 2025 论文代码） |
| `spo/` / `sela/` | SPO 提示优化、SELA 自动搜索工作流 |
| `stanford_town/` | 生成式小镇多智能体模拟 |
| `werewolf_game/` | 狼人杀多智能体博弈 |
| `ui_with_chainlit/` | 用 Chainlit 快速搭 Web UI |

---

## 七、常用配置项速查（config2.yaml）

```yaml
repair_llm_output: true     # LLM 输出非法 JSON 时自动用 LLM 修复
enable_llm_cost_control:    # 开启成本统计
  cost_control_switch: true
redis: {...}                # 多进程/服务化部署时的消息存储
s3: {...}                   # 产物存储到 S3
exp_pool:                   # 经验池（复用历史任务经验，显著提升成功率）
  enabled: true
  retrieval_type: chroma    # 或 bm25
role_zero:                  # RoleZero/DI 的记忆配置
  enable_longterm_memory: true
browser:
  engine: playwright        # 浏览器工具
search:
  api_type: google          # 搜索引擎（google/bing/duckduckgo/serpapi）
```

---

## 八、学习路径建议

1. **跑通最小示例**：`examples/hello_world.py` → 验证配置正确。
2. **理解角色机制**：精读 `examples/build_customized_agent.py`，对照 `metagpt/roles/role.py` 与 `metagpt/actions/action.py` 源码。
3. **理解多智能体协作**：`examples/build_customized_multi_agents.py` + `metagpt/team.py`、`metagpt/environment/`（消息订阅路由）。
4. **实战**：CLI 跑 `metagpt "你的需求"` 看完整 SOP 产物；或基于 `DataInterpreter` 做垂直任务。
5. **深入方向**：RAG（`metagpt/rag/`）、经验池（`exp_pool`）、AFlow/SPO 自动工作流优化、Stanford Town/狼人杀等模拟场景。
6. **中文文档**：`docs/README_CN.md` 及 https://docs.deepwisdom.ai/main/zh/

## 九、FAQ 速答

- **API 报错/连接被重置？** 在 `config2.yaml` 的 `llm.proxy` 和顶层 `proxy` 里配置代理。
- **费用太高？** 用 `roles:` 段为不同角色配不同模型（如 PM 用 GPT-4，Engineer 用便宜模型/本地 Ollama）。
- **输出 JSON 解析失败？** 确保 `repair_llm_output: true`（默认开启）。
- **想接国产模型？** `api_type: "openai"` + 改 `base_url`/`model` 即可（DeepSeek、Qwen、Kimi 均兼容）。
