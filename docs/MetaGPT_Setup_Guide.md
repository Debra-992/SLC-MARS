# MetaGPT: structure analysis and usage guide

> The framework is **not vendored into this release**. Clone it yourself, as in README §6:
> `git clone --depth 1 https://github.com/FoundationAgents/MetaGPT.git` into `<PROJECT_ROOT>/MetaGPT`
> (master, shallow). Everything below describes upstream MetaGPT as that clone provides it, plus the two
> places where SLC-MARS touches it (§10).

---

## 1. What MetaGPT is

MetaGPT is a **multi-agent collaboration framework** built on one idea:

```
Code = SOP(Team)
```

i.e. the Standard Operating Procedures of a software company are materialized and applied to a team of LLMs.
You give it **one line of requirements**, and it returns the full set of software deliverables: user stories,
a competitive analysis, a requirements document, data structures, API design, code, documentation. Internally
it simulates a software company -- product manager, architect, project manager, engineer, QA engineer -- each
role doing its own job in a fixed sequence.

- Documentation: https://docs.deepwisdom.ai/ (the repository's own `docs/` tree carries the multilingual READMEs)
- Python version required by upstream: **>= 3.9 and < 3.12**. This project runs it under **3.11**.
- License: MIT

---

## 2. Repository layout

```
MetaGPT/
|-- metagpt/              # the core Python package (the framework itself)
|-- config/               # configuration examples (config2.example.yaml and friends)
|-- examples/             # official examples: DI, debate, AFlow, SPO, werewolf, ...
|-- docs/                 # documentation and the multilingual READMEs
|-- tests/                # unit tests
|-- requirements.txt      # dependency list
|-- setup.py              # install script (defines the `metagpt` CLI entry point)
`-- Dockerfile            # Docker deployment
```

### The core package, module by module

| module | responsibility |
|---|---|
| `roles/` | **Role layer.** The `Role` base class (name / profile / goal attributes plus a ReAct loop) and the built-in roles: `ProductManager`, `Architect`, `ProjectManager`, `Engineer`, `QAEngineer`, `Searcher`, `Researcher`, `Teacher`, ... ; `roles/di/` holds the Data Interpreter family (`DataInterpreter`, `RoleZero`, `SWEAgent`, `TeamLeader`) |
| `actions/` | **Action layer.** The `Action` base class plus 40+ concrete actions: `WritePRD`, `WriteDesign`, `WriteTasks`, `WriteCode`, `RunCode`, `CodeReview`, `PrepareDocuments`, ... Each Action owns its `run()` logic and its prompts |
| `team.py` / `software_company.py` | **Team / company layer.** `Team` hires roles (`hire`), injects a budget (`invest`) and runs the environment; `SoftwareCompany` wraps it and offers `generate_repo()` to produce a repository from one line |
| `environment/` | **Environment layer.** Message routing between roles -- an "instant-messaging group" model, where a role receives the messages it subscribed to |
| `memory/` + `context.py` | **Memory layer.** A role's short-term memory (the `Message` history) and long-term memory retrieval |
| `provider/` | **LLM provider layer.** One adapter for a dozen APIs: OpenAI, Azure, Anthropic, Ollama, Gemini, Bedrock, Zhipu, Spark, Qianfan, DashScope, OpenRouter, ... |
| `llm.py` | The `LLM` class, the direct way to call a model (`aask`, `acompletion`, ...) |
| `tools/` | **Tool layer.** Search engines (Bing / Google / DuckDuckGo / SerpApi), browsers (Playwright / Selenium), text-to-image, TTS, `tool_registry`, ... |
| `rag/` + `document_store/` | **Retrieval-augmentation layer.** FAISS / Chroma / Milvus vector stores, knowledge bases, document loading and retrieval |
| `exp_pool/` | **Experience pool.** Reuses solutions from past tasks (BM25 or Chroma retrieval plus an LLM re-rank) |
| `strategy/` | **Planning strategies.** `Planner`, ToT (Tree of Thoughts), ... |
| `skills/` | **Skills.** The skill-directory mechanism `RoleZero` can load |
| `ext/` | **Extended applications.** St. Stanford Town, werewolf, Android assistant, Chainlit UI, ... |
| `configs/` | The data models of the configuration (`Config`, `LLMConfig`, `EmbeddingConfig`, ...) |
| `base/` | Abstract base classes: `BaseRole`, `BaseEnv`, the serialization bases |
| `startup.py` | The CLI entry point (`metagpt "your requirement"`) |
| `schema.py` | Core data structures: `Message`, `Document`, `RoleState`, ... |

### The layers at a glance

```
        +-------------------------------------------+
        |  CLI / generate_repo() / your own code     |  entry layer
        +-------------------------------------------+
        |  Team / SoftwareCompany / Environment      |  team layer (message routing, budget control)
        +-------------------------------------------+
        |  Role (ProductManager -> Architect ->      |  role layer (the ReAct loop:
        |        Engineer -> QA ...)                  |  observe -> think -> act)
        +-------------------------------------------+
        |  Action (WritePRD / WriteCode / ...)       |  action layer (concrete prompts + logic)
        +-------------------------------------------+
        |  LLM Provider / Tools / RAG / Memory       |  capability layer (models, tools, retrieval, memory)
        +-------------------------------------------+
```

The data flow: a requirement is published into the environment as a `Message` -> roles that subscribed to it
receive it -> `Role.run()` thinks and executes an `Action` -> the action publishes its document or code back
into the environment -> the downstream role consumes it. PRD, then architecture, then task list, then code.

---

## 3. Installation and configuration

### 1. Install

```powershell
# Option A: the stable release from pip
pip install --upgrade metagpt

# Option B: install the clone this project keeps (development mode)
cd <PROJECT_ROOT>\MetaGPT
pip install --upgrade -e .
```

> This project installs it with **`--no-deps`** and pins the handful of packages MetaGPT actually imports
> here (README §6, ARCHITECTURE §5.1). That keeps the ~100 dependency set of the SLC tool environment out of
> the agent environment; if you add a MetaGPT feature and it fails on an import, that import is the missing
> pin.

> Before use, install [Node.js](https://nodejs.org/en/download) and [pnpm](https://pnpm.io/installation) --
> some features (mermaid rendering among them) depend on them.

### 2. Initialize the configuration

```powershell
metagpt --init-config   # writes ~\.metagpt\config2.yaml
```

Edit `~\.metagpt\config2.yaml` (the template is the repository's `config\config2.example.yaml`):

```yaml
llm:
  api_type: "openai"                        # azure / ollama / anthropic / gemini / zhipuai ... also work
  base_url: "https://api.openai.com/v1"     # a gateway or relay address may be substituted
  api_key: "YOUR_API_KEY"
  model: "gpt-4-turbo"

# Optional: give each role its own model, which is the usual way to control cost
roles:
  - role: "ProductManager"
    llm: {api_type: "openai", model: "gpt-4-turbo", api_key: "...", base_url: "..."}
  - role: "Engineer"
    llm: {api_type: "ollama", model: "qwen2:7b", base_url: "http://localhost:11434/v1"}

search:                          # only needed by the search-using roles
  api_type: "bing"
```

> Most regional model services (Qwen, Zhipu, Spark, DeepSeek, ...) speak the OpenAI protocol: change
> `base_url`, `api_key` and `model` and nothing else has to move.

### 3. The environment-variable route (the cheapest way to verify a setup)

```powershell
$env:OPENAI_API_KEY = "sk-xxx"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"   # optional
$env:LLM_MODEL = "gpt-4o"                            # optional
```

> In this repository the keys live in `secrets.local.ps1` and never in a tracked file; the baseline runner
> injects them into the framework at run time (§10), so `~\.metagpt\config2.yaml` may stay key-free.

---

## 4. Basic usage

### 1. CLI (one requirement in, one project out)

```powershell
metagpt "create a 2048 game"     # writes a complete repository under ./workspace
```

### 2. As a library (software-company mode)

```python
from metagpt.software_company import generate_repo

repo = generate_repo("create a 2048 game")
print(repo)  # the repository structure and every generated file
```

### 3. Calling the LLM directly

```python
import asyncio
from metagpt.llm import LLM

async def main():
    llm = LLM()
    ans = await llm.aask("explain multi-agent systems in one sentence", stream=False)
    print(ans)

asyncio.run(main())
```

### 4. Data Interpreter (the general-purpose task agent, the most used single role)

```python
import asyncio
from metagpt.roles.di.data_interpreter import DataInterpreter

async def main():
    di = DataInterpreter()
    # plans -> writes code -> executes -> reflects and corrects, until the task is done
    await di.run("visualize sklearn's Iris dataset and save one figure")

asyncio.run(main())
```

### 5. A ready-made single role

```python
import asyncio
from metagpt.roles import Searcher        # search, then summarize

async def main():
    searcher = Searcher()
    await searcher.run("which open-source LLM agent frameworks are mainstream?")

asyncio.run(main())
```

---

## 5. Advanced: a custom role and a multi-agent team

### 1. Custom Action + Role

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
        rsp = await self._aask(prompt)          # calls the LLM bound to this action
        return SimpleWriteCode.parse_code(rsp)  # extracts the code from ```python ... ```

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
        self.set_actions([SimpleWriteCode, SimpleRunCode])               # the action sequence
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)   # run them in order

    async def _act(self) -> Message:
        todo = self.rc.todo  # in BY_ORDER mode: SimpleWriteCode, then SimpleRunCode
        msg = self.get_memories(k=1)[0]
        result = await todo.run(msg.content)
        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        self.rc.memory.add(msg)
        return msg


asyncio.run(RunnableCoder().run("write a function that calculates the product of a list"))
```

The points that matter:

- an `Action` subclass declares `name` and implements `run()`, calls the model through `self._aask(prompt)`,
  and keeps its prompt in `PROMPT_TEMPLATE`;
- a `Role` subclass carries the `name` / `profile` / `goal` attributes, binds actions with `set_actions([...])`
  and overrides `_act()` to decide how memories and messages are consumed;
- there are two ReAct modes: `BY_ORDER` (walk the action list in order, a fully controlled flow) and
  `THINK_AND_ACT` (observe, think and pick an action each round -- the default, for open-ended tasks);
- when several roles collaborate, `watch()` declares which upstream messages a role cares about, filtered by
  `Message.role` / `cause_by`.

### 2. Assembling a multi-agent team

```python
import asyncio
from metagpt.team import Team
from metagpt.roles import ProductManager, Architect, Engineer, ProjectManager, QAEngineer

async def main():
    company = Team()
    company.hire([ProductManager(), Architect(), ProjectManager(), Engineer()])
    company.invest(10.0)          # budget ceiling in USD; exceeding it raises NoMoneyException
    company.run_stage(3)          # three rounds (about one round per role is enough to finish the flow)
    # the equivalent await form: await company.run(n_round=3, with_message="build a todo app")

asyncio.run(main())
```

### 3. Serialization for debugging (save and restore a role's state)

```python
from metagpt.roles import ProductManager

role = ProductManager()
role.dump("/tmp/pm.json")       # write the state out
role2 = ProductManager()
role2.load("/tmp/pm.json")      # read it back, which is how you debug a long run
```

---

## 6. Tour of the official examples (`examples/`)

| example | what it shows |
|---|---|
| `hello_world.py` | the most basic LLM calls (aask / aask_batch / acompletion / streaming) |
| `use_off_the_shelf_agent.py` | a single ready-made role |
| `build_customized_agent.py` | one custom Action + one custom Role -- the entry template, worth reading first |
| `build_customized_multi_agents.py` | custom multi-agent collaboration (the "goose and rabbit" story team) |
| `debate.py` / `debate_simple.py` | two-role debate (teacher versus student, RoleZero reading both memories) |
| `di/` | the Data Interpreter family: data analysis, stock scraping, PPT / PDF generation |
| `search_enhanced_qa.py` | search-augmented question answering |
| `invoice_ocr.py` | an invoice OCR assistant |
| `write_novel.py` / `write_tutorial.py` | writing a novel / writing a tutorial |
| `aflow/` | AFlow: generating agentic workflows automatically (ICLR 2025 code) |
| `spo/` / `sela/` | SPO prompt optimization, SELA automatic workflow search |
| `stanford_town/` | the generative-town multi-agent simulation |
| `werewolf_game/` | werewolf, a multi-agent game |
| `ui_with_chainlit/` | a quick web UI with Chainlit |

---

## 7. Configuration items worth knowing (config2.yaml)

```yaml
repair_llm_output: true     # when the LLM emits invalid JSON, ask the LLM to repair it
enable_llm_cost_control:    # turn on cost accounting
  cost_control_switch: true
redis: {...}                # message storage for multi-process / service deployment
s3: {...}                   # store artifacts in S3
exp_pool:                   # experience pool: reuse past task solutions, which raises the success rate
  enabled: true
  retrieval_type: chroma    # or bm25
role_zero:                  # memory settings for RoleZero / DI
  enable_longterm_memory: true
browser:
  engine: playwright        # the browser tool
search:
  api_type: google          # the search engine (google / bing / duckduckgo / serpapi)
```

---

## 8. Suggested order for learning it

1. **Run the smallest example**: `examples/hello_world.py`, which proves the configuration works.
2. **Understand the role mechanism**: read `examples/build_customized_agent.py` against the sources
   `metagpt/roles/role.py` and `metagpt/actions/action.py`.
3. **Understand collaboration**: `examples/build_customized_multi_agents.py` plus `metagpt/team.py` and
   `metagpt/environment/` (the subscription-based message routing).
4. **Use it**: run `metagpt "your requirement"` through the CLI to see the whole SOP produce its artifacts,
   or build a vertical task on `DataInterpreter`.
5. **Go deeper**: RAG (`metagpt/rag/`), the experience pool (`exp_pool`), AFlow / SPO automatic workflow
   optimization, and the simulations (Stanford Town, werewolf).

---

## 9. FAQ

- **API errors, connections reset?** Set the proxy in `llm.proxy` and in the top-level `proxy` of
  `config2.yaml`.
- **Too expensive?** Give the roles different models in the `roles:` section -- a strong model for the product
  manager, a cheap or local Ollama model for the engineer.
- **JSON parsing failures?** Keep `repair_llm_output: true`, which is the default.
- **A regional model service?** `api_type: "openai"` with a different `base_url` and `model` is enough;
  DeepSeek, Qwen and Kimi all speak the protocol.

---

## 10. Where SLC-MARS touches MetaGPT

MetaGPT appears in this project as **infrastructure for one baseline and as the runtime of the three
agents**, never as a dependency of the scoring path.

| touchpoint | what it does |
|---|---|
| `knowledge_agents/baselines/run_metagpt.py` | the **standalone** MetaGPT baseline: the framework's own Role/Action shell around a single `_aask`, answering from parametric knowledge only -- no tools, no retrieval, none of this pipeline's evidence. This is the row reported as `MetaGPT` in the released tables |
| `knowledge_agents/baseline_metagpt.py` | the **in-harness** diagnostic arm: the same parametric-only idea (`BaselineRole` + `BaselineAnswer`) evaluated inside `compare_eval`, which is the row `metaGPT-native(in-harness)` in `outputs/traces/baselines_table_master.md`. Its abstentions are failed or unparseable calls under the harness's short per-call budget rather than refusals, it never receives the pipeline's evidence view, and it is *not* the citable MetaGPT baseline |
| `knowledge_agents/roles/*`, `actions/*` | the three SLC-MARS agents are MetaGPT `Role` / `Action` subclasses, so `Team`-style message passing (`cause_by`, `watch`) is what carries Kira and Cael into Vega |
| `tools/sync_metagpt_config.ps1` | writes `~/.metagpt/config2.yaml` **with no credential in it**. MetaGPT only reads configuration from a file, so `run_metagpt.py` overlays `SLC_LLM_*` onto the framework in-process at run time; the key never has to be written down |

Two consequences of installing it `--no-deps`: the endpoint libraries actually used (the OpenAI-compatible
client, `pymongo`, `tenacity`) have to be present in `.venv` from the pinned list, and a MetaGPT feature you
add may fail on a missing transitive dependency -- an import error there is a missing pin, not a MetaGPT bug.
