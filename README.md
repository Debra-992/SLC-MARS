# SLC-MARS

**Solute Carrier Multi-Agent Retrieval-and-Scoring System** — a three-agent pipeline that answers
CRISPR essentiality questions about the SLC (solute carrier) gene family from *retrievable, auditable
scientific evidence* rather than from a language model's parametric memory.

> **Frozen release: v20** (2026-11-02).
> Default trace key `SLC_TRACE_VERSION=20`; reproduction entry point [`tools/slc_mars_env.ps1`](tools/slc_mars_env.ps1).
> Full design document: [`knowledge_agents/ARCHITECTURE.md`](knowledge_agents/ARCHITECTURE.md).
> Paper sources (`paper/`: LaTeX, PDFs, figures) are **not part of this code release** — every
> `paper/…` link below points at the companion manuscript, and every number it cites is
> reproduced by §7 from the frozen traces that *are* included here.

---

## 1. Abstract

Given a question of the form *"is SLCx essential in cancer c under screen s?"* (membership), or
*"which of SLCa / SLCb is more essential?"* (pairwise), SLC-MARS assembles evidence from a knowledge
graph, a literature database, and four independent computational pipelines, and returns a decision
together with the evidence that produced it.

The system is built on three cooperating agents on Role/Action layering:

| Agent | Role | Supplies |
|---|---|---|
| **Kira** | knowledge-retrieval agent | Neo4j three-path evidence, Mongo literature, LLM-generated read-only Cypher |
| **Cael** | computing agent | PPR graph proximity, TIMER pan-cancer expression, KM survival, DESeq2 differential expression, paralog redundancy |
| **Vega** | decision agent | consumes the assembled evidence view only — **never re-fetches** |

The two supply sides are independent and run in parallel; everything downstream depends only on the
typed contract in [`knowledge_agents/schema.py`](knowledge_agents/schema.py).

Headline result on the 489-question benchmark: **73.7 %** pairwise and **67.8 %** membership overall
accuracy (binary tasks, chance = 50 %), and the best TopK ranker on the graph candidate pool
(NDCG@5 = 0.3838, hit@5 = 0.909). On the anti-memorization subset (Dataset B, n = 265) SLC-MARS is
the only system that remains above chance on membership (56.5 % vs. 27.5 % for a pure LLM).

---

## 2. Task and benchmark

### 2.1 Question pool

The benchmark is derived from real MAGeCK/CRISPR screens on SLC genes, packaged as 647 formal
questions (`data/dataset-manifest.json`, protocol `formal-647-v1`, with SHA-256 pins on the source,
question and ground-truth files).

Scientific-validity screening removes 147 entries before evaluation. As recorded per item in
`data/questions_removed.jsonl`, their `treatment` field names a knockout of a *different* SLC gene
than the one the question asks about (e.g. treatment `SLC16A1.KO`, question about `SLC7A5`), so the
screen result cannot adjudicate the queried gene. All 147 fall in the coarse `Colorectal cancer`
label; the two retained entries of that label are topK items, where the perturbed gene is the screen
reference rather than the object of the claim.

| pool | membership | pairwise | topK | total |
|---|---:|---:|---:|---:|
| raw (`data/questions.jsonl`) | 288 | 348 | 11 | **647** |
| clean (`data/questions_clean.jsonl`) | 230 | 259 | 11 | **500** |
| **Dataset A** — primary benchmark | 230 | 259 | — | **489** |

17 cancer labels survive cleaning. Ground truth: `evaluation/ground_truth_clean.jsonl`.

### 2.2 Dataset B — the anti-memorization subset (n = 265)

An early audit found that a pure LLM with *zero* retrieval and *zero* tools scored 86.4 % on pairwise,
well above the graph pipeline. The parsimonious explanation is recall, not reasoning: DepMap/CRISPR
screen results are inside pretraining corpora. Evaluating on the full pool therefore conflates
inference with memory.

Dataset B isolates questions where memory is demonstrably absent or fragile, using a **dual-probe**
procedure (`tools/label_antimemorization.py`, construction details in
[`paper/dataset_b_construction.tex`](paper/dataset_b_construction.tex)):

- **Probe 1** — original wording, reused from the pure-LLM baseline answers.
- **Probe 2** — a paraphrase that strips the original interrogative frame (pairwise is re-cast as a
  fill-in-the-blank assertion about what drives the observed differential essentiality; membership is
  re-cast as a claim-audit), preserving semantics while breaking verbatim memory cues.

`B := novel ∪ unstable ∪ probe_error`, i.e. anything the model does not get right on *both* probes.
Ground-truth labels are read only inside the labeling tool and never enter any answering path.

**Interpretation caveat.** B's definition references pure-LLM probe 1, so `pure_llm` scoring low on B
is a property of the construction, not an independent finding. State this whenever the B column for
`pure_llm` is cited.

### 2.3 Metric

Unless stated otherwise, accuracy is the **overall** measure: an abstention and a missing answer both
count as wrong, over a fixed denominator. This is deliberate — calibrated refusal is part of the
system's claim, and an answered-only metric would reward systems that refuse hard questions.
Both tasks are binary, so chance = 50 %.

---

## 3. Architecture

```
                     downstream consumer (watches both cause_by messages)
                                  ▲                 ▲
              BackgroundKnowledge            ComputedKnowledge
               (knowledge_type=                (knowledge_type=
                 background_kg)                  candidate_data)
                     │                 │
        ┌────────────┴───┐       ┌─────┴──────────────┐
        │ KnowledgeQuery │       │    ComputeRole      │
        │   Role (Kira)  │       │  (Cael, routes 4    │
        │                │       │   computation kinds)│
        └───────┬────────┘       └─────┬──────────────┘
   KnowledgeRetrieval Action        Computation Action
        │                                │
   RetrievalTools                   ComputationTools
  (project_knowledge /              (graph_computation /
   kg_schema / neighbors / cypher)   timer / km / deseq2)
        └──────────────┬───────────────┘
                slc_backend.Backend
        ┌──────────────┴───────────────┐
   RealMcpBackend                 UnavailableBackend
   (mcp.ClientSession.call_tool   (offline fallback: uniformly
    against 4 stdio MCP servers)   "unknown", verified not to fabricate)
```

Kira and Cael wrap the **8 real tools** of the SLC `multi_agent` handoff bundle over MCP stdio; there
is no mock data layer in the graded path. Vega fuses three equal-weight channels — evidence magnitude,
causal direction, and paralog buffering — into a decision.

### 3.1 SLC-MARS


Vega's fusion layer (`_fuse` / `_fuse_triad`) then applies:

| situation | action |
|---|---|
| evidence channel agrees with L1 | 2-0 direct pass-through, zero extra LLM calls |
| channels conflict | cross-critique round (L1/L2 see each other's arguments and re-vote) → triad vote → final agent `adjudicate_multi` |
| evidence channel abstains | evidence-augmented LLM chief (observed values only, no conclusions handed over) |
| chief unavailable | deterministic majority vote; no majority → prior outranks the evidence channel |
| every path abstains | honest abstention is maintained — never fabricated |

**Pairwise branch.** A trained graph-feature logistic artifact (`SLC_PAIRWISE_MODEL`, features defined
once in [`knowledge_agents/graph_features.py`](knowledge_agents/graph_features.py) and shared by
calibration dump, offline fitting, serialization and online inference — no train/serve skew). Online
scoring needs no scikit-learn: `(x−μ)/σ → coef·z + b → sigmoid`, argmax picks the winner. If the
artifact is absent or neither candidate carries graph path evidence, the code falls back to the
transparent direction + magnitude + buffering rule rather than emitting a fabricated model ranking.

**Membership branch — the one v20 delta (R2).** The decision construct is restated from
"promotion/suppression polarity" to "CRISPR loss-of-function essentiality". Contradictory
promotion *and* suppression literature for the same gene is no longer read as directional evidence;
it is read as an essentiality signal. Five features are re-weighted
(`_membership_r2` in [`knowledge_agents/actions/decision_action.py`](knowledge_agents/actions/decision_action.py)),
coefficients fitted by 5-fold CV on v16 @ n=53 membership, out-of-sample accuracy 66.8 % over a 60.4 %
floor:

| feature | meaning | points to | weight |
|---|---|---|---:|
| `conflict` | same gene both promoted and suppressed in the literature | essential | +1.732673 |
| `dir_present` | any single-direction causal evidence | non-essential | −1.733301 |
| `solo` | paralog orphan (no co-expressed subfamily partner) | essential | +2.634866 |
| `buffered` | paralog buffering (active subfamily co-expression) | non-essential | −0.619402 |
| `mag = tanh(strength)` | association / differential-expression magnitude | essential | +1.934390 |

`P(essential) = σ(z − 0.408812)`; `> 0.5` ⇒ essential. R2 never abstains and never touches the
pairwise branch or the v16 path taken when the gate is off. Provenance marker `[R2 必需性重排·v20]`
is written into the rationale.

### 3.2 Evidence-integrity invariants

These are enforced by the code and are the reason the pipeline's numbers mean anything:

1. **Failure, unsupported, and no-data are all *unknown*, never negative.** They map to
   `ToolStatus.{unavailable, download_required, unsupported_cancer, no_seed, rejected, error}` — never
   to a silent `ok`.
2. **No fabricated values.** TIMER `plot_only` returns no invented number; KM without an HR reports
   none; DESeq2 without counts returns `download_required`.
3. **Read-only boundary.** text2cypher output passes
   [`knowledge_agents/tools/cypher_guard.py`](knowledge_agents/tools/cypher_guard.py) (mirrors the real
   validator: writes, procedures and multi-statement payloads are `rejected`).
4. **Source independence is declared, not assumed.** `source_independence=NOT_ESTABLISHED` and
   `public_web_used=False` are propagated verbatim.
5. **Routing is deterministic, not model-chosen.** Computation tools are triggered by explicit rules;
   unrouted tools are recorded in `skipped_tools` and are not treated as negative evidence.
6. **Prior / debate / vote / R2 answers skip conformal filtering.** Conformal thresholds are
   calibrated on the evidence channel's confidence semantics; applying them to vote or prior
   confidences would carry no statistical guarantee. Their origin is labelled in
   `decision_mode`/`basis`/`rationale` and never dressed up as graph or statistical evidence.

Calibrated abstention is still available ([`knowledge_agents/conformal.py`](knowledge_agents/conformal.py),
Conformal Risk Control, Bates et al. 2021) — mount a threshold file with `SLC_CONFORMAL_FILE`.
Note that v20's 100 % coverage is a *design outcome* (R2 does not abstain; the triad vote recovers
former abstentions), not a coverage inflation.

---

## 4. Results (frozen v20, full clean pool)

Accuracy is the overall measure; chance = 50 %. Δchance is in percentage points.
Regenerate with §7; authoritative LaTeX in [`paper/results_tables.tex`](paper/results_tables.tex).

### 4.1 Main decision tasks

| System | A·Membership | Δ | A·Pairwise | Δ | B·Membership | Δ | B·Pairwise | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *Baselines* | | | | | | | | |
| `pure_llm` | 56.1 % | +6.1 | 70.7 % | +20.7 | 27.5 % | −22.5 | 40.2 % | −9.8 |
| LitRAG | 50.9 % | +0.9 | 64.1 % | +14.1 | 31.2 % | −18.8 | 49.6 % | −0.4 |
| AutoGen | 62.6 % | +12.6 | 68.3 % | +18.3 | 43.5 % | −6.5 | 48.0 % | −2.0 |
| MDAgents † | 33.9 % | −16.1 | 27.8 % | −22.2 | 34.1 % | −15.9 | 21.3 % | −28.7 |
| MetaGPT | 60.9 % | +10.9 | 71.4 % | +21.4 | 42.8 % | −7.2 | 51.2 % | +1.2 |
| *SLC-MARS pipeline* | | | | | | | | |
| **SLC-MARS heuristic** | **67.8 %** | +17.8 | **73.7 %** | +23.7 | **56.5 %** | +6.5 | 52.0 % | +2.0 |
| SLC-MARS auto | 67.4 % | +17.4 | 73.0 % | +23.0 | 52.2 % | +2.2 | **52.8 %** | +2.8 |

† MDAgents is an incomplete snapshot (≈49 % coverage on A, ≈51 % on B); missing answers count as wrong.

A · overall (n = 489): SLC-MARS heuristic **71.0 %**, auto 70.3 %, MetaGPT 66.5 %, AutoGen 65.6 %,
pure-LLM 63.8 %, MDAgents 59.3 %, LitRAG 57.9 %. B · overall (n = 265): heuristic 54.3 %.

A diagnostic eighth row, **`metaGPT-native(shared-evidence)`**, is reported separately in
`baselines_table_master.md`: it hands native MetaGPT *this pipeline's own assembled evidence*
(`tracestore.systems.baseline`) instead of letting it answer standalone. It scores 35.4 % overall on A
— 48.1 % coverage, abstaining on half the pool. Handing raw multi-channel evidence to an uncalibrated
agent is *worse* than letting it use its prior, which is what Vega's fusion layer exists to prevent.
Do not cite that row as "MetaGPT"; the paper's MetaGPT baseline is the standalone one.

The decisive reading is the **A → B** contrast. Every pure-memory system collapses on B
(pure-LLM membership 56.1 % → 27.5 %, i.e. *below* chance); SLC-MARS degrades mildly and stays the
only system above chance on B·membership. That is the evidence that the pipeline is consuming
retrievable information the baselines do not have.

### 4.2 TopK ranking (11 CRISPR ranking questions, separate task, no A/B split)

Every system is scored on two candidate pools: the **graph pool** (PPR top-15 ∪ ground truth) and the
**LLM self-generated pool** (whatever the model nominates).

| System | graph NDCG@5 | graph hit@5 | LLM-pool NDCG@5 | LLM-pool hit@5 |
|---|---:|---:|---:|---:|
| `pure_llm` | 0.3587 | 0.848 | 0.1244 | 0.727 |
| MetaGPT | 0.3357 | 0.818 | 0.1353 | **0.909** |
| AutoGen | 0.3553 | 0.818 | 0.1219 | 0.818 |
| MDAgents ‡ | 0.1972 | 0.273 | 0.1466 | 0.600 |
| LitRAG | 0.3086 | 0.667 | 0.1173 | 0.606 |
| **SLC-MARS w/o computing agents** | **0.3838** | **0.909** | **0.1521** | 0.818 |
| SLC-MARS full | 0.3704 | 0.818 | 0.1315 | 0.636 |
| SLC-MARS w/o knowledge agents | 0.3631 | 0.818 | 0.1349 | 0.545 |
| SLC-MARS graph only  | 0.3460 | 0.818 | 0.1187 | 0.636 |

‡ averages 10 of 11 questions; one ended in a transient API timeout.

Switching from the graph pool to the self-generated pool drops NDCG@5 by ≈0.23 for *every* system.
Candidate-pool construction, not the ranker, is the bottleneck for this task — and the graph pool is
supplied by Kira.

### 4.3 Agent ablation (three tasks)

| Configuration | Membership | Pairwise | All tasks | TopK NDCG@5 | TopK hit@5 |
|---|---:|---:|---:|---:|---:|
| SLC-MARS full | 69.8 % | 76.6 % | **73.0 %** | 0.3704 | 0.818 |
| without Kira (knowledge) | 56.6 % (−13.2) | 61.7 % (−14.9) | 59.0 % (−14.0) | 0.3631 | 0.818 |
| without Cael (computing) | 67.9 % (−1.9) | 61.7 % (−14.9) | 65.0 % (−8.0) | **0.3838** | **0.909** |

The two supply agents are complementary on the decision tasks; on ranking, knowledge-only is the
strongest single variant — per-gene KG retrieval beats adding a numeric channel to the pool.

Component/pool ablation (TopK, per-row deltas against "Cael graph only") is in
[`paper/results_tables.tex`](paper/results_tables.tex), Table 3: PPR refinement +0.0171, family
feature +0.0038 (negligible), larger candidate pools *hurt* (cap=32 −0.0223, cap=48 −0.0209,
all-features cap=16 −0.0422).

---

## 5. Repository layout

```
SLC-MARS/
├── run_route_b.ps1              Route-B launcher: env wiring + mode dispatch (demo|question|answer|eval|baseline|compare|calib|prewarm)
├── secrets.example.ps1          Credential template — copy to secrets.local.ps1 and fill in
├── secrets.local.ps1            YOURS, untracked (.gitignore): the only file holding real keys
├── config2.yaml                 LLM endpoint template (placeholders only; values come from env)
├── knowledge_agents/            ── the system ──
│   ├── schema.py                typed contract: ToolStatus / BackgroundKnowledge / ComputedKnowledge
│   ├── roles/                   KnowledgeQueryRole (Kira), ComputeRole (Cael), DecisionRole (Vega)
│   ├── actions/                 retrieval, computation, decision (incl. R2 + triad fusion)
│   ├── tools/                   MCP backend, retrieval/computation adapters, cypher_guard, paralog
│   ├── answer_pipeline.py       end-to-end Kira ∥ Cael ∥ L1 ∥ L2 → Vega
│   ├── compare_eval.py          master harness (sample manifest, trace cache, three pipeline arms)
│   ├── prior_agent.py · debate.py · conformal.py · conformal_report.py · graph_features.py
│   ├── baselines/               five baselines + input export + seven-system merge
│   └── ARCHITECTURE.md          full design document + frozen v20 specification (§XVI–XX)
├── tools/                       flat script directory: repro drivers, experiment arms, offline tests, figure scripts
│   ├── slc_mars_env.ps1         FROZEN v20 environment — the reproduction entry point
│   ├── _load_secrets.ps1        env-first credential resolver (shared by both launchers)
│   ├── sync_metagpt_config.ps1  regenerates ~/.metagpt/config2.yaml with NO key in it
│   └── package_release.py       builds the public release tree (allow-list + path scrub + gate)
├── data/                        question pool + dataset manifest (SHA-256 pinned) + evidence jsonl
├── evaluation/                  ground truth (never on the answering path)
├── outputs/
│   ├── traces/                  tracestore.jsonl, calibstore.jsonl, per-question results (traces/by_question, 489 files), tables
│   ├── mcp-venv/ autogen-venv/ mdagents-venv/  per-system interpreters
│   ├── R-libs/                  R library path for DESeq2/TCGAbiolinks
│   ├── slc-tools-cache/ slc-side-cache/        DESeq2/GDC + TIMER/KM caches
│   └── MDAgents/                cloned upstream framework (unmodified; patched by monkeypatch only)
├── paper/                       LaTeX + PDF: results_tables, pipeline_structure, figures/ — NOT in the public release
├── docs/                        MetaGPT usage guide, R package install script
├── .release/                    build output of tools/package_release.py (git-ignored)
└── MetaGPT/                     vendored upstream framework — NOT in the public release; install per §6 step 1
```

`tools/package_release.py` builds the public tree: it copies the allow-listed directories above, rewrites
machine-local absolute paths inside the frozen traces to `<SLC_ROOT>` / `<PROJECT_ROOT>`, and **fails the
build** if any personal absolute path survives. Traces only carry those paths as print/truthiness fields
(never `open()`ed) and the trace cache key is `SLC_TRACE_VERSION`+qid with no content hash, so the rewrite
cannot change a number: `merge_baselines` run against the release tree reproduces
`outputs/traces/baselines_table_master.md` byte-for-byte.

`tools/` is intentionally **flat**: ~30 scripts resolve siblings by module name off `sys.path[0]`
(e.g. `from _topk_head import …`), so sub-packaging it would break the experiment arms.

---

## 6. Environment setup

Windows / PowerShell. Requires Python 3.11, R 4.6.x, and network reachability to the Neo4j + MongoDB
backends (see step 4).

```powershell
# 1) main venv — MetaGPT agents + MCP client
#    MetaGPT is NOT bundled in this release. Clone upstream at the commit v20 was frozen
#    against and install it editable, without pulling its dependency tree:
uv venv .venv --python 3.11
git clone https://github.com/FoundationAgents/MetaGPT.git MetaGPT
git -C MetaGPT checkout 11cdf46
.venv\Scripts\python.exe -m pip install -e .\MetaGPT --no-deps
#    Windows + Python >= 3.9 only: metagpt/_compat.py imports the removed
#    `semantic_kernel.orchestration.sk_function` shim before setting the Proactor event
#    loop policy. Wrap that one import in `try: ... except Exception: pass`, otherwise
#    every entry point dies at import (no such wheel exists on PyPI today).
.venv\Scripts\python.exe -m pip install mcp==1.27.2 pydantic==2.* tenacity json5 `
  qianfan dashscope==1.19.3 scikit-learn pandas==2.1.1 "numpy<2" "pyarrow<18" `
  tree-sitter tree-sitter-python gitignore-parser grep-ast pygithub nbformat ipython `
  chardet Pillow python-docx

# 2) tool venv — what RealMcpBackend spawns each stdio MCP server with
uv venv outputs\mcp-venv --python 3.11
outputs\mcp-venv\Scripts\python.exe -m ensurepip --upgrade
outputs\mcp-venv\Scripts\python.exe -m pip install -r "<SLC_fresh>\requirements-tools.txt" `
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) baseline venvs (only if reproducing the comparison) — see knowledge_agents/baselines/
#    autogen-venv: py3.11 + pyautogen 0.2.35 (all versions require py<3.13)
#    mdagents-venv: py3.13.5 (upstream f-strings require py>=3.12) + httpx pinned <0.28

# 4) credentials — nothing secret is checked in
#    Copy the template and fill in your LLM key + Neo4j/Mongo endpoints & read-only accounts:
Copy-Item secrets.example.ps1 secrets.local.ps1    # secrets.local.ps1 is git-ignored
notepad secrets.local.ps1
#    tools/slc_mars_env.ps1 and run_route_b.ps1 load it via tools/_load_secrets.ps1.
#    Pre-set SLC_LLM_* / NEO4J_* / MONGO_* in your shell or CI store to win over the file.
#    MetaGPT insists on a config *file*, but it needs no key in it: run_metagpt.py injects
#    SLC_LLM_* into the framework in-process. Regenerate a key-free home config with
#    .\tools\sync_metagpt_config.ps1
#
# 5) handoff bundle + R libraries
#    Put $env:SLC_ROOT = '<your extracted SLC_fresh package root>' in secrets.local.ps1 too —
#    it is a machine-local path, not a credential, but the launchers read it from the same
#    untracked file so no personal path is ever checked in. Unset falls back to a placeholder
#    and the local tool caches simply report unavailable.
#    R libraries (TCGAbiolinks / DESeq2 / SummarizedExperiment into outputs/R-libs):
#    Rscript --vanilla docs/install_R_pkgs_tsinghua.R   # with R_LIBS_USER pointing at outputs/R-libs
```

DESeq2 reads local GDC counts when assembled (BRCA/COAD ship ready). For other cancers the counts must
be pulled from GDC **once** — use `-Mode prewarm`, which is the only mode that enables downloads; graded
runs keep downloads off so that data-less cancers return `download_required` promptly rather than stall.

---

## 7. Reproduction

Dot-source the frozen environment first, then run in the same shell (`Start-Process` will not inherit
a later block's environment).

```powershell
. tools\slc_mars_env.ps1        # echoes: SLC-MARS v20 env: TRACE=20 ... MEM_R2=1 | MONGO_set=True creds=<source>

# 1) three pipeline arms on the full clean pool (n=489, seed=5). TRACE_VERSION=20 hits the
#    trace cache, so a re-run is near-instant; a bumped version recomputes everything.
.\.venv\Scripts\python.exe -m knowledge_agents.compare_eval --n 500 --seed 5

# 2) same-question baseline inputs (runner-readable file carries NO ground truth) + five baselines
.\.venv\Scripts\python.exe -m knowledge_agents.baselines.export_inputs
.\.venv\Scripts\python.exe tools\run_baselines_all500.ps1     # each runner resumes from where it stopped

# 3) anti-memorization labelling (dual probe; GT read only inside the tool)
.\.venv\Scripts\python.exe tools\label_antimemorization.py

# 4) tables and figures
.\.venv\Scripts\python.exe -m knowledge_agents.baselines.merge_baselines   # -> outputs/traces/baselines_table_master.md
.\.venv\Scripts\python.exe tools\_v20_master_report.py                     # three-task performance & token cost
.\.venv\Scripts\python.exe tools\_v20_big_table.py                         # TopK dual candidate pool
.\.venv\Scripts\python.exe tools\_v20_main_figure.py                       # -> paper/figures/ (mkdir paper/ first; not bundled)

# 5) paper (LaTeX sources ship separately from this code release; run from the full working tree)
cd paper; xelatex results_tables.tex; xelatex pipeline_structure.tex
```

Single-question end-to-end trace (prints each agent's intermediate and final output):

```powershell
.\run_route_b.ps1 -Mode answer -Seed 42          # or -Qid <qid>, -Buckets membership, -DecisionMode heuristic
```

**Offline self-check** — no database, no LLM, no tools. Twelve scripts; run them all:

```powershell
Get-ChildItem tools\test_*.py | ForEach-Object {
    .\.venv\Scripts\python.exe -W error::SyntaxWarning $_
    if ($LASTEXITCODE) { "FAILED: $($_.Name)" }
}
```

One of them, `tools\test_no_secrets.py`, is a credential guard: it fails the build if any tracked
source, config or document ever again carries a literal API key, password or authenticated URI, and
checks that the local vault stays git-ignored.

All test scripts locate the repo root themselves; no `PYTHONPATH` needed, and each pins the runtime
gates it asserts on, so they report the same result whether or not the frozen environment was dot-sourced first.

---

## 8. Configuration

The frozen values live in `tools/slc_mars_env.ps1`; defaults in code match them.

| variable | v20 | effect |
|---|---|---|
| `SLC_TRACE_VERSION` | `20` | trace-cache key. Bump only when the decision/evidence *structure* changes — it invalidates every cached row. |
| `SLC_PRIOR_AGENT` | `1` | third evidence channel (bare question, zero evidence) |
| `SLC_DEBATE` | `1` | expert debate: one parallel vote on ambiguous items, confirms but never overrules evidence |
| `SLC_LLM_CHIEF` | `1` | LLM chief: knowledge/computation as audit material, verdict handed to the LLM judge |
| `SLC_TRIAD_VOTE` | `1` | three-channel vote (evidence ∥ pure-LLM ×2) |
| `SLC_TRIAD_CRITIQUE` | `1` | parallel cross-critique on conflict, role-heterogeneous personas |
| `SLC_PRIOR_RACE` | `2` | double-fire the prior channel, kills endpoint long-tail timeouts |
| `SLC_MEM_R2` | `1` | **the v20 delta**: membership essentiality re-ranking |
| `SLC_PAIRWISE_MODEL` | `outputs/traces/pairwise_graph_model.json` | pairwise graph-feature logistic artifact |
| `SLC_CONFORMAL_FILE` | unset | path to a threshold json ⇒ live conformal abstention |
| `SLC_DECISION_MODE` | `auto` | `heuristic` (deterministic, LLM-free) \| `llm` \| `auto` |
| `SLC_PARALOG_KSAT` | `4` | paralog-redundancy saturation constant |
| `SLC_DE_TOPN` | `400` | DESeq2 wide panel size (controls paralog visibility) |
| `SLC_DATASET` | `clean` | `clean` \| `full` (raw 647) |

Two known environment traps, both already handled in code: `mcp.ClientSession`'s
`StdioServerParameters.env` does **not** inherit by default
(`knowledge_agents/tools/slc_backend.py` passes `env=dict(os.environ)` explicitly), and
`SLC_TOOLS_CACHE_ROOT` must point at a writable in-project
directory or the tool setup phase raises and gets misclassified as `unavailable`.
PowerShell 5.1 needs a UTF-8 **BOM** on `.ps1` files containing CJK, and `PYTHONUTF8=1` because rpy2
decodes R's `Sys.getenv()` CSV without an encoding and fails on the CJK project path.

**Rollback.** `SLC_MEM_R2='0'` returns byte-identical v16 behaviour; clearing `SLC_PAIRWISE_MODEL`
returns pairwise to the transparent rule path.

---

## 9. Baselines

Five comparison systems under one `_score` implementation, one endpoint, one question set
(`knowledge_agents/baselines/`):

| system | runner | notes |
|---|---|---|
| pure-LLM | `run_pure_llm.py` | single completion, no tools/functions; timeout recorded honestly as `error` and retried on re-run |
| LitRAG | `run_lit_rag.py` + `mongo_rag.py` | Mongo regex recall over question SLC genes → re-rank by disease terms + gene frequency → excerpts into the *same* prompt; a retrieval failure abstains rather than silently degrading to pure-LLM |
| AutoGen | `run_autogen.py` | AssistantAgent + UserProxyAgent(`NEVER`, no code execution); scans assistant history backwards for the last parseable JSON |
| MDAgents | `run_mdagents.py` | **upstream source unmodified** — monkeypatch only: endpoint/model/timeout forcing, Google-generativeai stub, UTF-8 stdout; adaptive basic/intermediate/advanced routing preserved |
| native MetaGPT | `run_metagpt.py` / `baseline_metagpt.py` | the framework's own five-role handoff, no SLC tools |

Zero-retrieval audit: no tool registration in ①, `code_execution_config=False` in ③,
`dataset='custom'` + empty exemplars to close MDAgents' built-in question bank in ④, and no search
libraries present in either framework venv. Baseline strength on Dataset A therefore comes from
*abstention being disallowed by the prompt* plus pretraining exposure to DepMap/CRISPR literature —
which is exactly what Dataset B is designed to expose.

Leakage controls, enforced by file separation rather than convention:
`baselines_input.json` is runner-readable and carries **no ground truth**; `baselines_gt.json` is read
only by the merge step. `tool_runtime.BLOCKED_PARTS` shields `onco_crisprdb_slc` and `benchmark_791` —
the only real CRISPR essentiality data, which is also the ground-truth source, so using it as a
feature would be cheating.

---


---

## 10. Provenance and stated limitations

- **Trace cache as evidence.** Every graded answer is a row in `outputs/traces/tracestore.jsonl` keyed
  by `trace_version`; per-question views are in `outputs/traces/by_question/` (489 files). v20 is
  complete on the pool — 489 rows over 489 distinct qids. Historical arms v10–v18 remain in the same
  file and are reachable by re-gating, which is why the tables can be re-derived rather than re-asserted.
- **MDAgents is an incomplete snapshot** (≈49 %/51 % coverage) and is reported as such, with missing
  answers scored wrong. It is not evidence against the framework, only against that run.
- **Dataset B is defined using pure-LLM probe 1**, so pure-LLM's B scores are partly definitional; see §2.2.
- **TopK is n = 11.** It is a separate ranking task with no A/B split, and the honest finding is the
  pool effect (≈−0.23 NDCG@5 for every system), not fine ordering among rankers.
- **Membership positive/negative is imbalanced** in the clean pool (50 positive / 180 negative), so a
  "always non-essential" responder already scores 78 % on Dataset A·membership. This is the main reason
  Dataset B and the Δchance columns are reported alongside raw accuracy — read them together, and read
  the pre-registered floor comparisons in `ARCHITECTURE.md` §XII before crediting any membership number.
- **Endpoints are shared infrastructure.** Timed-out runs are re-tried by the resume logic, and
  coverage is reported whenever a system's coverage is not 100 %.

---

## 11. Documentation map

| where | what |
|---|---|
| [`knowledge_agents/ARCHITECTURE.md`](knowledge_agents/ARCHITECTURE.md) | full design: tool→agent mapping, schema contract, evidence invariants, paralog buffering, conformal abstention, graph features, baseline harness (§I–XV); frozen v20 flow, gate table, rejected-alternative register, repro commands, results (§XVI–XX) |
| [`paper/results_tables.tex`](paper/results_tables.tex) · [`.pdf`](paper/results_tables.pdf) | main + TopK + ablation tables, typeset |
| [`paper/pipeline_structure.tex`](paper/pipeline_structure.tex) · [`.pdf`](paper/pipeline_structure.pdf) | pipeline figure write-up |
| [`paper/figures/`](paper/figures/) | `main_figure`, `fig_ab_components`, `fig_cd_topk` (pdf + png) |
| [`paper/dataset_b_construction.tex`](paper/dataset_b_construction.tex) | Dataset B supplementary methods |
| [`paper/qa_clean_stats.tex`](paper/qa_clean_stats.tex) | cleaning statistics and clean-pool table |
| [`paper/llm_pool_hit_ndcg.tex`](paper/llm_pool_hit_ndcg.tex), [`paper/topk_knowledge_only_analysis.tex`](paper/topk_knowledge_only_analysis.tex) | TopK pool analyses |
| [`docs/MetaGPT使用指南.md`](docs/MetaGPT使用指南.md) | MetaGPT vendor notes (installed `--no-deps`, what to touch) |
| [`docs/install_R_pkgs_tsinghua.R`](docs/install_R_pkgs_tsinghua.R) | R package bootstrap |

---

## 12. External dependencies and terms

- **[MetaGPT](https://github.com/FoundationAgents/MetaGPT)** is vendored under `MetaGPT/` under its own
  license; this repository does not relicense it.
- The **SLC `multi_agent` handoff bundle** (`SLC_fresh`) supplies the 8 real MCP tools, and is expected
  outside this repository at the path set by `SLC_ROOT`.
- Backends: Neo4j knowledge graph, MongoDB literature collection, TCGA/GDC counts, TIMER, cBioPortal
  (via KM), and an OpenAI-compatible LLM endpoint.
- **Credential policy.** No secret is tracked by this repository. Every backend is addressed through
  environment variables (`SLC_LLM_BASE_URL` / `SLC_LLM_API_KEY` / `SLC_LLM_MODEL`, `NEO4J_URI` /
  `NEO4J_USER` / `NEO4J_PASSWORD`, `MONGO_URI` / `MONGO_DB_NAME` / `MONGO_COLLECTION_PAPERS`), resolved
  by `tools/_load_secrets.ps1`: an already-exported variable wins, otherwise the untracked
  `secrets.local.ps1` at the repo root is loaded. `config2.yaml` ships placeholders only, and
  `llm_settings()` treats an unsubstituted placeholder as *unconfigured* — a misconfigured checkout
  fails with "no endpoint" instead of sending the literal string as an API key.
  Anyone who ran this project before this change should assume their key was visible in
  `run_route_b.ps1`, `tools/slc_mars_env.ps1`, `config2.yaml` and `~/.metagpt/config2.yaml`, and rotate it.

No warranty of fitness for clinical use. Every number in this project is a benchmark measurement on
re-analyzed public screen data, not a therapeutic claim.
