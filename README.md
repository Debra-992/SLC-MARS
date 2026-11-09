# SLC-MARS

**Solute Carrier Multi-Agent Retrieval-and-Scoring System** — a three-agent pipeline that answers
CRISPR essentiality questions about the SLC (solute carrier) gene family from *retrievable, auditable
scientific evidence* rather than from a language model's parametric memory.

> **Frozen release** (2026-11-02).
> Trace key of the released run `SLC_TRACE_VERSION=20`; reproduction entry point [`tools/slc_mars_env.ps1`](tools/slc_mars_env.ps1).
> Full design document: [`knowledge_agents/ARCHITECTURE.md`](knowledge_agents/ARCHITECTURE.md).
> Paper sources (`paper/`: LaTeX, PDFs, figures) are **not part of this code release**, and neither are the
> raw per-question traces. The decision-task numbers of §4.1 are read straight from the frozen tables that
> *do* ship under `outputs/traces/`; the TopK and agent-ablation figures of §4.2–§4.3 are transcribed here
> from the manuscript's frozen result tables, because their LaTeX sources and generating drivers are not
> bundled — the arm-level ranking record that *does* ship is
> [`outputs/traces/topk_eval_summary.txt`](outputs/traces/topk_eval_summary.txt) (§4, §7).

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
procedure (`tools/label_antimemorization.py`; the construction write-up is manuscript-side
`paper/dataset_b_construction.tex`, which does not ship with this release):

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

### 3.1 Decision flow of the released configuration

The release = **the evidence base + R2, a membership essentiality re-ranking**. Upstream of Vega, four
supply paths run in one `asyncio.gather`
([`knowledge_agents/answer_pipeline.py`](knowledge_agents/answer_pipeline.py)): Kira, Cael, L1 (a Prior
agent answering the bare question with zero evidence), and L2 (a second independent pure-LLM pass with a
heterogeneous persona at temperature 0.7).

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

**Membership branch — the one delta over the base (R2).** The decision construct is restated from
"promotion/suppression polarity" to "CRISPR loss-of-function essentiality". Contradictory
promotion *and* suppression literature for the same gene is no longer read as directional evidence;
it is read as an essentiality signal. Five features are re-weighted
(`_membership_r2` in [`knowledge_agents/actions/decision_action.py`](knowledge_agents/actions/decision_action.py)),
coefficients fitted by 5-fold CV on the base configuration at n=53 membership, out-of-sample accuracy
66.8 % over a 60.4 % floor:

| feature | meaning | points to | weight |
|---|---|---|---:|
| `conflict` | same gene both promoted and suppressed in the literature | essential | +1.732673 |
| `dir_present` | any single-direction causal evidence | non-essential | −1.733301 |
| `solo` | paralog orphan (no co-expressed subfamily partner) | essential | +2.634866 |
| `buffered` | paralog buffering (active subfamily co-expression) | non-essential | −0.619402 |
| `mag = tanh(strength)` | association / differential-expression magnitude | essential | +1.934390 |

`P(essential) = σ(z − 0.408812)`; `> 0.5` ⇒ essential. R2 never abstains and never touches the
pairwise branch or the path taken when the gate is off. The provenance marker
`[R2 essentiality re-ranking]` is written into the rationale.

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
Note that the released run's 100 % coverage is a *design outcome* (R2 does not abstain; the triad vote
recovers former abstentions), not a coverage inflation.

---

## 4. Results (frozen release, full clean pool)

Accuracy is the overall measure; chance = 50 %. Δchance is in percentage points.
The rows of §4.1 are re-printed by §7 from the tables that ship in `outputs/traces/`; §4.2–§4.3 quote the
manuscript's frozen TopK and ablation tables (`paper/results_tables.tex`), transcribed here because that
manuscript is not part of this release.

### 4.1 Main decision tasks

| System | A / membership | Δ | A / pairwise | Δ | B / membership | Δ | B / pairwise | Δ |
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

A, overall (n = 489): SLC-MARS heuristic **71.0 %**, auto 70.3 %, MetaGPT 66.5 %, AutoGen 65.6 %,
pure-LLM 63.8 %, MDAgents 59.3 %, LitRAG 57.9 %. B, overall (n = 265): heuristic 54.3 %.

A diagnostic eighth row, **`metaGPT-native(in-harness)`**, is reported separately in
`baselines_table_master.md`. It is the *same* parametric-only stock MetaGPT role as the baseline above, but
evaluated inside the pipeline's comparison harness (`tracestore.systems.baseline`) under that harness's short
per-call budget. It scores 35.4 % overall on A at 48.1 % coverage — and those abstentions are **failed or
unparseable calls, not refusals**: the arm's prompt forbids abstention, and it is never handed the pipeline's
evidence view. So the gap to the standalone `MetaGPT` row (66.5 % at 99.6 % coverage) measures how much the
run harness costs a caller, not what evidence is worth. Do not cite it as "MetaGPT"; the paper's MetaGPT
baseline is the standalone one.

The decisive reading is the **A → B** contrast. Every pure-memory system collapses on B
(pure-LLM membership 56.1 % → 27.5 %, i.e. *below* chance); SLC-MARS degrades mildly and stays the
only system above chance on B / membership. That is the evidence that the pipeline is consuming
retrievable information the baselines do not have.

### 4.2 TopK ranking (11 CRISPR ranking questions, separate task, no A/B split)

Every system is scored on two candidate pools: the **graph pool** (PPR top-15 ∪ ground truth) and the
**LLM self-generated pool** (whatever the model nominates). These values are the manuscript's Table 2; the
shipping record of the same experiment in arm-level form is `outputs/traces/topk_eval_summary.txt` (best
released arm NDCG@5 0.370, graph PPR baseline 0.028).

| System | graph NDCG@5 | graph hit@5 | LLM-pool NDCG@5 | LLM-pool hit@5 |
|---|---:|---:|---:|---:|
| `pure_llm` | 0.3587 | 0.848 | 0.1244 | 0.727 |
| MetaGPT | 0.3357 | 0.818 | 0.1353 | **0.909** |
| AutoGen | 0.3553 | 0.818 | 0.1219 | 0.818 |
| MDAgents ‡ | 0.1972 | 0.273 | 0.1466 | 0.600 |
| LitRAG | 0.3086 | 0.667 | 0.1173 | 0.606 |
| **SLC-MARS per-gene (Kira)** | **0.3838** | **0.909** | **0.1521** | 0.818 |
| SLC-MARS RRF fusion (Kira+Cael) | 0.3704 | 0.818 | 0.1315 | 0.636 |
| SLC-MARS graph+PPR (Cael) | 0.3631 | 0.818 | 0.1349 | 0.545 |
| SLC-MARS graph only (Cael) | 0.3460 | 0.818 | 0.1187 | 0.636 |

‡ averages 10 of 11 questions; one ended in a transient API timeout.

Switching from the graph pool to the self-generated pool drops NDCG@5 by ≈0.23 for *every* system.
Candidate-pool construction, not the ranker, is the bottleneck for this task — and the graph pool is
supplied by Kira.

### 4.3 Agent ablation (three tasks)

Manuscript Table 4; the generating ablation driver stays in the working tree.

| Configuration | Membership | Pairwise | All tasks | TopK NDCG@5 | TopK hit@5 |
|---|---:|---:|---:|---:|---:|
| SLC-MARS full | 69.8 % | 76.6 % | **73.0 %** | 0.3704 | 0.818 |
| without Kira (knowledge) | 56.6 % (−13.2) | 61.7 % (−14.9) | 59.0 % (−14.0) | 0.3631 | 0.818 |
| without Cael (computing) | 67.9 % (−1.9) | 61.7 % (−14.9) | 65.0 % (−8.0) | **0.3838** | **0.909** |

The two supply agents are complementary on the decision tasks; on ranking, knowledge-only is the
strongest single variant — per-gene KG retrieval beats adding a numeric channel to the pool.

Component/pool ablation (TopK, per-row deltas against "Cael graph only") is typeset in the manuscript's
Table 3 (`paper/results_tables.tex`, not bundled): PPR refinement +0.0171, family
feature +0.0038 (negligible), larger candidate pools *hurt* (cap=32 −0.0223, cap=48 −0.0209,
all-features cap=16 −0.0422).

---

## 5. Repository layout

The tree below is exactly what `tools/package_release.py` copies: an allow-list and nothing else.

```
SLC-MARS/
├── run_route_b.ps1              Route-B launcher: env wiring + mode dispatch (demo|question|answer|eval|baseline|compare|calib|prewarm)
├── secrets.example.ps1          Credential template — copy to secrets.local.ps1 and fill in
├── config2.yaml                 LLM endpoint template (placeholders only; values come from env)
├── knowledge_agents/            ── the system ──
│   ├── schema.py                typed contract: ToolStatus / BackgroundKnowledge / ComputedKnowledge
│   ├── roles/                   KnowledgeQueryRole (Kira), ComputeRole (Cael), DecisionRole (Vega)
│   ├── actions/                 retrieval, computation, decision (incl. R2 + triad fusion)
│   ├── tools/                   MCP backend, retrieval/computation adapters, cypher_guard, paralog
│   ├── answer_pipeline.py       end-to-end Kira ∥ Cael ∥ L1 ∥ L2 → Vega
│   ├── compare_eval.py          master harness (sample manifest, trace cache, the pipeline arms)
│   ├── prior_agent.py · debate.py · conformal.py · conf_calib.py · graph_features.py · negative_evidence.py
│   ├── demo.py · run_single_question.py   single-question entry points (§7)
│   ├── baselines/               five baselines + input export + seven-system merge
│   └── ARCHITECTURE.md          full design document + frozen specification (§I–XX)
├── tools/                       repro drivers, offline self-checks, the release builder
│   ├── slc_mars_env.ps1         frozen environment — the reproduction entry point
│   ├── load_secrets.ps1         env-first credential resolver (shared by both launchers)
│   ├── run_baselines_all500.ps1 launches the five baseline runners
│   ├── sync_metagpt_config.ps1  regenerates ~/.metagpt/config2.yaml with NO key in it
│   ├── clean_questions.py · gen_public_inputs.py · label_antimemorization.py
│   ├── test_*.py                twelve offline self-checks: no database, no LLM, no tools
│   └── package_release.py       builds this tree (allow-list + path scrub + content gates)
├── data/                        question pool + dataset manifest (SHA-256 pinned) + evidence jsonl
├── evaluation/                  ground truth (never on the answering path)
├── outputs/
│   ├── traces/                  ships: the frozen result tables (§4) plus the fitted artefacts — pairwise
│   │                            and full-feature model json, conformal thresholds, sample manifests,
│   │                            dataset-B qid lists, baseline inputs and baseline ground truth
│   ├── examples/                ships: end-to-end traces of two questions, one per task
│   ├── mcp-venv/ autogen-venv/ mdagents-venv/  created by §6: per-system interpreters
│   ├── R-libs/                  created by §6: R library path for DESeq2/TCGAbiolinks
│   └── slc-tools-cache/ slc-side-cache/        created at run time: DESeq2/GDC + TIMER/KM caches
└── docs/                        MetaGPT setup guide, R package install script
```

Deliberately **not** in the release: the `MetaGPT/` checkout (§6 step 1 clones it from upstream), the
`paper/` LaTeX sources, the cloned MDAgents framework (§9: unmodified upstream, patched by monkeypatch
only), the interpreters and tool caches listed above, and the per-question run record
(`tracestore.jsonl`, `answers_*.jsonl`, `traces/by_question/`) — that store is one machine's log of
endpoint calls, and the tables under `outputs/traces/` are its published, derived form (§11).

`tools/package_release.py` copies the allow-list, rewrites machine-local absolute paths inside the frozen
tables to `<SLC_ROOT>` / `<PROJECT_ROOT>`, and **fails the build** if any personal path, non-ASCII
filename, CJK run or stale experiment-arm name survives. The rewrite cannot move a number: those paths
are print-only fields in the shipped tables, never opened, and the trace-cache key is
`SLC_TRACE_VERSION` + qid with no content hash.

`tools/` stays **flat**: every script resolves its siblings by module name off `sys.path[0]`, so
sub-packaging the directory would break the drivers and the tests that import them.

---

## 6. Environment setup

Windows / PowerShell. Requires Python 3.11, R 4.6.x, and network reachability to the Neo4j + MongoDB
backends (see step 4).

```powershell
# 1) main venv — MetaGPT agents + MCP client
#    MetaGPT is NOT bundled in this release. Clone upstream at the commit this release was frozen
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
#    tools/slc_mars_env.ps1 and run_route_b.ps1 load it via tools/load_secrets.ps1.
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

DESeq2 reads local GDC counts when assembled (BRCA/COAD come pre-assembled in the handoff bundle). For
other cancers the counts must be pulled from GDC **once** — use `-Mode prewarm`, which is the only mode
that enables downloads; graded runs keep downloads off so that data-less cancers return
`download_required` promptly rather than stall.

---

## 7. Reproduction

Dot-source the frozen environment first, then run in the same shell (`Start-Process` will not inherit
a later block's environment).

```powershell
. tools\slc_mars_env.ps1        # echoes: SLC-MARS released env: TRACE=20 ... MEM_R2=1 | MONGO_set=<bool> creds=<source>

# 1) the pipeline over the full clean pool (n=489, seed=5). The release ships no run record, so this
#    is a fresh pass through the live backends; bumping SLC_TRACE_VERSION likewise recomputes all of it.
.\.venv\Scripts\python.exe -m knowledge_agents.compare_eval --n 500 --seed 5

# 2) same-question baseline inputs (runner-readable file carries NO ground truth) + five baselines
.\.venv\Scripts\python.exe -m knowledge_agents.baselines.export_inputs
.\.venv\Scripts\python.exe tools\run_baselines_all500.ps1     # each runner resumes from where it stopped

# 3) anti-memorization labelling (dual probe; GT read only inside the tool)
.\.venv\Scripts\python.exe tools\label_antimemorization.py

# 4) the seven-system comparison, built from the records steps 1-3 just wrote
.\.venv\Scripts\python.exe -m knowledge_agents.baselines.merge_baselines   # -> outputs/traces/baselines_table_master.md
```

The tables in §4 and the fitted artefacts ship as files under `outputs/traces/`, so the published numbers
can be read without running anything. Re-running steps 1–4 reproduces the *procedure*, not the digits:
endpoint sampling and graph contents drift, which is why the frozen tables, not an ad-hoc re-run, are the
citable record (§11). Typesetting the paper's own tables happens in the manuscript tree, which is not
part of this release.

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

| variable | released value | effect |
|---|---|---|
| `SLC_TRACE_VERSION` | `20` | trace-cache key. Bump only when the decision/evidence *structure* changes — it invalidates every cached row. |
| `SLC_PRIOR_AGENT` | `1` | third evidence channel (bare question, zero evidence) |
| `SLC_DEBATE` | `1` | expert debate: one parallel vote on ambiguous items, confirms but never overrules evidence |
| `SLC_LLM_CHIEF` | `1` | LLM chief: knowledge/computation as audit material, verdict handed to the LLM judge |
| `SLC_TRIAD_VOTE` | `1` | three-channel vote (evidence ∥ pure-LLM ×2) |
| `SLC_TRIAD_CRITIQUE` | `1` | parallel cross-critique on conflict, role-heterogeneous personas |
| `SLC_PRIOR_RACE` | `2` | double-fire the prior channel, kills endpoint long-tail timeouts |
| `SLC_MEM_R2` | `1` | **the one delta over the base**: membership essentiality re-ranking |
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
PowerShell 5.1 needs a UTF-8 **BOM** on `.ps1` files holding non-ASCII text, and `PYTHONUTF8=1` because
rpy2 decodes R's `Sys.getenv()` CSV without an encoding and fails when the working directory is named in
a non-ASCII script.

**Rollback.** `SLC_MEM_R2='0'` returns byte-identical behaviour to the base configuration (no R2);
clearing `SLC_PAIRWISE_MODEL` returns pairwise to the transparent rule path.

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
| stock MetaGPT | `run_metagpt.py` / `baseline_metagpt.py` | the framework's own Role/Action shell around one `_aask`: no tools, no retrieval, no SLC evidence; the standalone runner is the Table 1 row, `baseline_metagpt.py` is the in-harness diagnostic arm (§4.1) |

Zero-retrieval audit: no tool registration for pure-LLM, `code_execution_config=False` for AutoGen,
`dataset='custom'` + empty exemplars to close MDAgents' built-in question bank, and no search
libraries present in either framework venv. Baseline strength on Dataset A therefore comes from
*abstention being disallowed by the prompt* plus pretraining exposure to DepMap/CRISPR literature —
which is exactly what Dataset B is designed to expose.

Leakage controls, enforced by file separation rather than convention:
`baselines_input.json` is runner-readable and carries **no ground truth**; `baselines_gt.json` is read
only by the merge step. `tool_runtime.BLOCKED_PARTS` shields `onco_crisprdb_slc` and `benchmark_791` —
the only real CRISPR essentiality data, which is also the ground-truth source, so using it as a
feature would be cheating.

---

## 10. Designs that were tried and rejected

Kept behind default-off gates in `tools/slc_mars_env.ps1`, so each is reproducible without reviving it
into the released path. Recorded because two of them are load-bearing negative results.

| mechanism tried | verdict | how to replay |
|---|---|---|
| prior-channel ensemble: N samples in parallel, mode aggregation | **negative**: A 75.0→72.5, B 68.4→63.2 — aggregation forces low-consensus items into hard answers; the "agreement" gain is agreement among errors | `SLC_PRIOR_ENSEMBLE=3` |
| answer-layer ensemble: end-to-end N-sample vote | **negative**: the deterministic arm collapses (B 63.2→26.3) — N samples share one evidence backbone, so they are not i.i.d. reasoning samples | `SLC_ANSWER_ENSEMBLE=3` |
| serial triad vote (second vote at elevated temperature) | mechanism positive (+9.1 pp on the common subset) but **operationally negative**: 2 serial LLM calls/question → 17.5 % timed out of the table | only the parallel form ships (`SLC_TRIAD_VOTE=1`) |
| the voting stack that ships: L2 inside the parallel supply gather, persona heterogeneity, cross-critique, double-fire racing | **positive, kept**: 0/40 absent; n=100 overall 70.0 % > AutoGen 69.0 / MDAgents 63.0 | frozen in |
| Mongo excerpts as the second vote's source (de-correlation) | not adopted, no cross-subset gain | `SLC_PRIOR_RETRIEVE=1` |
| membership negative-evidence channel + confidence-selective abstention | **negative @n=100**; the per-slice leads were sampling artifacts | `SLC_NEG_EVID=1`, `SLC_ABSTAIN_THETA>0` |
| literature RAG on the baseline side | **harmful**: pairwise 86.4→54.5 — raw excerpts push the model toward "appears more in the literature = more essential", and long context induces timeouts | `run_lit_rag.py` |
| MDAgents-style fully serial expert debate | **zero accuracy gain at an order-of-magnitude cost** (4–17 min/question vs ~30 s) | `run_mdagents.py` |

Two laws fall out of this, and both were paid for in wall-clock:

1. **Ensembling only helps over independent, complete reasoning paths aggregated at the decision
   point.** When a deterministic backbone dominates the verdict, majority voting averages the correct
   answer away — the same mistake at the prior layer and at the answer layer costs the same amount.
2. **A mechanism claim must be sized to the metric.** n=40 "net-zero" readings were measurement
   insufficiency, not evidence: a 75 % reading at n=40 was exposed by n=100 as small-sample luck (true
   47 %), while the voting stack's real gain lives precisely in the "recover an answer from an honest
   abstention" scenario that n=40 does not contain.

---

## 11. Provenance and stated limitations

- **The run record stays with the working tree.** Every graded answer was written as a row of
  `outputs/traces/tracestore.jsonl`, keyed by `trace_version`, with per-question views alongside; the
  released run is complete on the pool — 489 rows over 489 distinct qids. That store is one machine's
  log of endpoint calls and does not ship; what ships is its derived form, the frozen tables under
  `outputs/traces/`. Earlier gated configurations stay reachable in the store by re-gating, which is why
  the tables can be re-derived rather than re-asserted.
- **MDAgents is an incomplete snapshot** (≈49 %/51 % coverage) and is reported as such, with missing
  answers scored wrong. It is not evidence against the framework, only against that run.
- **Dataset B is defined using pure-LLM probe 1**, so pure-LLM's B scores are partly definitional; see §2.2.
- **TopK is n = 11.** It is a separate ranking task with no A/B split, and the honest finding is the
  pool effect (≈−0.23 NDCG@5 for every system), not fine ordering among rankers.
- **Membership positive/negative is imbalanced** in the clean pool (50 positive / 180 negative), so a
  "always non-essential" responder already scores 78 % on Dataset A / membership. This is the main reason
  Dataset B and the Δchance columns are reported alongside raw accuracy — read them together, and read
  the pre-registered floor comparisons in `ARCHITECTURE.md` §XI before crediting any membership number.
- **Endpoints are shared infrastructure.** Timed-out runs are re-tried by the resume logic, and
  coverage is reported whenever a system's coverage is not 100 %.

---

## 12. Documentation map

| where | what |
|---|---|
| [`knowledge_agents/ARCHITECTURE.md`](knowledge_agents/ARCHITECTURE.md) | full design: tool→agent mapping, schema contract, evidence invariants, paralog buffering, conformal abstention, the membership levers, graph features, baseline harness (§I–XV); the frozen decision flow, released knob table, rejected-alternative register, repro commands, results (§XVI–XX) |
| [`outputs/traces/baselines_table_master.md`](outputs/traces/baselines_table_master.md) | seven systems × two dataset views, with counts |
| [`outputs/traces/compare_table_n489_seed5.md`](outputs/traces/compare_table_n489_seed5.md) | the n=489 / seed=5 comparison in coverage-and-accuracy form |
| [`outputs/traces/pipeline_full489_report.txt`](outputs/traces/pipeline_full489_report.txt) · [`baseline_final_report.txt`](outputs/traces/baseline_final_report.txt) · [`baselines_by_dataset_report.txt`](outputs/traces/baselines_by_dataset_report.txt) | stratified pipeline and baseline tables |
| [`outputs/traces/conformal_report.md`](outputs/traces/conformal_report.md) | risk–coverage, the fitted threshold and the guarantee check |
| [`outputs/traces/topk_eval_summary.txt`](outputs/traces/topk_eval_summary.txt) | TopK ranking arms, per-question NDCG@5 and slices |
| [`outputs/traces/efficiency_accuracy_master.txt`](outputs/traces/efficiency_accuracy_master.txt) | accuracy against token cost |
| [`outputs/examples/`](outputs/examples/) | end-to-end traces of two questions, one per task |
| [`docs/MetaGPT_Setup_Guide.md`](docs/MetaGPT_Setup_Guide.md) | MetaGPT vendor notes (installed `--no-deps`, what to touch) |
| [`docs/install_R_pkgs_tsinghua.R`](docs/install_R_pkgs_tsinghua.R) | R package bootstrap |

Typeset tables, the pipeline figure and the supplementary methods (Dataset B construction, cleaning
statistics, TopK pool analyses) live in the manuscript tree (`paper/`), which is distributed separately
from this code release.

---

## 13. External dependencies and terms

- **[MetaGPT](https://github.com/FoundationAgents/MetaGPT)** is not bundled: §6 step 1 clones upstream
  at the pinned commit and installs it editable. It stays under its own license and this repository does
  not relicense it.
- The **SLC `multi_agent` handoff bundle** (`SLC_fresh`) supplies the 8 real MCP tools, and is expected
  outside this repository at the path set by `SLC_ROOT`.
- Backends: Neo4j knowledge graph, MongoDB literature collection, TCGA/GDC counts, TIMER, cBioPortal
  (via KM), and an OpenAI-compatible LLM endpoint.
- **Credential policy.** No secret is tracked by this repository. Every backend is addressed through
  environment variables (`SLC_LLM_BASE_URL` / `SLC_LLM_API_KEY` / `SLC_LLM_MODEL`, `NEO4J_URI` /
  `NEO4J_USER` / `NEO4J_PASSWORD`, `MONGO_URI` / `MONGO_DB_NAME` / `MONGO_COLLECTION_PAPERS`), resolved
  by `tools/load_secrets.ps1`: an already-exported variable wins, otherwise the untracked
  `secrets.local.ps1` at the repo root is loaded. `config2.yaml` ships placeholders only, and
  `llm_settings()` treats an unsubstituted placeholder as *unconfigured* — a misconfigured checkout
  fails with "no endpoint" instead of sending the literal string as an API key.
  Keys belong only in `secrets.local.ps1` or in the environment: if one was ever written into
  `run_route_b.ps1`, `tools/slc_mars_env.ps1`, `config2.yaml` or `~/.metagpt/config2.yaml`, treat it as
  exposed and rotate it. `tools/test_no_secrets.py` is what keeps it that way.

No warranty of fitness for clinical use. Every number in this project is a benchmark measurement on
re-analyzed public screen data, not a therapeutic claim.
