# SLC-MARS · the three-agent knowledge-supply architecture

> **Frozen release** (2026-11-02). Reproduction entry point: `tools/slc_mars_env.ps1`. The released
> decision flow and the numbers it produced are §XVI and §XX; §I–XV describe the machinery, which is
> independent of any particular run. For the overview and quick start see `README.md` in the repository root.
>
> Depends on MetaGPT (`<PROJECT_ROOT>/MetaGPT`, cloned as in README §6). The code lives in `knowledge_agents/`.
> This design splits the eight real tools of the SLC `multi_agent` handoff bundle by semantics into two
> **supply-side** agents — **Kira** (knowledge retrieval, background evidence) and **Cael** (computation,
> candidate numeric evidence) — while **Vega**, the decision agent, consumes only the evidence those two
> already produced. The supply sides are independent and run in parallel; everything downstream depends
> only on the typed contract in `schema.py`.

## I. Tool → agent mapping

| agent | real tool it fronts | upstream source | output |
|---|---|---|---|
| **knowledge retrieval, `KnowledgeQueryRole`** | `project_knowledge` | slc_benchmark_context_mcp.py | Neo4j three-path evidence + Mongo literature |
| | `kg_schema` → `kg_cypher` (text2cypher) | slc_kg_mcp.py | LLM-generated read-only Cypher to fill background gaps |
| | `kg_gene_neighbors` | slc_kg_mcp.py | neighbours when a candidate has too little coverage |
| **computation, `ComputeRole`** | `project_graph_computation` (PPR) | ppr_weighted.py | disease-seed weighted proximity score |
| | `timer_pancancer_expression` | timer.py | pan-cancer tumour/normal expression |
| | `km_survival` | km_plotter.py | survival-analysis statistics as printed text |
| | `slc_differential_expression` | TCGA.py / DESeq2.py | DESeq2 differential expression |

## II. Overall architecture

```
                     downstream consumer (watches both cause_by messages)
                                  ▲                 ▲
              BackgroundKnowledge            ComputedKnowledge
               (knowledge_type=                (knowledge_type=
                 background_kg)                  candidate_data)
                     │                 │
        ┌────────────┴───┐       ┌─────┴──────────────┐
        │ KnowledgeQuery │       │    ComputeRole      │
        │   Role (Kira)  │       │  (routes 4 kinds of │
        │                │       │   computation)      │
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

## III. Files and layers

| layer | file | responsibility |
|---|---|---|
| contract | `schema.py` | `ToolStatus` / `BackgroundKnowledge` / `ComputedKnowledge` and their sub-structures, `parse_knowledge()` |
| Action | `actions/knowledge_retrieval_action.py` | project_knowledge + text2cypher (the latter only when coverage is thin) |
| Action | `actions/computation_action.py` | routes PPR/TIMER/KM/DESeq2 by the semantics of the question |
| Action | `actions/decision_action.py` | the decision kernel: three channels, fusion, R2, triad vote, conformal hook |
| Role | `roles/knowledge_query_role.py` | `KnowledgeQueryRole` (Kira), wires RetrievalTools |
| Role | `roles/compute_role.py` | `ComputeRole` (Cael), wires ComputationTools, `set_task()` injects the routing parameters |
| Role | `roles/decision_role.py` | `DecisionRole` (Vega), consumes the assembled evidence view |
| Tool | `tools/retrieval_tools.py` | adapters for the four retrieval tools, read-only guard |
| Tool | `tools/computation_tools.py` | adapters for the four computation tools, keeps `evidence_scope` / `interpretation` |
| Tool | `tools/cypher_guard.py` | mirrors the real `validate_cypher` (rejects writes, procedures, multi-statement payloads) |
| Tool | `tools/paralog.py` | subfamily parsing, functional-symbol filter, redundancy from the DE panel |
| Tool | `tools/slc_backend.py` | RealMcpBackend (live calls) / UnavailableBackend (fallback), `get_default_backend()` chooses |
| Pipeline | `answer_pipeline.py` | end-to-end Kira ∥ Cael ∥ L1 ∥ L2 → Vega |
| Harness | `compare_eval.py` | sample manifest, trace cache, the pipeline arms, the comparison table |
| Demo | `demo.py`, `run_single_question.py` | two-role and end-to-end single-question runs |
| Calibration | `calib_dump.py`, `conformal_report.py` | heuristic-only pool dumper, then threshold fit, risk-coverage report and threshold export |
| Cache warm-up | `prewarm_de.py` | pays the per-cancer DESeq2 cost before the scored run |

The two model-fitting drivers (`tools/fit_pairwise_model.py`, `tools/train_membership_logit.py`) belong to
the working tree and are not part of the release; the artefacts they produced do ship
(`outputs/traces/pairwise_graph_model.json`, `fullfeature_model.json`, `conformal_thresholds.json`), and
the online scorer needs only those. Everything the shipped launcher can dispatch, and everything the
reproduction commands below name, ships with the tree.

## IV. Evidence-integrity invariants (copied from the real tools, never relaxed)

1. **Failure, unsupported and no-data are all *unknown*, never negative.** They map to
   `ToolStatus.{unavailable, download_required, unsupported_cancer, no_seed, rejected, error}` and are
   never written as `ok` or as an empty success.
2. **No fabricated values.** TIMER `plot_only` invents no number; KM without an HR reports none;
   DESeq2 without counts returns `download_required`.
3. **Read-only boundary.** Cypher from text2cypher passes `cypher_guard`; write clauses, stored
   procedures and multi-statement payloads come back `rejected`.
4. **Source independence is declared, not assumed.** `source_independence=NOT_ESTABLISHED` and
   `public_web_used=False` are propagated verbatim.
5. **Routing is deterministic, not model-chosen.** Computation tools are triggered by explicit rules
   rather than autonomous function calling; unrouted tools land in `skipped_tools` and are not read as
   negative evidence.

## V. Running it

### 5.1 Environment preparation (one-off)

```powershell
# main venv: MetaGPT agents + MCP client (Python 3.11)
uv venv .venv --python 3.11
.venv\Scripts\python.exe -m pip install -e .\MetaGPT --no-deps
.venv\Scripts\python.exe -m pip install mcp==1.27.2 pydantic==2.* tenacity json5 `
  qianfan dashscope==1.19.3 scikit-learn python-docx Pillow chardet `
  pandas==2.1.1 "numpy<2" "pyarrow<18" tree-sitter tree-sitter-python `
  gitignore-parser grep-ast pygithub nbformat nbclient ipython ipykernel ipywidgets
# (that set is the minimum MetaGPT actually pulls in for this project)

# tool venv: what the SLC MCP servers run under (Python 3.11 + requirements-tools)
uv venv outputs\mcp-venv --python 3.11
outputs\mcp-venv\Scripts\python.exe -m ensurepip --upgrade
outputs\mcp-venv\Scripts\python.exe -m pip install `
  -r "<SLC_fresh>\requirements-tools.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5.2 Running

```powershell
# offline (no mcp installed, no SLC_ROOT): everything goes through UnavailableBackend, which is how
# the "never fabricate" property is exercised
python -m knowledge_agents.demo

# route B (real MCP), the intended path: the launcher wires the environment and dispatches a mode
.\run_route_b.ps1 -Mode answer -Seed 42
```

Five variables have to be set or the tools fall back to `unavailable` inside the child process; the
frozen launcher exports all of them:

| variable | value | why |
|---|---|---|
| `SLC_ROOT` | root of the SLC_fresh handoff bundle | lets `RealMcpBackend` locate `mcp_servers` |
| `SLC_MCP_PYTHON` | `outputs\mcp-venv\Scripts\python.exe` | interpreter used to spawn each stdio server |
| `SLC_KMTIMER_ROOT` | `$SLC_ROOT\outputs\tool-cache\KMTIMER` | root TIMER/KM read their cache from |
| `SLC_TOOLS_CACHE_ROOT` | a writable directory inside the project | cwd and temp for `_tools_worker.py` |
| `SLC_TOOLS_OFFLINE` | `1` | cache reads only; a missing cache returns `download_required` |

### 5.3 Verified real output (offline, over MCP)

```
[candidate knowledge] status=ok
  TIMER: status=ok scope=plot_only        boxplot=...boxplot_SLC25A1.png
  KM    : status=ok scope=plot_and_stats  stats_head='P value: 0.0032'
  computation finished: 2 tools usable (ppr=not_routed, timer=ok, km=ok, de=download_required);
              unrouted/unknown: ['ppr:not_routed']. Unknown is not negative.
[background knowledge] status=rejected   # the qid is not in the SLC_MCP_PUBLIC_INPUTS allow-list,
                                         # so an honest "unknown" is the expected answer
```

### 5.4 Two environment traps, both paid for

1. **`mcp.ClientSession`'s `StdioServerParameters.env` does not inherit by default.** Without
   `env=dict(os.environ)` the SLC server child loses `SLC_KMTIMER_ROOT`, `_tools_worker.py` cannot find
   the cache root, and `safe_error` answers `unavailable` for everything. Passed explicitly in
   `slc_backend.py`.
2. **`SLC_TOOLS_CACHE_ROOT` must point at a writable in-project directory.** The default is
   `%TEMP%\slc-tools`; a Windows sandbox combined with `install_read_guard` blocks the `mkdir`/`chdir`,
   and a `FileNotFoundError` from the setup phase then gets misclassified as `unavailable` instead of
   surfacing as an error.

## VI. Relation to the existing five-role SLC pipeline

The two supply agents are exactly the `Knowledge → Tool` stages of that pipeline. The two message types
they emit can be handed to an `Analyst` as context (`watch` on both `cause_by`), and adding Reviewer and
Adjudicator downstream reconstructs the full five-role handoff. The stock-MetaGPT baseline of §XIV is
MetaGPT's own Role/Action machinery answering without this repository's tools -- not that five-role chain.

## VII. The decision agent Vega (Kira ∥ Cael → Vega)

Above the two supply sides sits the **decision role `DecisionRole` (Vega)**, which consumes only the
`BackgroundKnowledge` + `ComputedKnowledge` Kira and Cael produced and **never re-fetches**. The kernel
is `actions/decision_action.py` and honours three settings of `SLC_DECISION_MODE`: `heuristic`
(deterministic scoring, no LLM), `llm`, `auto` (heuristic where there is direct evidence, cross-cancer
analogy only where evidence is thin). The invariants of §IV carry over unchanged: unknown ≠ negative,
insufficient evidence means abstention, nothing is fabricated.

Vega aggregates each question's evidence into three **equal-weight** channels
(`_collect_scores` returns `(strength, pol, buf, basis)`):

| channel | meaning | range | source |
|---|---|---|---|
| `strength` | magnitude: which candidate has louder evidence (association / differential expression) | ≥ 0 | KG paths / PPR / DE / TIMER |
| `pol` | causal direction: promotion or suppression polarity from the temporal causal graph, which sets the answer direction | {-1, 0, +1} | RelaEvent hyper-edge direction ⊗ the verb in the question |
| `buf` | paralog buffering: functional redundancy argues against essentiality (§VIII) | [-1, 0] | co-expression inside the same subfamily in the target cancer's DESeq2 panel |

Fusion (`W_DIRECTION = 0.6`, `W_BUFFER = 0.8`, `MIN_SCORE_TO_ANSWER = 0.5`):

- **pairwise**: `combined = strength + W_DIRECTION·pol + W_BUFFER·buf`, the maximum wins; if all three
  components are short of the floor, abstain.
- **membership**: net polarity `net = pol + W_BUFFER·buf`; `net > 0` → essential, `net < 0` →
  non-essential, `net == 0` (neither causal direction nor buffering evidence) → **honest abstention**.
  Confidence `conf = 0.3 + 0.12·|net| + 0.1·log1p(strength)`.

> Key design: `buf` is **one-sided evidence** — it can only push a verdict towards "less essential";
> the absence of a paralog is never read as proof that a gene is essential.

## VIII. Paralog buffering

**Mechanism.** SLC genes cluster into subfamilies (the `SLC<number>` prefix, e.g. SLC7, SLC25, SLC12)
and members of a subfamily often overlap functionally. If several co-subfamily paralogs of a gene are
actively co-expressed *in the target cancer*, knocking that gene out is more likely to be buffered, so a
CRISPR loss-of-function screen is less likely to call it essential — which speaks directly to the
construct gap between "associated / expressed" and "CRISPR essential".

**Implementation** (`tools/paralog.py`; the data source is the cancer's DESeq2 `top_genes`, and without a
panel the channel is unknown and switched off):

- `subfamily()`: `SLC25A32 → SLC25`; names that do not follow the SLC convention → `None` (they take no
  part in the redundancy estimate).
- `is_functional_symbol()`: drops pseudogenes, antisense and intronic transcripts
  (`SLC25A14P1`, `SLC22A18AS`, `SLC7A6OS`, `SLC7A2-IT1`, `SLC25A21-AS1`) so that a large family cannot
  inflate buffering by member count alone.
- `panel_from_de(de)`: turns the DE table into `{gene: expression weight}` with
  `weight = min(|log2FC|, LNCAP) / LNCAP` (`LNCAP = 5`) — significance *and* effect size, because a
  strongly displaced paralog is the one that can actually buffer.
- `redundancy(gene, panel, k_sat)`: `redundancy = min(1, Σ_paralog weight / k_sat)`, graded by the
  strength of the co-expression.
- `computation_action.py` computes `redundancy_map(candidates, panel)` after DE and writes
  `ComputedKnowledge.paralog_redundancy` / `paralog_partners`; with no panel it records
  `skipped=paralog:no_expression_panel`.

## IX. Conformal calibrated abstention

Choosing the abstention threshold by hand (`MIN_SCORE_TO_ANSWER`) is arbitrary. This mechanism calibrates
a confidence threshold instead, with **Conformal Risk Control** (Bates et al. 2021) on a disjoint
calibration split, and reports the risk–coverage curve, so the operating point rests on a statistical
guarantee rather than on taste.

- Numeric core, `conformal.py` (pure functions): non-conformity score `s = 1 - confidence`; a question is
  answered iff `s ≤ τ`; the finite-sample "add one" empirical risk is
  `L̂(τ) = (Σ loss·1[s ≤ τ] + 1) / (N + 1)`; `τ̂ = sup{τ : L̂(τ) ≤ α}`, and if no τ qualifies then
  `τ̂ = None`, i.e. abstain everywhere, which is the correct conservative behaviour. The guarantee is
  `E[loss·1[s ≤ τ̂]] ≤ α`, hence **the error rate among answered questions ≤ α / coverage**.
- Calibration and reporting, `conformal_report.py`: splits questions into disjoint
  calibration and test halves by `sha1(seed|qid)`, fits τ̂ on the calibration half, reports
  coverage / answered error rate / ECE on the test half, and exports the threshold file.
- In-flight hook, `decision_action.py`: every answer returned by `run()` passes `_conformal_filter(ans,
  bucket)`; if `SLC_CONFORMAL_FILE` points at a json containing `thresholds[bucket]` and the answer's
  `confidence` is below it, the answer is **converted into an abstention**, recorded in
  `Answer.conformal_threshold` and prefixed in the rationale with `[conformal abstention ...]`.

> Note. A threshold file calibrated on `pipeline_heuristic` confidences may only be fed to heuristic
> decisions, and **the calibration input must be raw confidence, before any conformal filtering**
> (`calib_dump.py` explicitly clears `SLC_CONFORMAL_FILE` to avoid a self-fulfilling loop). With a small
> calibration set the finite-sample floor `1/(N+1)` exceeds α, so `τ̂ = None` — correctly "abstain
> everywhere"; a deployable tight threshold needs a large calibration pool.

## X. Reproducing buffering and conformal

```powershell
# (optional) warm the DESeq2 cache for more cancers: buffering only fires where a DE panel exists
.\run_route_b.ps1 -Mode prewarm        # downloads enabled, one pass per cancer, cache written

# calibration pool (heuristic only, no LLM calls, tools and retrieval dominate)
.\run_route_b.ps1 -Mode calib -Eval 250 -Seed 5      # -> outputs/traces/calibstore.jsonl

# fit the threshold on that pool, report risk-coverage and ECE, export the deployable file
.\.venv\Scripts\python.exe -m knowledge_agents.conformal_report `
    --traces outputs\traces\calibstore.jsonl --system pipeline_heuristic --alpha 0.10
#   -> outputs/traces/conformal_report.md + conformal_thresholds.json
#      (no threshold exported when alpha is too small: that is the correct conservative answer)

# fair three-system comparison (heuristic / auto / native MetaGPT on one shared evidence view)
.\run_route_b.ps1 -Mode compare -Eval 40 -Seed 5

# deploy in-flight conformal: mount the exported thresholds and answer again
$env:SLC_CONFORMAL_FILE = "<PROJECT_ROOT>\outputs\traces\conformal_thresholds.json"
```

Knobs:

| variable | default | effect |
|---|---|---|
| `SLC_PARALOG_KSAT` | `4` | paralog-redundancy saturation constant (Σ weight = k_sat ⇒ redundancy 1) |
| `SLC_CONFORMAL_FILE` | empty | points at `{thresholds: {bucket: conf}}` json; empty disables in-flight conformal |
| `SLC_PAIRWISE_MODEL` | empty | points at the pairwise graph-feature logistic artefact; empty means pairwise runs on the transparent rule |
| `SLC_DECISION_MODE` | `auto` | `heuristic` \| `llm` \| `auto` |
| `SLC_DE_TOPN` | `400` | DESeq2 wide panel size, which sets how visible paralogs are |
| `SLC_TRACE_VERSION` | `20` | trace-cache key; bump it when the decision/evidence structure changes |

Offline self-check (no database, no LLM, no tools):

```powershell
.\.venv\Scripts\python.exe tools\test_paralog_conformal.py   # buffering + conformal math + in-flight hook
```

---

## XI. Membership: what the evidence can and cannot carry

Membership ("is SLCx essential in cancer c under screen s?") was the weakest bucket, so it was worked
through in priority order: one correctness fix, three ideas blocked by data availability and the leakage
rules, one learned calibration that moved the ceiling, and one tuning idea that the calibration made
redundant. Measurements are on the 250-question clean calibration pool (seed 5, qids drawn from the same
distribution the heuristic decides on).

> The drivers behind the numbers in §XI–§XIII divide in two. The calibration pool and the fitted
> conformal thresholds were produced by `calib_dump.py` and `conformal_report.py`, which ship, so the
> measurement can be repeated on this tree. The two logistic fits
> (`tools/train_membership_logit.py`, `tools/fit_pairwise_model.py`) belong to the working tree and do not
> ship (§III). What they produced does ship: the serialized artefacts under `outputs/traces/`. These three
> sections therefore record **what was measured and why the alternatives were rejected**, which is what a
> reader needs in order to trust an artefact they cannot refit.

### Lever 1 -- sign from causal direction only, and the `net == 0` float trap

The membership branch of `decision_action._heuristic` was rewritten so that the **sign** of the verdict is
decided by the causal direction `pol` alone. Paralog buffering may only *weaken* the net strength once a
direction exists; with `pol = 0` it must never carry a gene to "non-essential" on its own, since that reads
unknown as negative (§IV). The old rule was

```
net = pol + W_BUFFER * buf;   net == 0  ->  abstain
```

which manufactured false negatives in bulk: with `pol = 0` and any buffering evidence at all,
`net = 0.8 * buf < 0` answered "non-essential" without one directional observation, and the exact float
equality `== 0` failed to catch the cases that landed off zero by rounding. The new rule is
`pol == 0 -> abstain` and `abs(net) < 1e-9 -> abstain`. The offline self-check for this branch is 29
assertions, all passing, including "buffering alone -> abstain, never a fabricated negative".

Effect on the same 250-question pool: membership coverage fell from 56 % to **40.3 %** (48 of 119 answered),
which is the honest contraction of the guesses that were withdrawn -- but accuracy **among answered**
questions was only **41.7 %**, at or below the 53 % majority-class floor. So the polarity-driven residual
signal is not enough to decide essentiality, and conformal finds **no certifiable threshold** for membership
at alpha = 0.10 (its calibration half contained just 22 answered questions).

### Levers 2-4 -- CRISPR positive channel, polarity redirection, cell-line matching: blocked, not skipped

Source inspection settles this quickly: the ground truth of the benchmark *is* CRISPR screen output (MAGeCK
Low/High-Growth), and the supplied tooling deliberately withholds that class of signal so that no feature
can smuggle the label back in.

- `slc_benchmark_context_mcp.project_knowledge` hard-codes `matched_screen_measurements_available=False`;
- `tool_runtime.BLOCKED_PARTS` shields `onco_crisprdb_slc` and `benchmark_791` -- the only real CRISPR
  essentiality data, i.e. the blocked truth source itself; using it as a feature would be cheating;
- the graph carries only `SLCGene -[:IS_SOURCE]-> RelaEvent <-[:IS_TARGET]- Disease`, with
  `rel_AB / rel_BC in {PROMOTION, SUPPRESSION}` meaning **gene-to-disease association polarity**; there is
  **no knockout or viability edge**, so there is nothing for a polarity redirect to attach to;
- expression is TIMER/TCGA **bulk tumour vs adjacent normal** only; without a CCLE cell-line panel there is
  no cell-line-level matching to be done.

Conclusion: under the hard constraints "do not leak the CRISPR truth, use only the eight tools", levers 2-4
are **not implementable**. They are registered as blocked rather than worked around with a proxy that would
silently inflate the score.

### Lever 5 -- leakage-free logistic calibration, conformal on top

Hand-set constants (`W_BUFFER`, `0.3 + 0.12|net|`) were compressing a weak signal towards chance, so the
mapping from evidence to essentiality was instead **learned** against the 500-question clean truth set. A
leakage-free feature vector is read off raw tool numbers -- gene-to-disease promotion/suppression counts
(`n_promote`, `n_suppress`, `n_paths`, `n_cancer_hit`), PPR, TIMER/KM availability, the queried gene's DE
`log2FC`, panel size, paralog redundancy -- and the label is `ground_truth.answer`. The guards are strict:

- train and evaluation are split by **disjoint qid hash**, so a question's label touches one side only; `C`
  is selected by cross-validation *inside* the training half; probabilities go through
  `CalibratedClassifierCV` so that they may be used as confidence;
- the learned model is compared with the heuristic **on the same test subset**, with the majority-class
  dummy as an explicit floor.

Results (N = 119 membership, essential base rate 52.9 %):

- **repeated 10 x 5-fold CV: AUC = 0.730 +/- 0.017 (worst fold 0.701)**, acc@0.5 between 68 and 71 %. There
  *is* usable signal in the leakage-free features: the hand rules were leaving it on the table, the data was
  not empty.
- one disjoint split (train 81 / test 38), logistic answering everything scores 60.5 % (above the 50 %
  dummy); adding conformal on top moves the operating point:

  | decision rule | coverage | acc (answered) | acc (overall) |
  |---|---:|---:|---:|
  | majority-class dummy | 100% | 50.0% | 50% |
  | hand heuristic | 47% | 61.1% | **28.9%** |
  | logistic, answer at 0.5 | 100% | 60.5% | **60.5%** |
  | logistic + conformal alpha=0.20 | 89.5% | 64.7% | **57.9%** |
  | logistic + conformal alpha=0.10 | 21.1% | 75.0% | 15.8% (high precision, heavy refusal) |

  The learned version beats both the heuristic and the dummy **at the same time** on coverage and on
  accuracy, and roughly doubles the net contribution.
- the fitted weights put `de_log2FC` (+1.16), `paralog_redundancy` (+0.79), `n_promote` (+0.48) and
  `n_paths` (+0.40) in charge. Note that `paralog_redundancy` is learned **positive**, opposite in sign to
  the heuristic prior "buffering argues against essentiality". That is precisely why lever 6 could not be
  rescued by tuning `k_sat`: the prior itself pointed the wrong way, and a data-driven fit corrects it
  without anyone having to notice.

### Lever 6 -- paralog saturation and gating tuning: superseded by lever 5

The weight analysis of lever 5 shows paralog redundancy correlating *positively* with essentiality in this
data, against the prior that buffering weakens it. Scaling `SLC_PARALOG_KSAT` or the saturation curve cannot
fix a prior that is mis-signed, so this lever was not tuned by hand; it is left to the data-driven weights.
Hooking the learned membership model into Vega itself would be an architecture-level change (a serialized
model artefact plus a trace-cache bump) and was not taken for membership -- only for pairwise, §XIII.

> Operational note. `project_knowledge` and `project_graph_computation` are spawned as MCP stdio servers from
> `.venv`. If the runtime -- a restricted sandbox, typically -- **refuses to spawn that server**, Kira returns
> `status=unavailable` for the whole batch, so `kg.evidence = 0 -> pol = 0`, heuristic membership abstains on
> everything and pairwise coverage collapses. That is an **environment** failure, not a decision-logic one.
> The check is `evidence_view.kg_evidence_n`: its mean should be a single-digit count, not zero.

---

## XII. Squeezing the SLC -> Pathway -> Disease chain: disaggregated graph features

This answers a fair question -- "the graph visibly contains SLC-pathway-disease relations; is any of it still
usable?" The answer is yes, but task-dependently, and up to this point only one derived quantity was being
consumed: the composite polarity `gene_disease_effect = rel_AB (x) rel_BC`. The chain was therefore broken
into its parts and the raw path rows persisted, so the increment could be measured rather than asserted.

**Change (no decision logic touched)**: the feature dump gains the disaggregated counts `n_ab_promote` /
`n_ab_suppress`, `n_bc_promote` / `n_bc_suppress`, `n_promote_matched` / `n_suppress_matched` (split over
diseases matched to the queried cancer) and `n_distinct_pathways(_matched)` (diversity of the mediating
pathways). The raw `gene_disease_paths` rows for **every candidate gene of every question** (gene, pathway,
rel_AB, rel_BC, effect, disease) are persisted together with the cancer label, which turns any later
graph-feature experiment into an offline recomputation instead of another pool run. The per-gene pairwise
features of §XIII are rebuilt from exactly those rows.

**Membership -- disaggregation never helps (stable across 5 seeds)**. Logistic on `features`, disjoint qid
split, 10 x 5-fold CV-AUC:

| seed | base (composite polarity + omics) | base + graph (disaggregated) |
|---|---:|---:|
| 5 | 0.758 +/- 0.012 | 0.742 +/- 0.016 |
| 7 | 0.751 +/- 0.013 | 0.741 +/- 0.016 |
| 11 | 0.747 +/- 0.012 | 0.732 +/- 0.011 |
| 23 | 0.623 +/- 0.025 | 0.622 +/- 0.025 |
| 99 | 0.756 +/- 0.015 | 0.736 +/- 0.019 |

The gain is **never positive** (0 to -0.02): the composite polarity already extracts all the
essentiality-relevant direction in this chain, and splitting it further plus pathway diversity adds more
noise than signal at a training point of roughly 81 rows (mild dilution). The learned disaggregated weights
(`n_bc_promote` +0.36, `n_ab_promote` +0.25, `n_distinct_pathways` +0.23) do carry signal -- but it does not
extrapolate. So what the graph chain is worth for membership is the composite polarity already inside
`base`, and that is spent.

**Pairwise -- disaggregation plus a learned pointwise reduction is the decisive gain (stable across 5
seeds)**. "Which of two" is reduced to per-gene scoring (the winner of a question is a positive row, its
loser a negative row; per-gene features use **only** the `kg_paths` graph features), and the winner is the
argmax over the two candidates at test time:

| seed | logistic, winner argmax (all graph features) |
|---|---:|
| 5 | 74.0% (37/50) |
| 7 | 75.9% (44/58) |
| 11 | 79.6% (39/49) |
| 23 | 77.3% (34/44) |
| 99 | 75.6% (34/45) |

Held-out accuracy sits stably at **74-80 % (mean about 76.5 %)**, far above the 50 % dummy and above the
hand heuristic measured on the same seed-5 test set (43 of 50 answered, 60.5 % among answered, 52.0 %
overall). And this uses **the graph chain alone** -- no PPR, TIMER, KM or DE. In other words the
SLC-pathway-disease relations still held a large amount of unconsumed value for *ranking*, and the learned
feature model converted it into accuracy.

---

## XIII. Wiring the pairwise graph-feature logistic into Vega

The cashing-in step of §XII: the heuristic "pick one of two" is **replaced** by the fitted logistic on graph
features, gated by `SLC_PAIRWISE_MODEL`, while the membership branch keeps its rule-based path untouched.
Changing the decision structure invalidates the cached traces, which is exactly what bumping
`SLC_TRACE_VERSION` is for (§XVII).

**Components -- one source of truth, so training and serving cannot drift**

| part | file | note |
|---|---|---|
| feature definition | `knowledge_agents/graph_features.py` | `PAIR_COLS` (the 12 disaggregated graph features), `gene_graph_feat`, `feat_vec`, `collect_path_rows` -- the calibration dump, the offline trainer, the serializer and the online decision all import this one implementation |
| fitting / serialization | `tools/fit_pairwise_model.py` (working tree) | GridSearchCV -> standardized logistic on the qid-hash TRAIN split (seed 5, fraction 0.6; 162 gene rows over 81 questions); exports a JSON artefact carrying `features` / `mu` / `sd` / `coef` / `intercept` / `trained_qids` / `holdout` to `outputs/traces/pairwise_graph_model.json` |
| online inference | `decision_action._pairwise_model` / `_score_pairwise_graph` | **no scikit-learn at serve time**: computes only `(x - mu)/sd -> coef * z + b -> sigmoid`, winner by argmax, confidence = the winner's probability |
| fallback invariant | same, plus the pairwise branch of `_heuristic` | variable unset / artefact unreadable / **neither candidate has any `gene_disease_paths` evidence** -> fall back honestly to the direction + magnitude + buffering rule; a model ranking is never fabricated |

**Consistency verification.** The fitter's holdout argmax is **74.0 % (37/50)**, question-for-question
identical to the §XII offline trainer at the same seed -- same shared feature module, same qid split, so
there is no train/serve skew. `tools/test_pairwise_graph_model.py` passes 7/7, and
`test_paralog_conformal.py` / `test_direction_logic.py` stay green. Those tests pin the runtime gates that
alter the decision path (`SLC_MEM_R2`, `SLC_PAIRWISE_MODEL`) back to their defaults in the file header, so
they report the same result even in a shell that has already dot-sourced `tools/slc_mars_env.ps1`; without
the pinning they would judge the released re-ranking behaviour against bare-heuristic assertions and fail
for the wrong reason.

**Refreshed three-way fair comparison** (n = 40, seed 5, clean pool; the pairwise arm is the deployed
pipeline with the artefact mounted; 40 questions computed fresh, 0 reused). Pairwise (n = 22): 12 correct /
8 wrong / 2 abstained -- 60.0 % among answered, **54.5 % overall**; the auto arm 50.0 % overall; native
MetaGPT answers at 91.7 % accuracy but only covers half the questions (54.5 %), giving 50.0 % overall.
Membership (n = 18) is untouched in this round and stays on its rule-based path (heuristic 16.7 %,
auto 27.8 %, in-harness native MetaGPT 22.2 % overall).

**Leakage split, reported honestly.** The artefact ships its `trained_qids`, so the arm can be partitioned
by whether a question had ever been seen in training:

| pairwise arm, split by qid | n | correct | wrong | abst. | acc (answered) | acc (overall) |
|---|---:|---:|---:|---:|---:|---:|
| seen in training (inside the artefact's 81) | 14 | 6 | 7 | 1 | 46.2% | 42.9% |
| **held out (never seen)** | **8** | **6** | **1** | **1** | **85.7%** | **75.0%** |

The 75 % held-out figure falls inside the 74-80 % band of §XII, and the seen-in-training slice is *lower*
(42.9 %): `C = 0.01` shrinks the coefficients hard, the model barely memorises, so there is no in-sample
bonus on offer -- the 54.5 % overall number is a conservative mixture, not leakage inflation. n is small
(eight held-out questions) and a single seed moves easily; that caveat travels with the number.

> Reproduction lives in the working tree: mount the artefact in `SLC_PAIRWISE_MODEL`, re-run the
> three-way comparison of §XIX, and clear the variable to return the pipeline to the rule-based pairwise
> path.

---

## XIV. The baseline panel: five comparison systems on one footing

One opponent is not a control group. The panel was expanded to **five** systems which, together with the two
pipeline arms, gives **seven systems answering the same questions through the same endpoint under the same
scoring function**. The round that designed the panel ran on the seed-5 clean sample of 40 questions (§XV);
the released numbers apply the identical protocol to the full 489-question pool (§XX).

**Components and isolation (`knowledge_agents/baselines/`)**

| file | system | runs under | essentials |
|---|---|---|---|
| `common.py` | all | any Python 3.11+ | the shared parts: `BASELINE_PROMPT` (copied verbatim from `baseline_metagpt`, with abstention explicitly forbidden so every system is forced to commit), `pending()` (resume: only rows without an error are skipped, so a re-run retries failures), `normalize()` (winner not among the candidates / field missing -> an honest abstention), `llm_settings()` (endpoint resolution: **`SLC_LLM_*` environment variables first**, then the repository `config2.yaml`; an unsubstituted placeholder in the yaml counts as *not configured* -- a placeholder is never sent out as if it were a key) |
| `export_inputs.py` | -- | `.venv` | exports the shared inputs as **two strictly separated files**: `baselines_input.json` (readable by the runners, no ground truth) and `baselines_gt.json` (read only by the scorer, so a runner cannot peek) |
| `run_pure_llm.py` | pure LLM | `.venv` (openai SDK) | `AsyncOpenAI` concurrency; a timeout is recorded honestly as an error row and retried on the next pass |
| `run_lit_rag.py` + `mongo_rag.py` | literature RAG | `.venv` (pymongo) | regex recall over the SLC genes named in the question (cached per gene) -> re-ranked by disease terms (title x3, abstract x1) and gene frequency -> top 5 excerpts pasted into the *same* prompt as the pure-LLM arm; **a retrieval failure never degrades silently into a pure-LLM answer, it abstains**; credentials come only from `MONGO_URI` |
| `run_autogen.py` | AutoGen | `outputs/autogen-venv` (Python 3.11 + pyautogen 0.2.35) | `AssistantAgent` + `UserProxyAgent` in `NEVER` mode (no code execution), the prompt identical to the pure-LLM arm; the last message is usually a bare `TERMINATE`, so the assistant history is scanned backwards for the most recent parseable JSON |
| `run_mdagents.py` | MDAgents | `outputs/mdagents-venv` + a clone of `outputs/MDAgents` | **official source untouched**, monkeypatching only: an `OpenAI` subclass pins the endpoint, model name and a 300 s timeout, `google.generativeai` is stubbed, stdout is reconfigured to UTF-8 (the upstream code prints doctor emoji through `cprint`, which kills a GBK console); the adaptive basic/intermediate/advanced routing is kept exactly as published; `_decision_text` flattens recursively, because an intermediate reply arrives as `{'majority': {0.0: text}}` and `str()` of it yields a single-quoted repr that destroys the JSON |
| `run_metagpt.py` | MetaGPT | `.venv` | the framework's own Role/Action shell around a single `_aask`: same prompt, endpoint and question set as the pure-LLM arm, so the row measures **what the framework shell alone is worth** with no tools and no retrieval. Distinct from the `metaGPT-native` diagnostic row of §XV, which is the same parametric-only idea (`baseline_metagpt.BaselineRole`) evaluated inside `compare_eval`'s harness under a short per-call budget -- neither arm receives this pipeline's evidence view. The released table and the paper's MetaGPT row are the standalone batch run; the two rows are not interchangeable |
| `merge_baselines.py` | aggregation | `.venv` | takes the two pipeline arms (heuristic / auto) straight out of the trace store by trace key, scores the five `answers_<system>.jsonl` files fresh (a qid appearing in history takes its last entry; an error row counts as an abstention), and appends the in-harness native-MetaGPT diagnostic row -> `outputs/traces/baselines_table_master.md` |

**Fairness policy.** All five baselines use "nothing but that LLM's own knowledge" (the RAG arm additionally
gets retrieved literature) -- no knowledge graph, no DESeq2, no conformal artefact. The frameworks' built-in
medical multi-agent logic runs as published: the endpoint is swapped, the semantics of their prompts are not.
Every failure is recorded as an honest abstention; no answer is manufactured.

**Zero-retrieval audit**, answering the obvious suspicion that a high-scoring `pure_llm` is quietly
searching: the pure-LLM arm issues a single completion with no `tools` or `functions` argument; the AutoGen
`AssistantAgent` registers no tool and its `UserProxyAgent` runs with `code_execution_config=False`; the
single exit point of the official MDAgents `Agent.chat()` is a chat completion, and the runner passes
`dataset='custom'` with empty examplers to close its question bank; neither framework venv contains
duckduckgo, googlesearch, tavily or bs4. **The real source of the baseline scores is the combination of
*banned abstention* x *DepMap/CRISPR literature being inside the pretraining corpus (ground truth and model
memory share a source)* x *the negative-bucket prior in this particular sample* -- not retrieval.**

**Install traps, all paid for.** Every pyautogen release requires Python < 3.13 (a 3.14 system interpreter
cannot install it -> the venv is built on a 3.11 base); MDAgents' `utils.py` has a backslash inside an
f-string, which requires Python >= 3.12; openai 1.14.2 is incompatible with httpx >= 0.28 (the `proxies`
keyword argument) -> pin `httpx<0.28`; `AsyncOpenAI` has no `max_connections` parameter -> the concurrency
cap must go through `httpx.AsyncClient(limits=...)`.

---

## XV. The n = 40 diagnostic round, and the memory problem it exposed

> **This section is a diagnostic snapshot, not the released result.** The two tables below are the n = 40
> round whose purpose was to fix the baseline components and the metric conventions. The released full-pool
> numbers (n = 489, plus the anti-memorization subset n = 265) are §XX and
> `outputs/traces/baselines_table_master.md`. The `metaGPT-native` row here is the **in-harness** arm
> (`tracestore.systems.baseline`): the same parametric-only MetaGPT role as the standalone baseline, but run
> inside `compare_eval` with a short per-call budget, so its abstentions are failed or unparseable calls rather
> than refusals -- it never sees the pipeline's evidence view.

Pairwise (n = 22):

| system | correct / wrong / abst. | acc (overall) | acc (answered) |
|---|---|---:|---:|
| pipeline-heuristic (graph-model arm) | 12/8/2 | 54.5% | 60.0% |
| pipeline-auto | 11/9/2 | 50.0% | 55.0% |
| metaGPT-native (in-harness) | 11/1/10 | 50.0% | 91.7% |
| pure-llm | 19/3/0 | **86.4%** | 86.4% |
| lit-rag | 12/10/0 | 54.5% | 54.5% |
| autogen | 19/3/0 | **86.4%** | 86.4% |
| mdagents | 19/3/0 | **86.4%** | 86.4% |

Membership (n = 18):

| system | correct / wrong / abst. | acc (overall) | acc (answered) |
|---|---|---:|---:|
| pipeline-heuristic | 3/5/10 | 16.7% | 37.5% |
| pipeline-auto | 5/6/7 | 27.8% | 45.5% |
| metaGPT-native (in-harness) | 4/0/14 | 22.2% | 100.0% |
| pure-llm | 10/8/0 | 55.6% | 55.6% |
| lit-rag | 9/9/0 | 50.0% | 50.0% |
| autogen | 12/6/0 | 66.7% | 66.7% |
| mdagents | 12/6/0 | 66.7% | 66.7% |

**The four honest readings -- the facts this round exposed.**

1. **Once abstention is banned, the bare LLM prior beats the graph pipeline on this sample.** pure-llm,
   autogen and mdagents all land on the same **19/3 = 86.4 %** overall, above the graph-model arm's 54.5 %.
   The earlier "91.7 % among answered" for native MetaGPT had been diluted by a 45 % abstention rate; under
   the same forced-answer prompt that bonus evaporates and the bare model eats the sample whole -- 17 of the
   22 questions sit in the negative bucket, where the ground-truth winner agrees closely with "how central
   this gene is in the literature", which is precisely what a language model knows. What differentiates the
   pipeline is **calibratable refusal and traceable evidence**, not the raw hit rate on this sample.
2. **Literature RAG actively hurt.** lit-rag's 54.5 % on pairwise is far below pure-llm's 86.4 %: pasted
   excerpts push the model towards the surface statistic "mentioned more often = more essential", and the
   long context slows the endpoint into timeouts. On membership, 50.0 % is likewise below pure-llm's 55.6 %.
3. **On membership, the forced-answer baselines win on the overall metric** (pure-llm 55.6 %,
   autogen = mdagents 66.7 %, against the pipeline's 16.7-27.8 %) because the pipeline abstains under
   conformal pressure while the baselines are forbidden to. Of those 18 questions **10 have ground truth
   `true`**, so "answer true every time" already floors at 55.6 % -- pure-llm sits exactly on the floor
   (10/18), and 66.7 % is barely above it. The two metric conventions must not be mixed: on the
   answered-only basis the pipeline's auto arm is 45.5 % against the baselines' 66.7 %.
4. **Multi-agent frameworks add nothing over the bare model and cost an order of magnitude more.** autogen
   (two roles, no tools, idling) matches pure-llm question for question (19/3); mdagents' adaptive routing,
   with a five-expert debate or a three-member tumour board, ends in exactly the same distribution
   (pairwise 19/3, membership 12/6) at roughly 4-17 minutes per question against the pure LLM's ~30 s. In
   this task shape, coordination complexity did not convert into accuracy.

**What this round bought.** An 86.4 % bare-model score on a benchmark whose labels come from published
CRISPR screens is not evidence of reasoning; the parsimonious explanation is recall. That finding is the
reason the anti-memorization subset exists -- Dataset B, built by the dual-probe procedure in
`tools/label_antimemorization.py` and described in README §2.2 -- and the reason the released result is read
as the **A -> B contrast** (§XX) rather than as an A-only scoreboard.

> Reproduction: the seven-system table is rebuilt from whatever traces and answer files a run leaves behind
> by running `python -m knowledge_agents.baselines.merge_baselines` from the repository root (see §XIX for
> the interpreter setup). MDAgents
> carries two defensive patches, and still zero edits to official source: `_safe_parse_hierarchy` (flatten
> when the hierarchy does not match the `A > B` template -- that tree is display-only) and
> `_RobustGroup.interact` (fall back to a single expert when the member list parses empty). Both only catch
> crashes; neither adds capability.

---

# Part II -- the released configuration (frozen 2026-11-02)

> Everything above this line describes the supply side and the decision layer: tool mapping, the typed
> contract, the evidence invariants, paralog buffering, conformal abstention, the graph features and the
> baseline panel. That is *how the system is built*; it does not depend on which run produced which number.
> Everything below states the released configuration as one decision flow, registers the mechanisms that were
> tried and did not pay off, and gives the reproduction commands and the published results.

## XVI. The released decision flow (the default configuration)

The release = **the evidence base + R2, a membership essentiality re-ranking** -- the one delta over the
base. The entry point is `actions/decision_action.py`; upstream, four supply paths run in parallel inside one
`asyncio.gather` (`answer_pipeline.supply`):

```
   +- Kira  (knowledge retrieval: Neo4j three-path evidence + text2cypher)            \
   +- Cael  (computation: PPR / TIMER / KM / DESeq2 / paralog redundancy)              +-> EvidenceView
   +- L1    (Prior agent: the bare question, zero evidence, literature-memory prior)   /
   +- L2    (second pure-LLM pass: heterogeneous persona, temperature 0.7, independent)
                          |
                          v
            Vega fusion layer   `_fuse` / `_fuse_triad`
   +----------------------------------------------------------------------+
   | evidence agrees with L1   -> 2-0 pass-through (agree note, 0 extra LLM calls) |
   | conflict                  -> cross-critique round (L1/L2 read each other's   |
   |                              arguments and re-vote) -> triad vote -> the      |
   |                              fact-checked brief goes to `adjudicate_multi`    |
   | evidence channel abstains -> evidence-augmented LLM chief (observed values    |
   |                              only, conclusions are never handed to it)        |
   | chief unavailable         -> deterministic majority vote; no majority ->      |
   |                              prior outranks the evidence channel             |
   | every path abstains       -> honest abstention is maintained (never fabricated)|
   +----------------------------------------------------------------------+
```

**Pairwise branch.** The graph-feature logistic artefact of §XIII, gated by `SLC_PAIRWISE_MODEL`: the 12
disaggregated SLC -> Pathway -> Disease features of `graph_features.PAIR_COLS`, winner by argmax. If neither
candidate carries graph path evidence, or the artefact is absent, the code falls back honestly to the
direction + magnitude + buffering rule rather than emitting a model ranking it did not compute.

**Membership branch = R2, the one delta over the base.** The decision construct is restated from "promotion /
suppression polarity" to **CRISPR loss-of-function essentiality**. Literature that both promotes and
suppresses the same gene is no longer read as evidence for fixing a sign; it is read as an essentiality
signal. Five evidence terms are re-weighted, the coefficients fitted by 5-fold cross-validation on the base
configuration over n = 53 membership questions, out-of-sample accuracy 66.8 % against a 60.4 % floor:

| feature | meaning | points to | weight |
|---|---|---|---:|
| `conflict` | the same gene is both promoted and suppressed in the literature | essential | +1.732673 |
| `dir_present` | any single-direction causal evidence | non-essential | -1.733301 |
| `solo` | paralog orphan (no co-expressed partner in the subfamily) | essential | +2.634866 |
| `buffered` | paralog buffering (active co-expression inside the subfamily) | non-essential | -0.619402 |
| `mag = tanh(strength)` | association / differential-expression magnitude | essential | +1.934390 |

`P(essential) = sigmoid(z - 0.408812)`, and `> 0.5` is judged essential. **R2 never abstains**; it alters only
the membership verdict, leaving the pairwise branch and the path taken when the gate is off completely
untouched. Its provenance marker `[R2 essentiality re-ranking]` is written into the rationale.

**Conformal and integrity.** Answers produced by the prior, the debate, the vote or R2 **skip conformal
filtering**: the thresholds are calibrated on the confidence semantics of the evidence channel, so applying
them to a vote or a prior confidence would carry no statistical guarantee. Their origin is labelled
explicitly in `decision_mode` / `basis` / `rationale`, so a memory- or vote-derived answer is never dressed up
as graph or statistical evidence (§IV).

## XVII. Environment knobs (released values pinned in `tools/slc_mars_env.ps1`)

> **Credentials never enter the repository.** `SLC_LLM_BASE_URL` / `SLC_LLM_API_KEY` / `SLC_LLM_MODEL`,
> `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` and `MONGO_URI` / `MONGO_DB_NAME` /
> `MONGO_COLLECTION_PAPERS` are all resolved by `tools/load_secrets.ps1::Import-SlcSecrets`: a variable
> already exported into the environment wins, otherwise the untracked `secrets.local.ps1` at the repository
> root is dot-sourced (template: `secrets.example.ps1`, blocked by `.gitignore`). The tracked
> `config2.yaml` holds placeholders only, and `llm_settings()` treats an unsubstituted placeholder as
> *not configured* -- it would rather fail than send a placeholder out as a key. The machine-local
> `$env:SLC_ROOT` (the root of the SLC_fresh handoff bundle) comes from the same file: it is not a
> credential, but it used to be hard-coded in `run_route_b.ps1` and `tools/slc_mars_env.ps1` with a
> personal directory name in it. MetaGPT itself only reads configuration *files*, so
> `run_metagpt.py` overlays `SLC_LLM_*` onto the framework at run time: `~/.metagpt/config2.yaml` needs no
> key in it, and `tools/sync_metagpt_config.ps1` generates the keyless version. Any key that ever sat in
> plaintext in a tracked file must be treated as exposed and rotated;
> `tools/test_no_secrets.py` is what keeps it that way.

| variable | released value | effect |
|---|---|---|
| `SLC_TRACE_VERSION` | `20` | trace-cache key; bump it only when the decision or evidence structure changes, since that invalidates the cached traces |
| `SLC_PRIOR_AGENT` | `1` | the Prior agent: a third, memory-based evidence channel |
| `SLC_DEBATE` | `1` | expert debate: one extra parallel vote round on ambiguous cases only; it never chairs the decision |
| `SLC_LLM_CHIEF` | `1` | LLM chief judge: retrieval and compute evidence serve as verification material, the verdict is the LLM's |
| `SLC_TRIAD_VOTE` | `1` | triad vote: the evidence channel plus two pure-LLM votes |
| `SLC_TRIAD_CRITIQUE` | `1` | on a split vote, one round of parallel cross-critique first (role-heterogeneous personas) |
| `SLC_PRIOR_RACE` | `2` | fire the prior channel twice and keep the first reply (absorbs endpoint long-tail timeouts) |
| `SLC_MEM_R2` | `1` | **the released delta**: R2 membership essentiality re-ranking |
| `SLC_PAIRWISE_MODEL` | `outputs/traces/pairwise_graph_model.json` | the pairwise graph-feature logistic artefact (§XIII) |
| `SLC_PRIOR_ENSEMBLE` / `SLC_ANSWER_ENSEMBLE` | `1` | set `> 1` to replay the two ensemble ablations, both negative (§XVIII) |
| `SLC_PRIOR_RETRIEVE` | `0` | set `1` to replay literature excerpts as an extra vote source (not adopted) |
| `SLC_NEG_EVID` / `SLC_ABSTAIN_THETA` | `0` | replay the membership negative-evidence channel / confidence-selective abstention (both negative) |
| `SLC_CONFORMAL_FILE` | empty | point it at the threshold json to mount in-flight conformal abstention (§IX) |
| `SLC_PARALOG_KSAT` | `4` | paralog-redundancy saturation constant |
| `SLC_DE_TOPN` | `400` | DESeq2 wide-panel size, which sets how visible paralogs are |
| `SLC_DATASET` | `clean` | the 500-question cleaned pool, of which 489 are the two decision tasks |

## XVIII. Rejected mechanisms (negative results, code kept so they can be replayed)

| mechanism tried | verdict | how to replay |
|---|---|---|
| Ensemble over the memory-prior channel: sample it N times, take the mode | **negative**: A 75.0 -> 72.5, B 68.4 -> 63.2. Ensembling an *intermediate* channel forces the hard questions -- the ones that should have been refused -- into a low-consensus answer; the `agree` gain was a consensus of errors | `SLC_PRIOR_ENSEMBLE=3` |
| Ensemble at the answer layer: N independent end-to-end decisions, mode over answers | **negative**: A 72.5, B 57.9, with the heuristic arm collapsing (B 63.2 -> 26.3). The N samples share one deterministic evidence backbone, so they are not i.i.d. reasoning samples | `SLC_ANSWER_ENSEMBLE=3` |
| Triad vote, first cut: evidence + two pure-LLM votes, second vote at higher temperature, run serially | **mechanism positive, execution negative**: +9.1 pp on the shared subset, but the two extra serial LLM calls per question left 17.5 % of questions timing out and absent | `SLC_TRIAD_VOTE=1` with the vote evaluated serially instead of inside `supply` |
| The voting stack as it ships: L2 moved into the parallel supply stage, role-heterogeneous personas, cross-critique round, double-firing prior | **positive, and it became the base**: 0 of 40 absent; at n = 100 it reaches 70.0 % overall against AutoGen 69.0 and MDAgents 63.0 | the released composition (§XVI) |
| Attaching Mongo literature excerpts to the second vote, to decorrelate it from the first | not adopted: no gain visible across the subsets | `SLC_PRIOR_RETRIEVE=1` |
| Membership negative-evidence channel + confidence-selective abstention | **negative at n = 100**: the lead on some slices was a sampling artefact | `SLC_NEG_EVID=1`, `SLC_ABSTAIN_THETA>0` |
| Literature RAG on the baseline side | **harmful**: pasting raw passages takes pairwise from 86.4 % to 54.5 %, and the long context slows the endpoint into timeouts | `knowledge_agents/baselines/run_lit_rag.py` |
| A fully serial expert debate in the MDAgents style | **zero accuracy gain, order-of-magnitude cost**: 4-17 minutes per question against the pure LLM's ~30 s, for the same score distribution | `knowledge_agents/baselines/run_mdagents.py` |

Two laws that came out of this register and now govern what gets merged:

1. **Ensembling only helps when the members are independent complete reasoning paths and the aggregation
   point is the decision point.** Where a deterministic backbone drives the verdict, majority voting averages
   the correct answers away as readily as the wrong ones -- both ensemble rows above.
2. **A mechanism's gain must be verified on the full sample, under the headline metric.** The "net zero"
   measured at n = 40 was an under-powered instrument: one earlier configuration read 75 % there and fell to
   47 % when the same mechanism was measured at n = 100, i.e. small-sample luck. Conversely, the gain of the
   stack that did ship lies exactly in rescuing answers from honest abstention -- a scenario the n = 40
   sample barely contains.

## XIX. Reproducing the released run

```powershell
# 0) environment and backends ready (§5.1, plus R / Neo4j / Mongo), then pin the released configuration
. tools\slc_mars_env.ps1

# 1) the pipeline arms over the whole clean pool (n = 489, seed 5); a trace-cache hit answers in seconds
.\.venv\Scripts\python.exe -m knowledge_agents.compare_eval --n 500 --seed 5

# 2) same-question baseline inputs (the runner-readable file carries NO ground truth), then the five
#    baselines -- each in its own interpreter, all of them resuming where they stopped
.\.venv\Scripts\python.exe -m knowledge_agents.baselines.export_inputs
.\tools\run_baselines_all500.ps1

# 3) the anti-memorization labels (dual probe; ground truth is read only inside this tool and never
#    enters any answering path)
.\.venv\Scripts\python.exe tools\label_antimemorization.py

# 4) the seven-system master table, built from the records steps 1-3 just wrote
.\.venv\Scripts\python.exe -m knowledge_agents.baselines.merge_baselines
#      -> outputs/traces/baselines_table_master.md

# 5) offline self-checks -- no database, no LLM, no tools: every file in tools\test_*.py, currently 12
Get-ChildItem tools\test_*.py | ForEach-Object {
    .\.venv\Scripts\python.exe -W error::SyntaxWarning $_
    if ($LASTEXITCODE) { "FAILED: $($_.Name)" }
}
```

`tools\test_no_secrets.py` inside that loop is the credential guard: it fails the build if a plaintext key,
password or authority-bearing URI reappears in source, configuration or documentation, and confirms that the
local `secrets.local.ps1` is still blocked by `.gitignore`.

Two limits on what steps 1-4 give a stranger, both stated on purpose:

- **Re-running reproduces the procedure, not the digits.** Endpoint sampling and graph contents drift, which
  is why the frozen tables under `outputs/traces/` -- not an ad-hoc re-run -- are the citable record (§XX).
- **The run record stays with the working tree.** `merge_baselines.py` reads a trace store and the per-system
  `answers_<system>.jsonl` files that a run writes on the machine that ran it; the release ships the derived
  tables instead. The TopK and agent-ablation figures of §XX come from drivers that likewise stayed in the
  working tree; their published values are transcribed in README §4.2-4.3, and the arm-level ranking record
  that does ship is `outputs/traces/topk_eval_summary.txt`.

## XX. Released results (full clean pool)

Dataset A = all 489 decision questions (230 membership / 259 pairwise); dataset B = the 265-question
anti-memorization subset (138 membership / 127 pairwise). Accuracy is the **overall** measure -- an
abstention and a missing answer both count as wrong over a fixed denominator -- and both tasks are binary, so
chance = 50 %. Read directly from `outputs/traces/baselines_table_master.md`.

| system | A membership | A pairwise | B membership | B pairwise |
|---|---:|---:|---:|---:|
| pure-LLM | 56.1% | 70.7% | 27.5% | 40.2% |
| LitRAG | 50.9% | 64.1% | 31.2% | 49.6% |
| AutoGen | 62.6% | 68.3% | 43.5% | 48.0% |
| MDAgents <sup>†</sup> | 33.9% | 27.8% | 34.1% | 21.3% |
| MetaGPT | 60.9% | 71.4% | 42.8% | 51.2% |
| **SLC-MARS heuristic** | **67.8%** | **73.7%** | **56.5%** | 52.0% |
| SLC-MARS auto | 67.4% | 73.0% | 52.2% | **52.8%** |

<sup>†</sup> MDAgents is an incomplete run (coverage about 49 % of A and 51 % of B); unanswered questions are
counted wrong, which is the metric convention of §2.3 in the README.

The decisive reading is the **A -> B contrast**, not the A column: every pure-memory system falls sharply on
the subset built to exclude recall (pure-LLM membership 56.1 % -> 27.5 %, i.e. *below* chance), while
SLC-MARS degrades mildly and remains the only system above chance on B / membership. That is the evidence
that the pipeline is consuming retrievable information the baselines do not have -- and it is the same
finding that §XV turned into the construction of dataset B.

**TopK ranking** (11 CRISPR ranking questions, a separate task with no A/B split): on the graph candidate
pool the released system is the best ranker, NDCG@5 **0.3838** / hit@5 **0.909** for per-gene knowledge
retrieval (the arm-level view of the same experiment, including the RRF fusion variant at 0.370, ships in
`outputs/traces/topk_eval_summary.txt`). Moving to the pool the LLM nominates for itself collapses every
system's NDCG@5 into the 0.12-0.15 band, which says the **graph candidate pool, not the ranker, is what
makes this task work**.

**Agent ablation** (three tasks): dropping Kira takes the overall 73.0 % to 59.0 %, dropping Cael to 65.0 %
-- the two supply agents are complementary; on TopK, knowledge-only is the strongest single variant (0.3838).
Full tables in README §4.2-4.3.

**Honesty note on coverage.** 100 % coverage is the behaviour of the released configuration: R2 never
abstains and the voting stack recovers questions that the earlier arms refused. It is not a coverage
inflation achieved by discarding hard questions -- the conformal abstention machinery of the earlier rounds
remains available and untouched in `conformal.py` / `_conformal_filter`, mounted through
`SLC_CONFORMAL_FILE` (§XVII).
