# Example traces (translated, from the frozen release run)

Two complete per-question records of the released configuration, one per decision task. They show
the message shapes that the design documents describe: what each agent was asked, what each supply
side actually returned, and how the decision agent turned that into a verdict.

The full run record (`outputs/traces/tracestore.jsonl`, 489 questions, plus one view file per
question) is a single machine's log of endpoint calls and does not ship. These two files are the
part of it worth reading.

| file | task | what it shows |
|---|---|---|
| [`membership_negative_ALL_Human_NALM6_36Days_00035.json`](membership_negative_ALL_Human_NALM6_36Days_00035.json) | membership | the essentiality re-ranking path: four knowledge channels and one computation channel, the logistic score `P(essential) = 0.427`, the resulting `non-essential` verdict (ground truth `false`), and the triad short-circuit that skips the vote when two channels already agree |
| [`pairwise_negative_Lymphoma_Mouse_ATM-null_DLBCL_18days_00263.json`](pairwise_negative_Lymphoma_Mouse_ATM-null_DLBCL_18days_00263.json) | pairwise | the honest-abstention path: the evidence channel finds nothing above the floor and abstains, the triad vote with cross-critique then recovers a winner (`SLC33A1`, ground truth `SLC33A1`); the same record also carries a failed stock-MetaGPT in-harness call, logged as a timeout rather than as an answer |

## How to read one record

| key | content |
|---|---|
| `qid`, `bucket`, `task_type` | the sample-manifest identity: task, polarity of the ground-truth label, cancer, species, cell line, treatment, screen |
| `question`, `constraints` | the question text as posed, and the structured conditions parsed out of it |
| `ground_truth` | the published label (`evaluation/ground_truth.jsonl`), kept here for readability; it is never on any answering path |
| `effective_genes` | the candidate genes the question is actually decided over |
| `supply.kg` | what the knowledge-retrieval agent returned: one entry per evidence channel under `evidence`, each with its `cypher`, its rows and its own status; plus `schema_used`, `public_web_used` and `source_independence` |
| `supply.ck` | what the computing agent returned: `graph_computation` (PPR), `expression` (TIMER), `survival` (KM), `differential_expression` (DESeq2), the paralog channels, and `skipped_tools` listing why each unused tool stayed unused |
| `systems.pipeline_heuristic` | the deterministic decision over exactly the `supply` view above |
| `systems.pipeline_auto` | the same evidence view, with the cross-cancer analogy and the triad vote on top |
| `systems.baseline` | the stock-MetaGPT arm evaluated in the same harness: parametric knowledge only, no tools, no retrieval, and never handed the evidence view |
| `provenance` | dataset tag, trace-cache key and timestamp of the frozen run |

No field was tuned for presentation: `basis` lists the channels that fired,
`decision_mode` names the path taken, `confidence` is what the conformal layer would threshold on,
and `abstained` distinguishes "no evidence" from a negative call.

## Translation policy

The frozen run wrote its agent text in Chinese. In these two files:

- every string that a **rule template** produced is rebuilt from the English template that the
  shipped code emits today, with the same numbers substituted, so a translated record reads
  exactly like a live run of this code;
- the one **free-text LLM passage** that came back in Chinese is translated literally;
- the iteration digits that the frozen labels carried inside them (the suffixes on the
  re-ranking, vote and critique tags) are dropped, because the shipped code emits the same tags
  without them;
- **no number, gene symbol, status flag, verdict or confidence was altered** — the only digits
  removed anywhere are those label suffixes.

The remaining 487 records were produced by the same run and follow the same shapes.
