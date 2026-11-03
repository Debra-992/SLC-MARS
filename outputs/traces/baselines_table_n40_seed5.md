# Baseline 扩展公平对比（同 v10 40 题 · seed5 · 同一 _score 口径）

## pairwise（n=22）

| 系统 | ✓/✗/abstain | acc(全体) | acc(已答) |
|---|---|---|---|
| pipeline-heuristic | ✓12 ✗8 ab2 | 54.5% | 60.0% |
| pipeline-auto | ✓11 ✗9 ab2 | 50.0% | 55.0% |
| metaGPT-native | ✓11 ✗1 ab10 | 50.0% | 91.7% |
| pure-llm | ✓19 ✗3 ab0 | 86.4% | 86.4% |
| lit-rag | ✓12 ✗10 ab0 | 54.5% | 54.5% |
| autogen | ✓19 ✗3 ab0 | 86.4% | 86.4% |
| mdagents | ✓19 ✗3 ab0 | 86.4% | 86.4% |

## membership（n=18）

| 系统 | ✓/✗/abstain | acc(全体) | acc(已答) |
|---|---|---|---|
| pipeline-heuristic | ✓3 ✗5 ab10 | 16.7% | 37.5% |
| pipeline-auto | ✓5 ✗6 ab7 | 27.8% | 45.5% |
| metaGPT-native | ✓4 ✗0 ab14 | 22.2% | 100.0% |
| pure-llm | ✓10 ✗8 ab0 | 55.6% | 55.6% |
| lit-rag | ✓9 ✗9 ab0 | 50.0% | 50.0% |
| autogen | ✓12 ✗6 ab0 | 66.7% | 66.7% |
| mdagents | ✓12 ✗6 ab0 | 66.7% | 66.7% |

## 运行健康度（answers_*.jsonl 中含 error 的题数）
- pure-llm: 已跑 40/40，其中 error 0
- lit-rag: 已跑 40/40，其中 error 0
- autogen: 已跑 40/40，其中 error 0
- mdagents: 已跑 40/40，其中 error 0
