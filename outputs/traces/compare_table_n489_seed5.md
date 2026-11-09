[Fair side-by-side comparison] coverage = answered / total | accuracy (answered) = correct / answered |
accuracy (all) = correct / total, an abstention counting as wrong | edge over chance = accuracy (all) - 50%
Sample: n=489 seed=5 dataset=clean buckets=['pairwise', 'membership'] (107 computed fresh / 382 served from the trace cache)

### pairwise  (n=259)
| system | N | correct | wrong | abst. | coverage | accuracy (answered) | accuracy (all) | edge over chance |
|------|----:|---:|---:|---:|------:|--------:|--------:|--------:|
| pipeline (heuristic)     | 259 | 191 |  68 |   0 | 100.0% |   73.7% |   73.7% |   +23.7% |
| pipeline (auto, analogy)      | 259 | 189 |  70 |   0 | 100.0% |   73.0% |   73.0% |   +23.0% |
| native MetaGPT        | 259 | 118 |  28 | 113 |  56.4% |   80.8% |   45.6% |    -4.4% |

### membership  (n=230)
| system | N | correct | wrong | abst. | coverage | accuracy (answered) | accuracy (all) | edge over chance |
|------|----:|---:|---:|---:|------:|--------:|--------:|--------:|
| pipeline (heuristic)     | 230 | 156 |  74 |   0 | 100.0% |   67.8% |   67.8% |   +17.8% |
| pipeline (auto, analogy)      | 230 | 155 |  75 |   0 | 100.0% |   67.4% |   67.4% |   +17.4% |
| native MetaGPT        | 230 |  55 |  34 | 141 |  38.7% |   61.8% |   23.9% |   -26.1% |

### OVERALL  (n=489)
| system | N | correct | wrong | abst. | coverage | accuracy (answered) | accuracy (all) | edge over chance |
|------|----:|---:|---:|---:|------:|--------:|--------:|--------:|
| pipeline (heuristic)     | 489 | 347 | 142 |   0 | 100.0% |   71.0% |   71.0% |   +21.0% |
| pipeline (auto, analogy)      | 489 | 344 | 145 |   0 | 100.0% |   70.3% |   70.3% |   +20.3% |
| native MetaGPT        | 489 | 173 |  62 | 254 |  48.1% |   73.6% |   35.4% |   -14.6% |