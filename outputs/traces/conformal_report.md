[Conformal abstention] the calibration and test splits are disjoint by qid; coverage takes the whole
test set as its denominator (the baseline's own abstentions included); the guarantee is
E[wrong AND selected] <= alpha, hence the error rate among answered questions <= alpha / coverage.

### system: pipeline_heuristic

- pairwise  calibration n=59 / test n=61 (48 answered, the baseline itself abstained on 13)
  baseline (accept everything): coverage=78.7%  error rate among answered=41.7%
  conformal (alpha=0.1, tau_hat=0.400 -> conf >= 0.6): coverage=29.5%  error rate among answered=33.3% [wrong/total=6/61]
  risk-coverage curve (self-consistent on the calibration split; the monotone trade-off):
  | alpha (target, whole-set basis) | tau_hat | coverage | error rate (answered) | wrong/total |
  |---:|---:|---:|---:|---:|
  | 0.00 | n/a | 0.0% | 0.0% | 0/59 |
  | 0.05 | n/a | 0.0% | 0.0% | 0/59 |
  | 0.10 | 0.400 | 35.6% | 23.8% | 5/59 |
  | 0.15 | 0.405 | 49.2% | 27.6% | 8/59 |
  | 0.20 | 0.474 | 62.7% | 29.7% | 11/59 |
  | 0.30 | 0.801 | 84.7% | 34.0% | 17/59 |
  | 0.40 | 0.881 | 100.0% | 39.0% | 23/59 |
  | 0.50 | 0.881 | 100.0% | 39.0% | 23/59 |
  ECE (test, answered only; confidence vs. realized accuracy)=0.194

- membership  calibration n=22 / test n=60 (26 answered, the baseline itself abstained on 34)
  baseline (accept everything): coverage=43.3%  error rate among answered=57.7%
  conformal (alpha=0.1, tau_hat=n/a -> conf >= n/a): coverage=0.0%  error rate among answered=0.0% [wrong/total=0/60]
  risk-coverage curve (self-consistent on the calibration split; the monotone trade-off):
  | alpha (target, whole-set basis) | tau_hat | coverage | error rate (answered) | wrong/total |
  |---:|---:|---:|---:|---:|
  | 0.00 | n/a | 0.0% | 0.0% | 0/22 |
  | 0.05 | n/a | 0.0% | 0.0% | 0/22 |
  | 0.10 | n/a | 0.0% | 0.0% | 0/22 |
  | 0.15 | n/a | 0.0% | 0.0% | 0/22 |
  | 0.20 | n/a | 0.0% | 0.0% | 0/22 |
  | 0.30 | n/a | 0.0% | 0.0% | 0/22 |
  | 0.40 | 0.000 | 59.1% | 61.5% | 8/22 |
  | 0.50 | 0.050 | 68.2% | 66.7% | 10/22 |
  ECE (test, answered only; confidence vs. realized accuracy)=0.250
