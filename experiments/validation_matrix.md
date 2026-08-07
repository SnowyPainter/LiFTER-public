# LiFTER validation matrix

| Question | Experiment | Completed evidence |
|---|---|---|
| Are intrinsic explanations competitive with CTDG explainers? | `explanation_quality` | 4 datasets, 5 methods, 256 queries/seed, 3 seeds, 6 ratio budgets |
| Does the recorded execution reconstruct the prediction? | `execution_certificate` | all 19,664 test traces, maximum logit error 1.31e-5 at tolerance 2e-5 |
| Do cited facts control the signed candidate decision? | `mechanism_decomposition` | 128 mechanism coalitions, query regimes, signed matched-fact deletion |
| When is K>1 semantically useful? | `predicate_phase_diagram` | 5x5 alpha-beta grid, 3 seeds, shuffle and type recovery |
| What do neural and symbolic components contribute? | `neural_symbolic_responsibility` | eight variants, two datasets, pilot and full-scale training |
| Does execution scale beyond 32k events? | `scalability` | 32k/128k/512k/full index, H/K/rank/batch/candidate sweeps |

All runners use `/opt/conda/bin/python`. Machine-readable results are stored in
each experiment's `results/` directory, with a concise `results/report.md`
beside them.
