# LiFTER mechanism decomposition

This experiment treats a frozen LiFTER checkpoint as an additive executable
program. For each historical-negative query it records seven score components,
evaluates each component alone, and removes each component from the complete
logit without retraining. It additionally evaluates all 128 component subsets,
computes exact Shapley allocations for Historical AUC/AP, and reports deletion
effects across recurrence, activity, popularity, and temporal query regimes.
For the dominant component on each dataset, it also deletes the highest-scoring
grounded historical fact from the available history prefix, re-executes the full program,
and compares the change with deletion of a fact from the same history bank and
recency quartile.

Datasets: Wikipedia, Reddit, MOOC, and LastFM.
Seeds: 7, 17, and 27.

```bash
/opt/conda/bin/python experiments/mechanism_decomposition/run_experiment.py
```

The runner accepts `--workers`, `--progress`, and `--force`.

The decomposition is exact at the logit level. Performance drops are diagnostic
effects on nonlinear AUC/AP and therefore are not expected to add.
