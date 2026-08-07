# Predicate necessity

This experiment isolates the role of LiFTER's latent fact typing while keeping
the grounded executor, temporal rule schemas, training objective, data split,
and optimization budget fixed.

Variants:

- `single_type`: every fact has the same predicate (`K=1`).
- `random_types`: a frozen random projection assigns one of eight predicates.
- `kmeans_types`: training-prefix fact features are clustered without labels.
- `learned_soft`: future-link supervision learns an eight-type soft mixture.
- `learned_hard`: future-link supervision learns one discrete type per fact.

The experiment reports forecasting, exact logit reconstruction, program
concentration, and predicate-to-rule-role reuse across entity novelty, later
time, source activity, and destination popularity strata. Predicate labels are
aligned across seeds by maximum-weight matching of their rule-role profiles.

Run:

```bash
/opt/conda/bin/python experiments/predicate_necessity/run_experiment.py
```

The complete interpretation is in [`results/report.md`](results/report.md), and
the machine-readable aggregate is in [`results/summary.json`](results/summary.json).
