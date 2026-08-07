# Grounding-budget validation

This experiment selects one global LiFTER grounding capacity without reading the
final 15% test interval. From each 32,768-event sequence, the first 85% forms a
development prefix; the last 15% of that prefix is chronological validation.

The predeclared candidates are `H = {10, 20, 40, 80}`. Every candidate uses the
same four datasets, epochs, and seeds. Selection maximizes macro-mean validation
Historical AUC, with the smaller horizon breaking an exact tie. The selected
value is then shared by all datasets in the main benchmark.

Run the complete protocol with:

```bash
/opt/conda/bin/python experiments/grounding_budget_validation/run_validation.py --progress
```

Raw cells are written to `outputs/`; the auditable selection result is written
to `results/summary.json`.
