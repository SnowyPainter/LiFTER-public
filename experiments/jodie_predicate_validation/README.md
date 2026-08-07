# JODIE predicate-count validation

This experiment tests whether LiFTER actually uses more than one latent
predicate on Wikipedia, Reddit, MOOC, and LastFM. It compares `K=1,2,4,8` under
the final 32,768-event, 10-epoch benchmark protocol and seeds 7/17/27.

Each trained `K>1` model is evaluated twice: normally and after rotating the
predicate assignments among valid historical facts while preserving each
query's predicate histogram. A useful learned vocabulary should show a
repeatable forecasting gain over `K=1`, differentiated rule roles, and a loss
under this association-breaking intervention.

A second, stronger checkpoint-only intervention rotates predicates across the
complete evaluation batch. It is run sequentially because the benchmark's
historical-negative sampler deliberately uses a reset process-global NumPy RNG.

```bash
/opt/conda/bin/python experiments/jodie_predicate_validation/run_experiment.py --workers 3
```
