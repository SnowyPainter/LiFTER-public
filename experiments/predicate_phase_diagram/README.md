# Predicate phase diagram

This controlled experiment varies semantic effect `alpha` and pre-event-context
identifiability `beta`. Every event is observed as the same `Link` relation.
K=1 and learned K=3 programs receive identical facts, templates, labels, and
disjoint train/test entities. Oracle hidden-type and fixed random-type controls
separate learnability from the mere capacity increase of a larger vocabulary.

```bash
/opt/conda/bin/python experiments/predicate_phase_diagram/run_experiment.py
/opt/conda/bin/python experiments/predicate_phase_diagram/plot_heatmaps.py
```

The default run evaluates the complete 5x5 grid with seeds 7/17/27, 6,000
training queries, 3,000 test queries, and 40 epochs. Type recovery is reported
as Hungarian permutation-matched accuracy together with NMI and ARI. Use
`--augment-controls` to add oracle/random controls to an existing learned grid
without retraining the latter.
