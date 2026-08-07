# Latent mechanism recovery from one observed relation

This controlled experiment tests whether future-link supervision can divide a
single observed `Link(source, destination, time)` relation into reusable latent
fact roles. The generator mixes recurrence, transition, and exploration
mechanisms while exposing neither their names nor their fact-level type labels
to the model.

Every predictor receives the same noisy continuous features, sampled so that
hidden roles leave statistically distinct but overlapping observational traces,
and the same structural rule-template identifier. These are synthetic features,
not causal statistics measured from JODIE. Entity identities are disjoint
between training and test and are never model inputs. Positive and negative
examples are balanced within every template. Hidden types are opened only after
training to compute permutation-matched recovery accuracy, NMI, and ARI.

The four controls are a single predicate, random fixed predicates, unsupervised
k-means predicates, and predicates learned jointly with executable rule weights
from the future-link objective.

Run:

```bash
/opt/conda/bin/python experiments/latent_mechanism_recovery/run_experiment.py
```

The output is written to `results/summary.json`.
