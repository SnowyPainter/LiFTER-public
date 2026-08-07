# RetailRocket predicate identifiability diagnosis

The experiment uses the same chronological 262,144-event transaction-query
task and seeds 7/17/27 as the predicate-count sweep. All historical events remain
available to both models. The oracle condition changes only fact typing: the
recorded `view`, `addtocart`, and `transaction` action is supplied as a fixed
three-predicate assignment. It is an audit upper bound, not a forecasting model.

## Forecasting with exact action predicates

| Typing | Historical AUC | Historical AP | Pairwise accuracy |
|---|---:|---:|---:|
| Single predicate (K=1) | 0.8555 ± 0.0015 | 0.8602 ± 0.0031 | 0.8532 ± 0.0053 |
| Fixed true action predicate (oracle K=3) | 0.8594 ± 0.0003 | 0.8664 ± 0.0053 | 0.8491 ± 0.0145 |

The oracle changes mean AUC by +0.0039, AP by
+0.0062, and pairwise accuracy by
-0.0041. Exact action typing therefore provides
only a small forecasting gain on this task.

## Decodability from LiFTER's causal fact context

| Probe metric | Result | Chance/reference |
|---|---:|---:|
| Balanced accuracy | 0.7679 ± 0.0035 | 0.3333 |
| NMI | 0.0975 ± 0.0199 | 0.0000 |
| ARI | 0.1524 ± 0.0573 | 0.0000 |

A supervised diagnostic probe can recover the hidden current action substantially
above chance from the same strictly earlier structural context available to the
fact encoder. Thus the input contains action-correlated information. The ordinary
future-link objective nevertheless does not organize K>1 predicates around these
actions, as established by the separate K sweep and predicate-shuffle test.

## Decision

The observed collapse is partly an objective-identification issue, but it is not
the main forecasting bottleneck: an exact, semantically correct predicate supplies
only +0.0062 historical AP. RetailRocket therefore
does not support redesigning LiFTER around action recovery merely to improve link
forecasting. A stronger predicate-learning claim requires a task in which latent
mechanisms affect the target conditionally and exact typed rules materially beat
the single-predicate program.
