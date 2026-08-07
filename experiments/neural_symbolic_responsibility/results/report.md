# Neural-symbolic responsibility

The full-scale evaluation uses 32,768 events, 10 epochs, uniform historical
alternatives, and seeds 7, 17, and 27. Every ablation is retrained rather than
disabled only at test time.

| Dataset | Variant | Historical AUC | Historical AP |
|---|---|---:|---:|
| MOOC | Full LiFTER (benchmark checkpoint) | 0.8674 | 0.8508 |
| | No exact binding | 0.8631 | 0.8317 |
| | Randomized binding schema | 0.8615 | 0.8285 |
| | No renewal | 0.8666 | 0.8486 |
| | No transitions | 0.5872 | 0.6203 |
| | Transition program only | **0.8706** | **0.8525** |
| | Symbolic clauses without transitions | 0.5872 | 0.6203 |
| | Same-input matched MLP | **0.8724** | 0.8502 |
| LastFM | **Full LiFTER (benchmark checkpoint)** | **0.7067** | **0.7187** |
| | No exact binding | 0.6941 | 0.7077 |
| | Randomized binding schema | 0.6928 | 0.7076 |
| | No renewal | 0.7001 | 0.7095 |
| | No transitions | 0.5832 | 0.6254 |
| | Transition program only | 0.6875 | 0.7026 |
| | Symbolic clauses without transitions | 0.5832 | 0.6254 |
| | Same-input matched MLP | 0.6578 | 0.6527 |

The full program is necessary on LastFM: exact binding, renewal, and the
non-transition clauses each add measurable AP beyond the learned transition
program, and Full LiFTER exceeds the same-input MLP by 6.60 percentage points.
MOOC is transition-dominated: the transition-only program matches the full
model, although replacing the prescribed binding schema lowers AP by 2.22
points. This dataset dependence is precisely why the components are reported
as responsibilities rather than treated as uniformly beneficial modules.

The 8,192-event pilot reaches a different ordering, recorded separately in
`pilot_summary.json`. It must therefore not be used as a proxy for full-scale
architecture selection.
