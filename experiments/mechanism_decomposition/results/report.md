# Mechanism decomposition

All values use the frozen K=1 checkpoints and the same uniform-historical
candidates as the forecasting benchmark. `Only AP` retains the prior and one
mechanism. `Deletion AP drop` subtracts that mechanism's exact signed logit
contribution from the full prediction; positive values indicate necessity.
Metric drops need not add because AUC and AP are nonlinear ranking metrics.

| Dataset | Full AUC/AP (%) | Largest AP deletion effect | AP drop (%p) |
|---|---:|---|---:|
| Wikipedia | 89.23 / 89.66 | Pair renewal | 8.62 |
| Reddit | 83.60 / 82.87 | Pair renewal | 1.23 |
| Mooc | 86.74 / 85.08 | Two-event transition | 11.76 |
| Lastfm | 70.67 / 71.87 | One-event transition | 4.69 |
| Retailrocket | 85.55 / 86.02 | Candidate context | 2.32 |

## Complete decomposition

| Dataset | Mechanism | Only AUC/AP (%) | Deletion AUC/AP drop (%p) | Mean logit margin |
|---|---|---:|---:|---:|
| Wikipedia | Direct pair | 85.40 / 86.09 | +0.19 / +0.19 | +0.3299 |
| Wikipedia | Source context | 50.00 / 50.00 | +0.81 / +0.63 | +0.0000 |
| Wikipedia | Candidate context | 85.08 / 86.17 | +0.96 / +0.77 | +1.1319 |
| Wikipedia | Pair renewal | 83.24 / 84.14 | +4.07 / +8.62 | +9.1602 |
| Wikipedia | Positioned recurrence | 86.30 / 87.78 | +0.18 / +0.04 | +1.3265 |
| Wikipedia | One-event transition | 76.57 / 72.36 | -1.42 / -1.39 | +3.1972 |
| Wikipedia | Two-event transition | 66.38 / 58.70 | -0.21 / -0.46 | +3.0845 |
| Reddit | Direct pair | 69.77 / 65.75 | -0.11 / -0.13 | +0.1466 |
| Reddit | Source context | 50.00 / 50.00 | +0.51 / +0.71 | +0.0000 |
| Reddit | Candidate context | 78.53 / 78.48 | +1.06 / +0.71 | +1.6808 |
| Reddit | Pair renewal | 74.24 / 74.76 | +0.67 / +1.23 | +5.6258 |
| Reddit | Positioned recurrence | 75.55 / 78.53 | -0.14 / -0.20 | +1.0124 |
| Reddit | One-event transition | 78.43 / 73.66 | -0.25 / -1.23 | +5.1509 |
| Reddit | Two-event transition | 76.89 / 76.00 | +0.14 / +0.04 | +4.6458 |
| Mooc | Direct pair | 55.59 / 60.75 | -0.58 / -0.93 | +0.0729 |
| Mooc | Source context | 50.00 / 50.00 | -0.22 / -0.54 | +0.0000 |
| Mooc | Candidate context | 59.17 / 61.87 | -0.19 / -0.19 | +0.1653 |
| Mooc | Pair renewal | 45.56 / 47.73 | +0.07 / +0.16 | -0.0213 |
| Mooc | Positioned recurrence | 52.90 / 56.55 | -0.77 / -1.31 | +0.0779 |
| Mooc | One-event transition | 80.57 / 78.01 | +2.99 / +2.71 | +1.3964 |
| Mooc | Two-event transition | 86.11 / 85.48 | +8.58 / +11.76 | +2.6943 |
| Lastfm | Direct pair | 57.99 / 56.83 | +0.04 / -0.04 | +0.0480 |
| Lastfm | Source context | 50.00 / 50.00 | +0.88 / +0.12 | +0.0000 |
| Lastfm | Candidate context | 56.74 / 59.77 | +0.02 / -0.01 | +0.0377 |
| Lastfm | Pair renewal | 56.57 / 56.82 | +0.61 / +0.71 | +0.8041 |
| Lastfm | Positioned recurrence | 58.32 / 59.64 | +0.21 / +0.04 | +0.3077 |
| Lastfm | One-event transition | 68.11 / 69.65 | +5.69 / +4.69 | +2.2907 |
| Lastfm | Two-event transition | 61.62 / 64.68 | +0.26 / +1.12 | +1.9698 |
| Retailrocket | Direct pair | 85.24 / 84.20 | +0.89 / +0.89 | +0.0500 |
| Retailrocket | Source context | 50.00 / 50.00 | -1.31 / -0.77 | +0.0000 |
| Retailrocket | Candidate context | 87.66 / 88.17 | +2.56 / +2.32 | +0.1192 |
| Retailrocket | Pair renewal | 69.25 / 69.14 | -1.01 / -1.94 | +0.3164 |
| Retailrocket | Positioned recurrence | 85.56 / 85.67 | +1.93 / +0.86 | +0.2417 |
| Retailrocket | One-event transition | 56.01 / 60.82 | +0.14 / +0.11 | +0.0070 |
| Retailrocket | Two-event transition | 55.44 / 59.38 | +0.00 / +0.01 | +0.0004 |
