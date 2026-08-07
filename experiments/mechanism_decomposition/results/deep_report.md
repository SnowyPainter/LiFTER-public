# Deep mechanism decomposition

## Shapley allocation

Values are Historical AP percentage-point contributions across all 128 subsets.

| Dataset | Largest positive component | Shapley AP (%p) |
|---|---|---:|
| Wikipedia | pair_renewal | 14.68 |
| Reddit | pair_renewal | 7.68 |
| Mooc | ordered_transition_2 | 20.14 |
| Lastfm | ordered_transition_1 | 9.70 |
| Retailrocket | candidate_context | 12.02 |

## Full-context pair interactions

Positive values indicate complementarity and negative values indicate overlap or interference after conditioning on the other five components.

| Dataset | Strongest complement | AP interaction (%p) | Strongest interference | AP interaction (%p) |
|---|---|---:|---|---:|
| Wikipedia | pair_renewal + ordered_transition_1 | +4.28 | pair_renewal + positioned_recurrence | -4.92 |
| Reddit | pair_renewal + ordered_transition_1 | +0.71 | pair_renewal + positioned_recurrence | -1.43 |
| Mooc | source_context + ordered_transition_2 | +1.34 | ordered_transition_1 + ordered_transition_2 | -9.27 |
| Lastfm | source_context + ordered_transition_2 | +0.65 | ordered_transition_1 + ordered_transition_2 | -5.10 |
| Retailrocket | source_context + positioned_recurrence | +1.30 | candidate_context + positioned_recurrence | -4.38 |

## Signed decision-margin intervention

Supporting and opposing facts are selected by their signed contribution to the positive-versus-historical-negative margin. The fact database is edited and the complete program is then re-grounded.

| Dataset | Supporting: top / random margin drop | Direction consistency | Opposing: top / random margin gain | Direction consistency |
|---|---:|---:|---:|---:|
| Wikipedia | 3.634 / 2.235 | 99.4% | 1.635 / 0.776 | 99.5% |
| Reddit | 3.778 / 2.893 | 99.5% | 2.655 / 0.747 | 93.9% |
| Mooc | 1.395 / 0.267 | 72.7% | 0.337 / 0.206 | 56.8% |
| Lastfm | 3.241 / 0.394 | 67.4% | 0.633 / -0.040 | 54.7% |
| Retailrocket | 0.691 / 0.418 | 97.5% | 0.097 / 0.043 | 81.6% |

## Grounded fact intervention

The top grounded fact and a random fact from the same history bank and recency quartile are deleted before the full program is re-grounded.

| Dataset | Component | Queries | Top / matched-random logit change | Top / matched-random AP drop (%p) |
|---|---|---:|---:|---:|
| Wikipedia | pair_renewal | 414 | 1.820 / 1.172 | 0.38 / 0.14 |
| Reddit | pair_renewal | 346 | 1.949 / 1.571 | -0.06 / 0.52 |
| Mooc | ordered_transition_2 | 492 | 1.784 / 0.492 | 13.38 / 1.50 |
| Lastfm | ordered_transition_1 | 510 | 4.490 / 0.484 | 9.61 / 0.42 |
| Retailrocket | candidate_context | 324 | 0.390 / 0.228 | 16.43 / 9.48 |

## Query regimes

### Wikipedia

| Regime | Queries | Full AP (%) | Largest deletion effect | AP drop (%p) |
|---|---:|---:|---|---:|
| direct_recurrence_present | 4203 | 94.90 | pair_renewal | 8.18 |
| direct_recurrence_absent | 713 | 40.51 | candidate_context | 0.48 |
| low_source_activity | 2482 | 84.83 | pair_renewal | 5.98 |
| high_source_activity | 2434 | 94.33 | pair_renewal | 10.90 |
| low_destination_popularity | 2475 | 83.58 | pair_renewal | 10.70 |
| high_destination_popularity | 2441 | 94.80 | pair_renewal | 8.47 |
| early_test_interval | 2458 | 90.99 | pair_renewal | 7.96 |
| late_test_interval | 2458 | 88.99 | pair_renewal | 9.92 |

### Reddit

| Regime | Queries | Full AP (%) | Largest deletion effect | AP drop (%p) |
|---|---:|---:|---|---:|
| direct_recurrence_present | 3686 | 92.74 | pair_renewal | 1.09 |
| direct_recurrence_absent | 1230 | 39.93 | candidate_context | 2.55 |
| low_source_activity | 2484 | 79.24 | candidate_context | 2.52 |
| high_source_activity | 2432 | 88.30 | pair_renewal | 3.18 |
| low_destination_popularity | 2460 | 66.48 | pair_renewal | 6.06 |
| high_destination_popularity | 2456 | 95.24 | candidate_context | 1.58 |
| early_test_interval | 2458 | 83.56 | pair_renewal | 2.64 |
| late_test_interval | 2458 | 84.51 | pair_renewal | 2.02 |

### Mooc

| Regime | Queries | Full AP (%) | Largest deletion effect | AP drop (%p) |
|---|---:|---:|---|---:|
| direct_recurrence_present | 2877 | 87.59 | ordered_transition_2 | 5.55 |
| direct_recurrence_absent | 2039 | 80.84 | ordered_transition_2 | 20.91 |
| low_source_activity | 2504 | 80.12 | ordered_transition_2 | 14.11 |
| high_source_activity | 2412 | 90.64 | ordered_transition_2 | 6.55 |
| low_destination_popularity | 2461 | 83.11 | ordered_transition_2 | 15.23 |
| high_destination_popularity | 2455 | 86.71 | ordered_transition_2 | 8.69 |
| early_test_interval | 2458 | 85.53 | ordered_transition_2 | 12.27 |
| late_test_interval | 2458 | 84.67 | ordered_transition_2 | 11.17 |

### Lastfm

| Regime | Queries | Full AP (%) | Largest deletion effect | AP drop (%p) |
|---|---:|---:|---|---:|
| direct_recurrence_present | 3470 | 75.92 | ordered_transition_1 | 4.70 |
| direct_recurrence_absent | 1446 | 60.75 | ordered_transition_1 | 5.69 |
| low_source_activity | 2458 | 71.23 | ordered_transition_1 | 4.96 |
| high_source_activity | 2458 | 73.12 | ordered_transition_1 | 4.93 |
| low_destination_popularity | 2508 | 65.92 | ordered_transition_1 | 4.29 |
| high_destination_popularity | 2408 | 77.61 | ordered_transition_1 | 5.49 |
| early_test_interval | 2458 | 74.04 | ordered_transition_1 | 5.40 |
| late_test_interval | 2458 | 70.16 | ordered_transition_1 | 4.31 |

### Retailrocket

| Regime | Queries | Full AP (%) | Largest deletion effect | AP drop (%p) |
|---|---:|---:|---|---:|
| direct_recurrence_present | 308 | 88.07 | candidate_context | 2.53 |
| direct_recurrence_absent | 19 | 43.35 | ordered_transition_1 | 0.16 |
| low_source_activity | 164 | 89.86 | candidate_context | 1.22 |
| high_source_activity | 163 | 84.57 | candidate_context | 2.15 |
| low_destination_popularity | 172 | 82.98 | candidate_context | 2.42 |
| high_destination_popularity | 155 | 89.28 | candidate_context | 2.36 |
| early_test_interval | 168 | 85.18 | candidate_context | 2.45 |
| late_test_interval | 159 | 87.17 | candidate_context | 2.26 |

