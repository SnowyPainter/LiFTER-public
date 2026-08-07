# Full-scale efficiency and scalability

Measurements use an NVIDIA RTX PRO 6000 Blackwell. The causal fact index now
keeps only the latest context-capacity rows per endpoint; an equivalence test
against the former full-history construction has zero numerical difference.

| Events | Nodes | Index time | Index throughput | Fact database |
|---:|---:|---:|---:|---:|
| 32,768 | 9,146 | 1.05 s | 31.2k events/s | 7.5 MiB |
| 131,072 | 10,032 | 4.15 s | 31.5k events/s | 30.0 MiB |
| 524,288 | 10,917 | 16.68 s | 31.4k events/s | 120.0 MiB |
| 672,447 | 10,984 | 21.63 s | 31.1k events/s | 153.9 MiB |

Construction time and stored fact payload are linear in stream length. Query
execution does not scan this global stream: with batch 128, H=10 takes 0.041
ms/query at K=1 and 0.053 ms/query at K=2. At fixed batch 32 and K=8, raising
H from 10 to 80 changes latency only from 0.197 to 0.197 ms/query because the
executor dispatches a small number of valid groundings rather than enumerating
the stream.

Batching increases throughput from 7.1k queries/s at batch 32 to 101.6k at
batch 2,048. Transition ranks 8, 16, 32, and 64 all remain near 24.1k
queries/s at batch 128. Candidate batching reaches 92.7k candidate scores/s
with 16 candidates per query.

Training throughput is 38.8k events/s (0.845 s per 32,768-event epoch) with
72.8 MiB peak allocated GPU memory for the measured executor batch. Prediction takes 0.137 ms/query. Editing a
fact mask and re-executing the complete program takes 0.138 ms/query. Materializing
the human-readable JSON certificate raises total time to 1.185 ms/query and
produces a mean 6.1 KiB trace; the cost is serialization rather than a second
explanation optimization.

The current vectorized dense renewal kernel is slightly faster than the sparse
dispatch at H up to 80 (about 6.0 versus 6.1 ms/batch). Therefore the empirical
result supports bounded local execution and linear database scaling, but not a
runtime advantage for sparse renewal at these capacities.

## Cross-model comparison

Forward latency was measured with the same recent-10 local-history tensors at
each batch size. This is an architecture-level timing comparison; parameter
values do not change the executed tensor operations, and these measurements are
not used as forecasting-quality results.

| Model | Throughput at batch 512 (queries/s) |
|---|---:|
| TGN | 233,998 |
| TGAT | 487,004 |
| DyGFormer | 299,264 |
| GraphMixer | 325,320 |
| CRAFT | 455,653 |
| LiFTER | 66,901 |

LiFTER remains slower than the neural predictors because it materializes and
scores grounded executions, while still processing 66.9k queries/s. Explanation
latency uses the trained models and common evidence bank from the explanation
evaluation: LiFTER takes 0.024 ms/query, compared with 91.533 ms/query for the
search-based T-GNNExplainer, 0.006 for TempME, 0.005 for TGIB, and 0.008 for SIG.
Thus intrinsic execution removes post-hoc search overhead, but it is not faster
than every built-in or feed-forward attribution method.
