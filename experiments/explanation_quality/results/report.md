# Explanation quality with a shared evidence universe

Wikipedia, Reddit, MOOC, and LastFM use 256 chronological queries per seed,
uniform-historical alternatives, and seeds 7/17/27. Every method receives the
same query-level bank: the latest ten source events and latest ten candidate
events. Ratio budgets from 5% to 30% and fixed event counts
`k = {1, 2, 3, 5, 10}` are both evaluated; here `k` is the number of selected
explanation events, not LiFTER's predicate count `K`. T-GNNExplainer is the actual
explorer–navigator MCTS over a frozen, causality-audited TGN; it is not the
former singleton-deletion proxy.

| Model | Base AP | ACC-AUC | Deletion AUFSC | Available / selected (%) | ms/query | Stability |
|---|---:|---:|---:|---:|---:|---:|
| **LiFTER** | **0.8095** | **0.8671** | **0.0552** | 17.59 / 3.22 (18.93) | 0.0241 | **0.9901** |
| TGN + T-GNNExplainer | 0.7783 | 0.8158 | 0.0276 | 17.59 / 3.22 (18.93) | 91.5325 | 0.5283 |
| TGN + TempME | 0.7783 | 0.6533 | 0.0043 | 17.59 / 3.22 (18.93) | 0.0060 | 0.8891 |
| TGIB | 0.7626 | 0.8034 | 0.0431 | 17.59 / 3.22 (18.93) | **0.0046** | 0.9558 |
| SIG | 0.7478 | 0.8159 | 0.0011 | 17.59 / 3.22 (18.93) | 0.0081 | 0.9555 |

| Model | ACC@1 | ACC@2 | ACC@3 | ACC@5 | ACC@10 |
|---|---:|---:|---:|---:|---:|
| **LiFTER** | **0.7578** | **0.8760** | **0.8774** | **0.8968** | 0.9316 |
| TGN + T-GNNExplainer | 0.7316 | 0.7840 | 0.8175 | 0.8600 | 0.9263 |
| TGN + TempME | 0.6479 | 0.6491 | 0.6519 | 0.6745 | 0.7324 |
| TGIB | 0.7388 | 0.7744 | 0.8040 | 0.8441 | 0.9064 |
| SIG | 0.6987 | 0.7435 | 0.8296 | 0.8766 | **0.9508** |

| Model | Deletion FID@1 | FID@2 | FID@3 | FID@5 | FID@10 |
|---|---:|---:|---:|---:|---:|
| **LiFTER** | **0.0361** | **0.0539** | **0.0544** | **0.0636** | 0.1224 |
| TGN + T-GNNExplainer | 0.0151 | 0.0234 | 0.0266 | 0.0350 | 0.0604 |
| TGN + TempME | -0.0010 | 0.0012 | 0.0051 | 0.0095 | **0.1324** |
| TGIB | 0.0159 | 0.0298 | 0.0393 | 0.0601 | 0.1034 |
| SIG | -0.0027 | -0.0002 | 0.0027 | 0.0055 | 0.0363 |

The complete per-dataset, per-seed ratio and fixed-budget curves are stored in
`{dataset}_summary.json`. Identical available counts, selected counts, and
selected fractions across methods show that the result is not caused by an
evidence-universe or budget mismatch.
