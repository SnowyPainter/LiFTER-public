# Explanation quality

Wikipedia, Reddit, MOOC, and LastFM comparisons over a shared 256-query per
dataset historical-negative sample. Every method receives the same query-level
evidence bank: the latest ten source events and latest ten destination events.
The evaluation uses both the explanation-ratio grid 5–30% and absolute budgets
`k = {1, 2, 3, 5, 10}`. It reports the available evidence count, selected count,
selected fraction, ACC-AUC, deletion AUFSC, sufficiency, necessity, runtime,
and perturbation stability.

The T-GNNExplainer comparison runs the explorer–navigator MCTS on a frozen TGN.
The navigator is trained from TGN deletion rewards and only orders candidate
removals; the frozen TGN evaluates every coalition reward used by the explorer.
It is not the former exhaustive singleton-deletion proxy.

```bash
for dataset in wikipedia reddit mooc lastfm; do
  /opt/conda/bin/python experiments/explanation_quality/run_experiment.py --dataset "$dataset" --queries 256
done
```
## T-GNNExplainer rollout sensitivity

The rollout--fidelity sensitivity on the same evidence
universe is reproduced with:

```bash
/opt/conda/bin/python experiments/explanation_quality/run_tgnn_rollout_sweep.py \
  --datasets wikipedia reddit mooc lastfm --queries 64
```

The sweep uses 40, 100, and 200 MCTS rollouts and writes
`results/tgnn_rollout_sweep.json`. The main comparison above retains 256
queries per dataset.
