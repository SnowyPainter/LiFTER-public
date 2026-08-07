# LiFTER scalability

Measures causal fact-index construction at 32k, 128k, 512k, and full Reddit;
executor scaling over H and K; candidate batching; sparse versus naive renewal;
batch size and transition-rank scaling; training throughput, GPU memory, fact
deletion re-execution, trace overhead, and serialized trace size.

```bash
/opt/conda/bin/python experiments/scalability/run_experiment.py
```

The cross-model comparison uses the same recent-10 local-history tensors and
batch sizes for TGN, TGAT, DyGFormer, GraphMixer, CRAFT, and LiFTER. It measures
forward architecture cost only; explanation latency is taken from the trained
models evaluated under the shared evidence universe in the explanation study.

```bash
/opt/conda/bin/python experiments/scalability/run_comparative.py
```

The machine-readable outputs are `results/summary.json` and
`results/comparative_summary.json`.
