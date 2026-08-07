# Neural-symbolic responsibility

Forecasting ablations on MOOC and LastFM: exact binding, randomized binding,
renewal, transitions, Q-only, symbolic-without-Q, and a same-input GraphMixer
MLP control. The default run is an 8,192-training-event pilot; `--full-scale`
uses the complete 32,768-event benchmark protocol.

```bash
/opt/conda/bin/python experiments/neural_symbolic_responsibility/run_experiment.py --full-scale --workers 3
```

The runner accepts `--datasets`, `--variants`, and `--seeds` for exact cells.
