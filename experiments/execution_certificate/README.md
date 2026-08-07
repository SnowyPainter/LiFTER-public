# Independently verifiable execution certificates

The verifier independently reconstructs clause groundings, aggregate evidence,
rule contributions and transition potentials from the raw pre-query history and
frozen parameters. It checks fact provenance, entity bindings, transition top-k
selection, temporal guards, duplicate executions and complete-logit reconstruction
at a float32 tolerance of 2e-5. Single-field and coordinated tampering provide
negative controls; execution permutation and JSON round-trip provide
semantics-preserving positive controls.

```bash
for dataset in wikipedia reddit mooc lastfm; do
  /opt/conda/bin/python experiments/execution_certificate/run_experiment.py --dataset "$dataset" --queries 0
done
```

`--queries 0` verifies every query in the chronological test split. A positive
value selects that many evenly spaced queries for a diagnostic run.
