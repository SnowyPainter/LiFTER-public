# Independent execution-certificate verification

The verifier reconstructs clause groundings, aggregate evidence, signed rule
contributions and transition potentials from the raw pre-query history and frozen
parameters. Values copied into the certificate are not trusted as computational
inputs. The numerical tolerance is 2e-5 for float32 execution.

| Dataset | Traces | Complete | Grounding valid | Temporally valid | Max logit error | Verification ms/query |
|---|---:|---:|---:|---:|---:|---:|
| Wikipedia | 4,916 | 100% | 100% | 100% | 1.00e-5 | 0.346 |
| Reddit | 4,916 | 100% | 100% | 100% | 1.31e-5 | 0.315 |
| MOOC | 4,916 | 100% | 100% | 100% | 1.21e-6 | 0.248 |
| LastFM | 4,916 | 100% | 100% | 100% | 1.05e-5 | 0.247 |
| Total / mean | 19,664 | 100% | 100% | 100% | 1.31e-5 | 0.289 |

All recorded candidate logits were reconstructed within tolerance. Q1 and Q2
also cite the most recent one or two outgoing facts in the raw source history.

| Tampering operation | Detection rate |
|---|---:|
| Fact omission | 100% |
| Timestamp modification | 100% |
| Entity modification | 100% |
| Nonexistent fact insertion | 100% |
| Contribution modification | 100% |

Execution-order permutation and JSON round-trip controls were accepted in 100%
of eligible cases on all four datasets.
