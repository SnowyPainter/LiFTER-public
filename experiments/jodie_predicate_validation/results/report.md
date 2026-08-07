# Does JODIE use more than one latent predicate?

LiFTER was trained with `K=1,2,4,8` on 32,768 events for 10 epochs and seeds
7/17/27. All cells used the final benchmark architecture, temporal split, and
uniform historical alternatives.

| Dataset | K | Historical AUC | Historical AP | Pairwise accuracy | Global-shuffle drop AUC/AP |
|---|---:|---:|---:|---:|---:|
| Wikipedia | 1 | 0.8923 ± 0.0015 | 0.8966 ± 0.0017 | **0.8544** | 0.0000 / 0.0000 |
| | 2 | 0.8916 ± 0.0005 | 0.8966 ± 0.0008 | 0.8528 | −0.0002 / −0.0004 |
| | 4 | 0.8917 ± 0.0014 | 0.8944 ± 0.0013 | 0.8511 | −0.0003 / −0.0002 |
| | 8 | **0.8929 ± 0.0014** | **0.8969 ± 0.0025** | 0.8525 | 0.0004 / 0.0007 |
| Reddit | 1 | 0.8360 ± 0.0018 | 0.8287 ± 0.0043 | 0.7577 | 0.0000 / 0.0000 |
| | 2 | **0.8366 ± 0.0009** | 0.8294 ± 0.0012 | 0.7592 | −0.0002 / −0.0006 |
| | 4 | 0.8356 ± 0.0017 | 0.8260 ± 0.0026 | 0.7588 | −0.0010 / −0.0011 |
| | 8 | 0.8365 ± 0.0024 | **0.8301 ± 0.0029** | **0.7593** | −0.0009 / −0.0006 |
| MOOC | 1 | 0.8674 ± 0.0008 | **0.8508 ± 0.0029** | 0.8632 | 0.0000 / 0.0000 |
| | 2 | 0.8674 ± 0.0032 | 0.8459 ± 0.0064 | 0.8626 | 0.0005 / 0.0005 |
| | 4 | **0.8693 ± 0.0021** | 0.8488 ± 0.0052 | 0.8642 | 0.0020 / 0.0019 |
| | 8 | 0.8680 ± 0.0037 | 0.8459 ± 0.0090 | **0.8645** | 0.0030 / 0.0043 |
| LastFM | 1 | 0.7067 ± 0.0057 | 0.7187 ± 0.0061 | 0.6983 | 0.0000 / 0.0000 |
| | 2 | **0.7125 ± 0.0055** | 0.7274 ± 0.0052 | 0.6996 | 0.0007 / 0.0007 |
| | 4 | 0.7033 ± 0.0041 | 0.7211 ± 0.0017 | 0.6905 | 0.0001 / 0.0004 |
| | 8 | 0.7113 ± 0.0092 | **0.7295 ± 0.0092** | **0.7030** | 0.0005 / 0.0007 |

Global shuffle rotates learned predicate assignments across all valid facts in
an evaluation batch while preserving their total histogram. `K=1` is exactly
unchanged, which verifies matched candidates and deterministic evaluation.

Wikipedia and Reddit show no repeatable material gain over `K=1`; shuffling is
also harmless. MOOC has the clearest fact-role use: its role-profile separation
is 0.117/0.106/0.077 for `K=2/4/8`, and shuffling `K=8` costs 0.0030 AUC and
0.0043 AP. This specialization nevertheless does not beat `K=1` AP. LastFM is
the only dataset with a consistent forecasting gain: `K=2` improves every seed
in AUC and AP, while `K=8` improves AP in every seed. Yet shuffling removes only
0.0005--0.0007, so most of that gain cannot be attributed to a stable mapping
between individual facts and functional predicates. It is more consistent with
extra parameterization or query-level score shaping.

The defensible conclusion is therefore not that `K>1` is numerically useless.
It is that these four JODIE benchmarks do not provide evidence that multiple
learned predicates act as reusable fact-level semantic roles. MOOC uses the
assignments weakly without a forecasting benefit; LastFM benefits from the
larger model without depending materially on the fact--predicate association.
