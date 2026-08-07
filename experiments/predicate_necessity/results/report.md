# Is learnable fact typing necessary in LiFTER?

## Protocol

All variants use the same 32,768 events, chronological 85/15 split, historical
negative sampler, grounded rule executor, `H=10`, optimizer, 10 epochs, and
seeds 7/17/27. Only fact typing changes.

| Variant | Assignment | Future-link supervision of typing | Execution |
|---|---|---:|---|
| Single type | every fact is `P0` | no | hard |
| Random types | frozen random feature projection, `K=8` | no | hard |
| Unsupervised clusters | training-prefix MiniBatch k-means, `K=8` | no | hard |
| Learned soft | neural encoder and prototypes, `K=8` | yes | soft mixture |
| Learned hard | neural encoder and prototypes, `K=8` | yes | discrete straight-through |

K-means sees only the training prefix and no future-link labels. Random and
k-means assignments are frozen before forecasting training.

## Forecasting

| Dataset | Typing | Historical AUC | Historical AP |
|---|---|---:|---:|
| Wikipedia | Single type | 0.8923 ± 0.0015 | **0.8966 ± 0.0017** |
| | Random types | 0.8898 ± 0.0013 | 0.8952 ± 0.0023 |
| | Unsupervised clusters | 0.8918 ± 0.0009 | 0.8950 ± 0.0007 |
| | **Learned soft** | **0.8939 ± 0.0008** | 0.8927 ± 0.0025 |
| | Learned hard | 0.8922 ± 0.0014 | 0.8959 ± 0.0030 |
| Reddit | Single type | 0.8360 ± 0.0018 | 0.8287 ± 0.0043 |
| | Random types | 0.8368 ± 0.0007 | **0.8310 ± 0.0008** |
| | **Unsupervised clusters** | **0.8369 ± 0.0015** | 0.8308 ± 0.0023 |
| | Learned soft | 0.8367 ± 0.0021 | 0.8255 ± 0.0037 |
| | Learned hard | 0.8347 ± 0.0029 | 0.8267 ± 0.0055 |
| MOOC | Single type | 0.8674 ± 0.0008 | **0.8508 ± 0.0029** |
| | Random types | 0.8661 ± 0.0024 | 0.8442 ± 0.0073 |
| | Unsupervised clusters | 0.8667 ± 0.0004 | 0.8473 ± 0.0030 |
| | Learned soft | 0.8663 ± 0.0010 | 0.8426 ± 0.0013 |
| | **Learned hard** | **0.8687 ± 0.0029** | 0.8495 ± 0.0055 |
| LastFM | Single type | 0.7067 ± 0.0057 | 0.7187 ± 0.0061 |
| | Random types | 0.7028 ± 0.0134 | 0.7196 ± 0.0129 |
| | Unsupervised clusters | 0.7069 ± 0.0111 | 0.7250 ± 0.0105 |
| | Learned soft | 0.7096 ± 0.0083 | 0.7226 ± 0.0101 |
| | **Learned hard** | **0.7107 ± 0.0097** | **0.7280 ± 0.0097** |

The macro means are 0.8256/0.8237 for single type and 0.8266/0.8250 for
learned hard. AUC and AP gains are therefore 0.0010 and 0.0013. Learned typing
does not produce a practically meaningful general forecasting improvement.

## Program concentration

`Rules@90%` is the minimum number of executed clause types accounting for 90%
of total absolute contribution. `Top-3` is the mean fraction of a query's
absolute non-prior logit contribution carried by its three largest executions.

| Typing | Macro Rules@90% ↓ | Macro Top-3 contribution ↑ |
|---|---:|---:|
| **Single type** | **4.6** | **0.869** |
| Random types | 16.2 | 0.819 |
| Unsupervised clusters | 43.2 | 0.774 |
| Learned soft | 78.4 | 0.689 |
| Learned hard | 74.0 | 0.722 |

Learned predicates expand rather than compress the executed program. The
single-type model reaches the same forecasting quality with roughly one
sixteenth as many clause types in the 90%-mass set.

## Discrete execution and reconstruction

The maximum absolute error between the candidate logit and the sum of prior,
grounded clause contributions, positioned recurrence, and transition
contributions is at most `7.6e-6` in every variant. Single, random, k-means, and
learned-hard variants produce discrete traces. Learned-soft also has an exact
algebraic decomposition, but its atoms are probability mixtures rather than a
single discrete predicate per fact.

Hard and soft learned typing have nearly identical macro AUC (0.8266 versus
0.8267). Hard execution has higher macro AP (0.8250 versus 0.8209) and a more
concentrated program, so hard execution remains preferable if `K=8` typing is
retained.

## Reuse and structural meaning

For each predicate, absolute clause contribution is accumulated over ten
structural rule roles. Role reuse is the contribution-weighted cosine between
the same predicate's role profile across seen/unseen entities, early/later test
intervals, low/high source activity, and low/high destination popularity.
Predicate labels are aligned across seeds by maximum-weight matching.

Learned-hard predicates are stable across seeds (0.977–0.991) and across later
time intervals (0.993–0.998). However, the controls are similarly stable. More
importantly, learned-hard predicates have low within-run role separation
(0.055, 0.066, 0.070, and 0.064 across the four datasets): different learned
predicates execute nearly the same mixture of structural roles. Seed stability
therefore reflects a stable but largely redundant partition, not distinct
reusable mechanisms.

## Answer

The present LiFTER does not establish the necessity of learnable predicates.
Learned typing is exact and reproducible, and it gives a small LastFM gain, but
it neither improves forecasting consistently nor compresses the program nor
separates distinct rule roles. The defensible conclusion is diagnostic:

> The grounded temporal rule architecture is responsible for the measured
> forecasting performance; the current latent predicate module is not yet a
> necessary or structurally differentiated component.

Claiming that predicates compress reusable mechanisms would contradict these
results. LiFTER should either adopt the single-type executor as the parsimonious
model, or redesign predicate learning with an explicit role-specialization and
minimum-description-length objective and then repeat this protocol.
