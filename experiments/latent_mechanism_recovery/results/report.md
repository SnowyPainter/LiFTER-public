# Single-relation latent-mechanism recovery

All historical events were exposed as the same `Link(source, destination,
time)` relation. The generator mixed three reusable functional roles across
recurrence, transition, and exploration rule templates. Neither the hidden
fact-role labels nor entity identities were given to the predictor. The only
supervision for learned typing was whether the future candidate Link occurred.
Training and test entity sets were disjoint.

| Typing | Test AUC | Test AP | Hidden-role accuracy | NMI | ARI |
|---|---:|---:|---:|---:|---:|
| Single predicate | 0.4948 ± 0.0091 | 0.4969 ± 0.0053 | 0.3594 ± 0.0030 | 0.0000 | 0.0000 |
| Random fixed predicates | 0.4977 ± 0.0072 | 0.4985 ± 0.0053 | 0.3393 ± 0.0008 | 0.0001 | −0.0000 |
| Unsupervised k-means | 0.4984 ± 0.0052 | 0.5001 ± 0.0033 | 0.3394 ± 0.0007 | 0.0002 | 0.0000 |
| **Future-link learned predicates** | **0.9931 ± 0.0006** | **0.9933 ± 0.0002** | **0.9752 ± 0.0007** | **0.8802 ± 0.0027** | **0.9270 ± 0.0019** |

Values are means and sample standard deviations over seeds 7, 17, 27, 37,
and 47. Hidden-role accuracy is permutation-matched, so predicate indices are
not assumed to have a predetermined meaning.

The result establishes the intended possibility: even when the observed event
vocabulary contains only one relation, future-link supervision can separate its
facts into reusable functional roles when the data-generating process actually
mixes distinct mechanisms. It does not establish that the four JODIE datasets
contain such recoverable roles; their real-data ablation answers that separate
empirical question.

The learned program's mean AUC was 0.9985 on recurrence, 0.9907 on transition,
and 0.9914 on exploration. Thus the aggregate result is not carried by only one
of the three mechanisms.
