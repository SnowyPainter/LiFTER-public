# RetailRocket: is K > 1 necessary?

## Protocol

The complete RetailRocket event log was downloaded and materialized as one
`Link(user,item,time)` relation. The original `view`, `addtocart`, and
`transaction` value is retained in `action_type`, outside every `feat_*` model
input. LiFTER therefore cannot read the action label.

The decisive experiment preserves the natural order of the most recent 262,144
events. Only actual transaction events are used as future-link queries, giving
1,705 training queries and 327 test queries; all views, carts, and transactions
remain available as untyped historical facts. Every positive query is paired
with one negative candidate. Models use 10 epochs and seeds 7/17/27.

## Results

| K | Historical AUC | Historical AP | Pairwise accuracy | AP drop after global predicate shuffle | Action-type NMI | Action macro recall |
|---:|---:|---:|---:|---:|---:|---:|
| **1** | 0.8555 ± 0.0015 | 0.8602 ± 0.0031 | 0.8532 ± 0.0053 | 0.0000 | 0.0000 | **0.3333** |
| 2 | 0.8566 ± 0.0048 | 0.8583 ± 0.0064 | **0.8563 ± 0.0081** | −0.0003 ± 0.0025 | **0.0150** | 0.3281 |
| 3 | 0.8560 ± 0.0031 | 0.8605 ± 0.0075 | 0.8532 ± 0.0081 | 0.0009 ± 0.0010 | 0.0108 | 0.3078 |
| 4 | **0.8583 ± 0.0033** | 0.8605 ± 0.0052 | 0.8532 ± 0.0140 | −0.0012 ± 0.0029 | 0.0105 | 0.3104 |
| 8 | 0.8565 ± 0.0021 | **0.8609 ± 0.0037** | 0.8542 ± 0.0124 | −0.0007 ± 0.0023 | 0.0128 | 0.3135 |

The maximum AUC difference from `K=1` is 0.0028 and the maximum AP difference
is 0.0007, both smaller than seed variation and not monotonic in K. Breaking
the learned fact--predicate association does not cause a repeatable loss.
Learned predicates also have near-zero NMI with the held-out action labels and
macro recall at or below the one-type 1/3 reference.

## Conclusion

`K>1` is not necessary in this relation-collapsed RetailRocket task. Once the
action label is hidden, the strictly earlier structural context does not contain
enough information for the current LiFTER objective to recover view, cart, and
transaction as distinct fact roles. Grounded temporal rules over one Link type
already express the useful recurrence and transition signal.

This result must not be generalized to settings where action type or another
role-bearing attribute is observed as an input. Supplying `view/addtocart/
transaction` directly would make multiple types useful, but would test typed
rule learning rather than latent predicate discovery.
