# RetailRocket latent-predicate validation

All `view`, `addtocart`, and `transaction` events are materialized as one
`Link(user,item,time)` relation. The original action is retained in the
non-feature `action_type` column and is never passed to LiFTER. It is opened
only after training to measure whether predicates recovered the action roles.

The decisive experiment preserves the natural order of the most recent 262,144
events but uses actual `transaction` events as future-link queries. Historical
views, carts, and transactions remain in the untyped history. Each positive
transaction query receives one negative candidate, preventing the 96% view
frequency from dominating the training objective. The experiment compares
`K=1,2,3,4,8` for 10 epochs and seeds 7/17/27, and reports forecasting, a global
predicate-shuffle intervention, and action-type NMI/ARI/macro recall.

```bash
/opt/conda/bin/python experiments/retailrocket_predicate_validation/run_experiment.py --workers 3
```
