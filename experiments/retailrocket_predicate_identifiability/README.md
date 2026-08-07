# RetailRocket predicate identifiability diagnosis

Two tests separate task necessity from learnability.

1. **Oracle K=3:** expose the original view/cart/transaction action only to a
   fixed predicate assignment. If this does not beat K=1, typed rules are not
   needed for the transaction-query task.
2. **Context probe:** predict the current action from exactly the strictly
   earlier structural context seen by LiFTER, using class-balanced loss. If it
   is at chance, the hidden action is not identifiable from the model input.

The oracle stream is explicitly audit-only and is never used as a main model.
