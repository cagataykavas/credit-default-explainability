# Global importance stability audit

A single permutation-importance ranking does not show whether the apparent top features survive a
different sample from the same review cohort. `creditxai.importance_stability` adds a bounded,
model-agnostic audit for that uncertainty.

The reference workflow:

1. draws deterministic, class-stratified bootstrap samples from a held-out cohort;
2. computes permutation importance independently for every sample;
3. measures agreement across complete feature rankings with tie-corrected Kendall's W;
4. measures how often each feature remains in the reviewed top-K set;
5. reports score quantiles, rank variation, policy violations, and a SHA-256 evidence digest.

```python
from creditxai.importance_stability import (
    ImportanceStabilityPolicy,
    audit_importance_stability,
    bootstrap_permutation_importance_runs,
)

runs = bootstrap_permutation_importance_runs(model, heldout_frame, runs=20, seed=42)
report = audit_importance_stability(
    runs,
    model_id="credit-risk-2026-11",
    cohort_id="heldout-2026-q4",
    policy=ImportanceStabilityPolicy(
        top_k=3,
        min_kendall_w=0.70,
        min_consensus_top_k_frequency=0.80,
    ),
)
```

The gate fails closed when rankings lack concordance or a consensus top feature is selected too
infrequently. Run, feature, row, and permutation caps bound the work before evaluation.

## Limitations

Bootstrap stability is not causal importance, fairness, feature actionability, or explanation
faithfulness. Correlated features may exchange ranks while the model's joint behavior remains
stable, and held-out cohort bias is reproduced by resampling. The evidence digest binds importance
scores, model/cohort identifiers, and policy—not the model artifact or raw cohort contents. A
production release should bind those identifiers to separately authenticated model and dataset
manifests and calibrate thresholds on historical review cohorts.
