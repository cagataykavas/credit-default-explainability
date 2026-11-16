# Explanation missingness robustness

Production scoring records are rarely as complete as a curated evaluation table. The model
pipeline imputes missing values, but successful scoring alone does not show whether the local
explanation remains reviewable after imputation.

`audit_explanation_missingness` masks one governed feature at a time and compares the resulting
explanation with the complete-row explanation. For each scenario it reports:

- absolute probability drift;
- overlap between the top-k remaining attributions;
- attribution-direction agreement across the remaining features;
- stable policy finding codes.

Direction comparisons treat deltas within a configurable epsilon as neutral so floating-point
noise is not promoted into an apparent sign reversal.

The masked feature is excluded from rank and direction comparisons because its absence is the
intervention being audited. All other features remain fixed. Work is bounded before inference,
invalid probabilities and attributions fail closed, and the result includes a deterministic
SHA-256 evidence commitment. The public result contains no raw row values; the commitment binds a
hash of the complete reference row, the policy, and all scenario metrics.

The commitment is an integrity aid, not anonymization or producer authentication. Low-entropy
input fields may still be guessable if an attacker can enumerate plausible complete rows.

```python
from creditxai.missingness import MissingnessPolicy, audit_explanation_missingness

audit = audit_explanation_missingness(
    model,
    applicant,
    policy=MissingnessPolicy(
        top_k=3,
        max_probability_drift=0.10,
        min_top_k_overlap=0.67,
        min_direction_agreement=0.80,
    ),
)
if not audit.passed:
    raise RuntimeError(audit.findings)
```

## Interpretation limits

This is a sensitivity audit, not proof that missingness is random, imputations are correct, or an
explanation is causal. Single-feature masking does not cover correlated multi-field outages,
upstream schema errors, distribution shift, protected-class effects, or legal adverse-action
requirements. Thresholds are engineering defaults and must be calibrated on representative,
temporally held-out data before they are used as a release gate.

The next increment should evaluate predeclared multi-feature missingness patterns observed in
production telemetry, while keeping work bounded and stratifying results by reviewed cohorts.
