# Counterfactual recourse audit

`find_actionable_counterfactual` proposes a small set of feature changes. Proposal
generation alone is not evidence that the result is safe to present as recourse.
`creditxai.recourse_audit.audit_counterfactual` independently reconstructs and
rescores the candidate before it can be admitted.

```python
from creditxai.counterfactual import find_actionable_counterfactual
from creditxai.recourse_audit import RecourseAuditPolicy, audit_counterfactual

proposal = find_actionable_counterfactual(model, applicant)
report = audit_counterfactual(
    model,
    applicant,
    proposal.changed_features,
    policy=RecourseAuditPolicy(target_probability=0.25),
)
if not report.accepted:
    raise RuntimeError(report.reason_codes)
```

## Contract

The audit fails closed on malformed rows, policies, feature values, or model
probabilities. A well-formed proposal is rejected unless all of these conditions
hold:

- the original score is above the target and the candidate reaches it;
- the probability reduction meets the configured minimum;
- only `income`, `debt_ratio`, `late_payments`, and `utilization` change;
- each change follows its declared beneficial direction;
- the number and normalized cost of changes remain within policy budgets; and
- removing any one supplied action makes the candidate miss the target.

The report is deterministic and JSON-ready. It includes probabilities, normalized
cost, ordered feature deltas, immutable or direction-invalid changes, unnecessary
features, and stable reason codes. Invalid evidence raises `RecourseAuditError`
with a machine-readable `code`; policy rejection returns `accepted=False`.

## Trust boundary and limitations

This gate verifies a proposal against the supplied model; it does not establish
that the model is calibrated, fair, causally valid, or legally suitable. The
direction rules and normalization scales are illustrative and require governance
and calibration for a real lending product. `late_payments` describes historical
behavior and may not be immediately actionable even though the demo generator
treats it as mutable.

The leave-one-out check proves only inclusion-minimality for the supplied action
set. It does not find a globally cheapest counterfactual, test smaller magnitudes,
model dependencies between features, or validate real-world feasibility. A
production next step is constrained optimization over institution-approved action
ranges followed by cohort-level feasibility and fairness monitoring.
