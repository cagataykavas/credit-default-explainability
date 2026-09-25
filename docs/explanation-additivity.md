# Exact explanation additivity audit

The existing local explanation replaces one feature at a time while holding all other
observed values fixed. Those deltas are useful sensitivity diagnostics, but interactions
can make their sum differ from the model's total probability change from baseline.
Treating them as additive shares can therefore over- or under-explain a decision.

`creditxai.additivity.audit_exact_additivity` evaluates every coalition of observed and
baseline feature values and computes exact Shapley values for the model's default
probability. For the repository's eight features this requires 256 coalition scores.

```python
from creditxai.additivity import audit_exact_additivity

report = audit_exact_additivity(model, one_row_frame)
assert report.passed
assert abs(report.additivity_residual) <= 1e-10
```

The report includes baseline and observed probabilities, total probability change,
exact per-feature contributions, the additivity residual, and the residual obtained by
summing conventional leave-one-out deltas. Comparing the two makes interaction-driven
double counting visible without presenting it as a model defect.

## Failure and evidence contracts

The audit fails closed on missing or duplicate features, multi-row inputs, non-finite or
unsupported feature values, invalid probabilities, stochastic endpoint drift, and
feature/coalition budget overruns. Baseline and observed coalitions are scored a second
time to detect prediction instability. Output is deterministic and JSON-ready, with a
SHA-256 identity over feature names, model outputs, contributions, and active policy.
Raw observed or baseline feature values are not copied into the report.

## Interpretation and limitations

Exact additivity is relative to the selected baseline and the model probability scale.
Changing the baseline changes the question and therefore the attribution. Probability-
scale contributions can also behave differently from log-odds contributions.

A passing audit verifies numerical reconciliation and endpoint repeatability; it does
not prove causal validity, fairness, calibration, robustness, actionability, or legal
suitability. Correlated features can receive unintuitive allocations, and this bounded
enumeration is exponential. Larger feature spaces need grouped features, sampling with
uncertainty intervals, or a model-specific polynomial-time method. The next increment
should version governed baseline cohorts and compare contribution distributions across
time and monitored subgroups.
