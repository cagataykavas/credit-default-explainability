# Baseline-sensitivity release audit

Local attributions are conditional on a reference profile. An explanation can
be exactly additive for one baseline and still tell a materially different
story when a second defensible baseline is used. This gate makes that dependency
measurable before explanation artifacts are released.

## Evidence contract

The CLI accepts strict JSON with a model and method identity plus one or more
cases. Each case supplies the observed probability and at least three
independently chosen baselines. Every baseline records its probability and an
exactly aligned vector of signed feature attributions:

```json
{
  "schema_version": "1.0",
  "method_id": "exact-shapley-v1",
  "model_id": "credit-risk-2026-09",
  "cases": [{
    "case_id": "case-001",
    "observed_probability": 0.8,
    "baselines": [{
      "baseline_id": "training-median",
      "baseline_probability": 0.2,
      "attributions": [
        {"feature_id": "income", "value": 0.3},
        {"feature_id": "debt_ratio", "value": 0.2},
        {"feature_id": "utilization", "value": 0.1}
      ]
    }]
  }]
}
```

The abbreviated example shows one baseline; an auditable artifact must contain
at least three. Producers should choose and version baselines before inspecting
the audited cases—for example a governed training reference, a recent eligible
cohort, and a policy-approved low-risk cohort.

Run the default gate with:

```bash
python -m creditxai.baseline_sensitivity evidence.json --output report.json
```

Exit `0` means accepted, `2` means valid evidence failed policy, and `3` means
the evidence or output was malformed.

## Checks

For every baseline, the gate reconciles the attribution sum with
`observed_probability - baseline_probability`. Across every pair of baselines it
then measures:

- signed cosine similarity, which detects global direction changes;
- top-K overlap, which detects churn in the most salient features;
- sign agreement among shared salient features;
- baseline-probability span and minimum observed-to-baseline contrast.

Admission is fail-closed. IDs must be unique and aligned, numeric values must be
finite, vectors must be non-zero, and object shapes are exact. Case, baseline,
feature, byte, and pair-comparison budgets bound CPU and memory use. Duplicate
JSON keys and non-standard `NaN`/`Infinity` constants are rejected.

Reports contain policy thresholds, aggregate and per-case findings, bounded
metrics, canonical SHA-256 identities, and hashes in place of case/baseline IDs.
Raw feature values and attribution values are not echoed.

## Trust boundary and limitations

The gate assumes the producer used the declared model, method, case, and
baseline inputs. A digest establishes content identity, not authenticity; sign
artifacts or store them in an immutable system if producer integrity matters.

A pass does not identify the *correct* baseline, prove causal faithfulness,
calibration, fairness, robustness, or legal suitability. Different baselines
can legitimately answer different questions. Thresholds require prespecified
calibration on representative governed cohorts, and correlated features can
redistribute credit while the model behavior remains unchanged. The gate should
therefore supplement—not replace—additivity, perturbation, randomization,
performance, and fairness evidence.

## Next step

Generate this contract directly from the exact-additivity runner, bind it to
versioned cohort manifests, and monitor sensitivity distributions across time
windows and decision-relevant subgroups.
