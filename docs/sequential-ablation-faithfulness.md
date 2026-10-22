# Sequential-ablation faithfulness audit

A stable or additive feature attribution can still rank features poorly. This audit
tests the ranking against the model itself: it starts from the governed baseline,
restores observed feature values in attribution order, and measures how quickly the
model output approaches the original prediction. The same restoration is repeated in
reverse attribution order as a deterministic negative control.

For every sufficiently contrasted case, `creditxai.faithfulness` reports:

- normalized output recovery after the configured Top-K features;
- area under the full attribution-ordered restoration curve;
- area under the reverse-order control curve;
- the ordering advantage between those curves;
- minimum and maximum recovery, exposing overshoot and interaction effects;
- repeated observed/baseline prediction drift.

Cases whose observed-to-baseline output gap is too small are excluded before division.
The release policy separately limits the excluded fraction and the failed fraction, so
a suite cannot pass by filling itself with near-zero-contrast examples.

## Run the deterministic reference audit

```bash
python -m creditxai.faithfulness \
  --rows 2200 \
  --audit-rows 64 \
  --seed 42 \
  --output artifacts/sequential-ablation-faithfulness.json
```

Exit codes are `0` for accepted, `2` for a well-formed policy rejection, and `3`
for malformed or operational input. The output is written atomically. It contains
hashed case references, curve summaries, policy/model/cohort/explainer identities,
bounded findings, prediction-call accounting, and a canonical SHA-256 evidence ID.
It does not contain applicant rows, category values, individual feature attributions,
or raw model outputs.

Library users can call `audit_sequential_ablation` with an already loaded model and
a held-out `pandas.DataFrame`. The runner validates the exact feature schema, numeric
domains, categorical bounds, unique case identities, model probabilities, complete
attribution identities, resource budgets, and repeated prediction stability before
issuing a report. Generate the declared cohort identity with
`cohort_content_sha256(frame)`: the audit recomputes it from canonical case references
and feature values, evaluates a deep snapshot, and fails closed if the caller supplies
a different identity or the source frame changes during execution.

## Interpretation and trust boundaries

- A positive ordering advantage means the supplied ranking restores this model's
  output faster than its reverse. It is evidence of ranking usefulness, not causality.
- The baseline defines the restoration question. A different defensible baseline can
  produce a different ranking and curve.
- Correlated features and interactions can create overshoot or temporary negative
  recovery. Those values are retained rather than clipped into a favorable range.
- Near-zero output gaps are statistically unstable after normalization and are
  explicitly reported as excluded, not silently treated as passes.
- The default explainer is the repository's one-feature baseline perturbation method.
  A production caller must bind the supplied model, cohort, and explainer digests to
  immutable artifacts; SHA-256 does not authenticate the producer.
- The cohort digest proves exact snapshot identity, not confidentiality or provenance.
  A signed manifest is still required to authenticate who produced the cohort.
- Passing does not establish fairness, calibration, legal suitability, human
  usefulness, counterfactual feasibility, or an adverse-action reason.

The next integration step is to run this audit on versioned production-like cohorts,
calibrate thresholds by model family, and bind an accepted evidence digest to the
same model-promotion record used by baseline-sensitivity and drift monitoring.
