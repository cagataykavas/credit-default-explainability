# Attribution drift release audit

`creditxai.attribution_drift` monitors whether a deployed explanation pipeline changes its feature story between a governed reference cohort and a current cohort. It evaluates attribution histograms in reference-defined bins, mean absolute attribution and Top-K selection rate. Prediction-score drift is measured separately so output drift cannot be mistaken for explanation drift.

Each versioned artifact binds the model, explainer configuration, cohort manifests, time windows and histogram-bin definitions by SHA-256. Feature records use standard/critical risk tiers. The policy supports aggregate tolerances for standard features and zero-tolerance drift for critical features.

The report contains Jensen-Shannon divergence normalized to `[0, 1]`, salience-weighted mean feature divergence, drifted-feature fraction, maximum Top-K/mean-attribution changes and prediction divergence. Feature and monitor identifiers are hashed; raw rows, predictions and attribution values are not emitted. JSON parsing rejects duplicate fields, non-finite values, unknown fields, invalid chronology and resource-budget violations.

```bash
python -m creditxai.attribution_drift drift-artifact.json --output drift-report.json
```

CLI exits are `0` for acceptance, `2` for a well-formed policy rejection and `3` for malformed evidence. Output replacement is atomic.

## Trust boundaries and limitations

The audit detects distribution change; it does not determine whether either cohort's explanations are faithful, causal, fair or legally suitable. Reference-defined bins and thresholds must be prespecified on representative traffic. Histogram aggregation can hide within-bin movement, subgroup drift and correlated feature swaps. Top-K rates depend on the chosen `K` and attribution normalization.

SHA-256 binds inputs but does not authenticate the collector. Production evidence should be signed or MACed, retained with the cohort manifests, and periodically checked against raw access-controlled samples. Multiple overlapping windows require alert-deduplication or multiplicity control.

## Next integration step

Generate the artifact directly from the production explanation path, version reference bins by model release, and add governed subgroup slices with minimum support. Route critical-feature drift or prediction drift to rollback, abstention or human review according to the deployment policy.
