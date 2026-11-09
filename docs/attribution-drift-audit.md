# Attribution drift release audit

`creditxai.attribution_drift` monitors whether a deployed explanation pipeline changes its feature story between a governed reference cohort and a current cohort. It evaluates attribution histograms in reference-defined bins, mean absolute attribution and Top-K selection rate. Prediction-score drift is measured separately so output drift cannot be mistaken for explanation drift.

Each versioned artifact binds the model, explainer configuration, cohort manifests, time windows and histogram-bin definitions by SHA-256. Feature records use standard/critical risk tiers. The policy supports aggregate tolerances for standard features and zero-tolerance drift for critical features.

The report contains Jensen-Shannon divergence normalized to `[0, 1]`, salience-weighted mean feature divergence, drifted-feature fraction, maximum Top-K/mean-attribution changes and prediction divergence. Feature and monitor identifiers are hashed; raw rows, predictions and attribution values are not emitted. JSON parsing rejects duplicate fields, non-finite values, unknown fields, invalid chronology and resource-budget violations.

```bash
python -m creditxai.attribution_drift drift-artifact.json --output drift-report.json
```

CLI exits are `0` for acceptance, `2` for a well-formed policy rejection and `3` for malformed evidence. Output replacement is atomic.

## Governed subgroup partitions

Aggregate stability can conceal a material change in a smaller cohort. `creditxai.subgroup_drift` wraps the aggregate artifact and one artifact per member of a prespecified, mutually exclusive partition. It applies the same drift policy to every subgroup and adds four release checks:

- every governed subgroup is present and undeclared slices are rejected;
- each reference/current subgroup has the configured minimum support;
- subgroup model, explainer, time windows, feature risks and bin definitions match the aggregate;
- slice counts form a complete partition by default, while overlap always fails closed.

Subgroup IDs are accepted only as bounded identifiers and appear in reports as SHA-256 digests. The report retains per-slice sample counts and nested drift metrics for review, but it does not emit the raw partition name, subgroup names, row data or feature identifiers.

```bash
python -m creditxai.subgroup_drift drift-partition.json \
  --governed-slice north \
  --governed-slice south \
  --min-samples 100 \
  --output subgroup-drift-report.json
```

The partition CLI uses the same exits as the aggregate audit: `0` accepted, `2` policy rejection and `3` malformed evidence. Output replacement is atomic. Partial coverage must be explicitly enabled with `--allow-partial`; this is intended for deliberately sampled monitoring, not as a workaround for missing governed cohorts.

Minimal envelope shape:

```json
{
  "schema_version": "credit-attribution-drift-partition/v1",
  "partition_id": "deployment-region",
  "aggregate": {"schema_version": "credit-attribution-drift/v1"},
  "slices": [
    {"slice_id": "north", "artifact": {"schema_version": "credit-attribution-drift/v1"}},
    {"slice_id": "south", "artifact": {"schema_version": "credit-attribution-drift/v1"}}
  ]
}
```

Each abbreviated attribution artifact above must contain the complete aggregate-audit contract.

## Trust boundaries and limitations

The audit detects distribution change; it does not determine whether either cohort's explanations are faithful, causal, fair or legally suitable. Reference-defined bins, subgroup definitions and thresholds must be prespecified on representative traffic. Histogram aggregation can hide within-bin movement and correlated feature swaps. Top-K rates depend on the chosen `K` and attribution normalization.

Count reconciliation cannot prove that subgroup manifests are disjoint: a duplicated row and a missing row can cancel numerically. The collector must enforce one membership per row and retain access-controlled manifests for audit. Small groups below minimum support fail closed instead of being silently pooled; production policy must define a privacy-preserving escalation path for those groups.

SHA-256 binds inputs but does not authenticate the collector. Production evidence should be signed or MACed, retained with the cohort manifests, and periodically checked against raw access-controlled samples. Multiple overlapping windows require alert-deduplication or multiplicity control.

## Next integration step

Generate the partition envelope directly from the production explanation path and bind signed subgroup manifests to a membership-assignment job that proves disjointness. Route aggregate or subgroup critical-feature drift to rollback, abstention or human review according to the deployment policy.
