# Model-to-explanation schema lineage

An explanation can be numerically stable and still describe the wrong feature. Common production
failures include a reordered model input, a changed scaler, a new one-hot category, or an attribution
vector decoded with an older label map. These failures are dangerous because the output remains
plausible while reason codes and governance evidence silently refer to different semantics.

`creditxai.explanation_contract` creates a content-addressed contract from the fitted credit model's
actual `ColumnTransformer`, then audits each explanation against that contract before release.

## Contract contents

For every raw feature, the contract records:

- exact input position;
- dtype class, unit, and versioned semantic identifier;
- a SHA-256 digest of fitted imputation/scaling or category-encoding state; and
- the ordered transformed columns produced from that feature.

The complete transformed feature order and immutable model artifact digest are bound to a canonical
pipeline digest and contract digest. The helper uses the fitted sklearn objects, rather than a
hand-maintained copy of the feature list.

```python
from datetime import UTC, datetime

from creditxai.explain import local_perturbation_attribution
from creditxai.explanation_contract import (
    audit_explanation_contract,
    bind_explanation,
    build_credit_feature_contract,
)

contract = build_credit_feature_contract(model, model_digest="<sha256 of promoted model>")
artifact = bind_explanation(
    contract,
    local_perturbation_attribution(model, row),
    generated_at=datetime.now(UTC),
)
report = audit_explanation_contract(artifact, now=datetime.now(UTC))
assert report.accepted
```

## Admission checks

The fail-closed audit recomputes the pipeline and contract digests, then reconciles:

- model, pipeline, and contract identities;
- raw and transformed feature ordering;
- complete one-to-one attribution coverage;
- feature-to-position mappings;
- attribution sign and direction metadata;
- approved attribution methods; and
- evidence freshness and future-clock skew.

Strict JSON parsing rejects duplicate fields, non-finite numbers, unexpected schema fields, invalid
digests, ambiguous transformed-output ownership, and bounded-resource violations. The report carries
only hashes, counts, timestamps, and stable reason codes; observed values, baselines, and attribution
values remain inside the admission boundary.

```bash
python -m creditxai.explanation_contract bound-explanation.json \
  --now 2026-10-01T09:30:00Z \
  --output lineage-report.json
```

Exit codes are `0` for accepted evidence, `2` for malformed input, and `3` for a well-formed policy
rejection. Report writes are atomic.

## Trust boundary and limitations

The caller must supply a trusted SHA-256 identity for the promoted model artifact. A hash identifies
bytes but does not authenticate the producer; production systems should sign the model, contract,
and audit report. The gate verifies schema lineage, not attribution faithfulness, fairness, causal
meaning, calibration, or legal suitability of a reason code. Units and semantic identifiers are
declared policy and require governance review when a feature definition changes.

The included builder supports this repository's fitted sklearn pipeline. Other frameworks need an
adapter that extracts equivalent ordered transform state without serializing raw training data.
