from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from itertools import combinations
from typing import Any, Protocol

import numpy as np
import pandas as pd

INTERACTION_AUDIT_VERSION = "creditxai-pair-interaction/v1"


class ProbabilityModel(Protocol):
    def predict_probability(self, row: pd.DataFrame) -> float: ...

    def baseline_row(self) -> dict[str, Any]: ...


class InteractionAuditError(ValueError):
    """Raised when interaction evidence cannot be evaluated safely."""


@dataclass(frozen=True)
class InteractionPolicy:
    max_abs_interaction: float = 0.05
    neutral_epsilon: float = 1e-8
    max_features: int = 16
    max_prediction_calls: int = 256
    max_findings: int = 64

    def validate(self) -> None:
        if (
            not math.isfinite(self.max_abs_interaction)
            or not 0 <= self.max_abs_interaction <= 1
        ):
            raise InteractionAuditError(
                "max_abs_interaction must be finite and in [0, 1]"
            )
        if (
            not math.isfinite(self.neutral_epsilon)
            or not 0 <= self.neutral_epsilon <= 1
        ):
            raise InteractionAuditError("neutral_epsilon must be finite and in [0, 1]")
        if self.neutral_epsilon > self.max_abs_interaction:
            raise InteractionAuditError(
                "neutral_epsilon cannot exceed max_abs_interaction"
            )
        if not 2 <= self.max_features <= 32:
            raise InteractionAuditError("max_features must be in [2, 32]")
        if not 1 <= self.max_prediction_calls <= 10_000:
            raise InteractionAuditError("max_prediction_calls must be in [1, 10000]")
        if not 1 <= self.max_findings <= 512:
            raise InteractionAuditError("max_findings must be in [1, 512]")


@dataclass(frozen=True)
class PairInteraction:
    feature_a: str
    feature_b: str
    probability_residual: float
    magnitude: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InteractionAudit:
    schema_version: str
    accepted: bool
    model_revision: str
    input_sha256: str
    baseline_sha256: str
    original_probability: float
    evaluated_features: tuple[str, ...]
    prediction_calls: int
    policy: InteractionPolicy
    maximum_abs_interaction: float
    total_abs_interaction: float
    interactions: tuple[PairInteraction, ...]
    findings: tuple[str, ...]
    evidence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_pairwise_interactions(
    model: ProbabilityModel,
    row: pd.DataFrame,
    *,
    model_revision: str,
    features: tuple[str, ...],
    policy: InteractionPolicy = InteractionPolicy(),
) -> InteractionAudit:
    """Audit second-order probability residuals for baseline substitutions.

    For features i and j, the residual is
    f(x) - f(x_i=base_i) - f(x_j=base_j) + f(x_i=base_i, x_j=base_j).
    It is a local probability-scale diagnostic, not a causal or SHAP interaction.
    """
    policy.validate()
    _validate_identity(model_revision, "model_revision")
    if not isinstance(row, pd.DataFrame) or len(row) != 1:
        raise InteractionAuditError("row must be a one-row DataFrame")
    if row.columns.has_duplicates:
        raise InteractionAuditError("row columns must be unique")
    if not 2 <= len(features) <= policy.max_features:
        raise InteractionAuditError("feature count is outside the policy budget")
    if len(features) != len(set(features)):
        raise InteractionAuditError("features must be unique")
    if tuple(row.columns) != features:
        raise InteractionAuditError(
            "row columns must exactly match the declared feature order"
        )
    for feature in features:
        _validate_identity(feature, "feature")

    required_calls = 1 + len(features) + (len(features) * (len(features) - 1) // 2)
    if required_calls > policy.max_prediction_calls:
        raise InteractionAuditError(
            f"audit requires {required_calls} predictions; policy permits {policy.max_prediction_calls}"
        )

    baseline = model.baseline_row()
    if not isinstance(baseline, dict) or set(baseline) != set(features):
        raise InteractionAuditError(
            "baseline must contain exactly the declared features"
        )
    observed_values = [
        _json_scalar(row.iloc[0][feature], f"row.{feature}") for feature in features
    ]
    baseline_values = [
        _json_scalar(baseline[feature], f"baseline.{feature}") for feature in features
    ]

    original = _predict(model, row)
    singles: dict[str, float] = {}
    for feature in features:
        perturbed = row.copy()
        perturbed[feature] = baseline[feature]
        singles[feature] = _predict(model, perturbed)

    interactions: list[PairInteraction] = []
    for feature_a, feature_b in combinations(features, 2):
        perturbed = row.copy()
        perturbed[feature_a] = baseline[feature_a]
        perturbed[feature_b] = baseline[feature_b]
        pair_probability = _predict(model, perturbed)
        residual = original - singles[feature_a] - singles[feature_b] + pair_probability
        residual = 0.0 if abs(residual) <= policy.neutral_epsilon else float(residual)
        interactions.append(
            PairInteraction(
                feature_a=feature_a,
                feature_b=feature_b,
                probability_residual=residual,
                magnitude=abs(residual),
                direction=(
                    "neutral"
                    if residual == 0
                    else "reinforcing"
                    if residual > 0
                    else "offsetting"
                ),
            )
        )

    interactions.sort(
        key=lambda item: (-item.magnitude, item.feature_a, item.feature_b)
    )
    excessive = [
        item for item in interactions if item.magnitude > policy.max_abs_interaction
    ]
    findings = [
        f"PAIR_INTERACTION_LIMIT_EXCEEDED:{item.feature_a}:{item.feature_b}"
        for item in excessive[: policy.max_findings]
    ]
    if len(excessive) > policy.max_findings:
        findings.append("FINDINGS_TRUNCATED")

    payload = {
        "schema_version": INTERACTION_AUDIT_VERSION,
        "accepted": not excessive,
        "model_revision": model_revision,
        "input_sha256": _digest({"features": features, "values": observed_values}),
        "baseline_sha256": _digest({"features": features, "values": baseline_values}),
        "original_probability": original,
        "evaluated_features": features,
        "prediction_calls": required_calls,
        "policy": policy,
        "maximum_abs_interaction": max(
            (item.magnitude for item in interactions), default=0.0
        ),
        "total_abs_interaction": sum(item.magnitude for item in interactions),
        "interactions": tuple(interactions),
        "findings": tuple(findings),
    }
    return InteractionAudit(**payload, evidence_sha256=_digest(_json_ready(payload)))


def _predict(model: ProbabilityModel, row: pd.DataFrame) -> float:
    try:
        value = float(model.predict_probability(row))
    except Exception as exc:
        raise InteractionAuditError("model prediction failed") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise InteractionAuditError("model probability must be finite and in [0, 1]")
    return value


def _json_scalar(value: Any, name: str) -> str | int | float | bool | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise InteractionAuditError(f"{name} must be a finite JSON scalar")


def _validate_identity(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or not value.isprintable()
        or len(value) > 128
    ):
        raise InteractionAuditError(
            f"{name} must be a printable identifier of at most 128 characters"
        )


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return _json_ready(value.to_dict())
    return value


def _digest(value: Any) -> str:
    rendered = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(rendered.encode()).hexdigest()
