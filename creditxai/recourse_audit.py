"""Fail-closed validation for counterfactual credit-risk recourse proposals.

The counterfactual search and its audit are deliberately separate: a search may
propose an action set, but it must not attest to its own validity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Protocol

import pandas as pd

from .data import FEATURES


class ProbabilityModel(Protocol):
    def predict_probability(self, row: pd.DataFrame) -> float: ...


class RecourseAuditError(ValueError):
    """Malformed evidence that cannot be evaluated as a policy decision."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RecourseAuditPolicy:
    target_probability: float = 0.25
    min_probability_reduction: float = 0.01
    max_changed_features: int = 4
    max_normalized_cost: float = 12.0
    comparison_tolerance: float = 1e-9


@dataclass(frozen=True)
class FeatureDelta:
    feature: str
    before: float
    after: float
    delta: float
    normalized_cost: float


@dataclass(frozen=True)
class RecourseAuditReport:
    accepted: bool
    reason_codes: tuple[str, ...]
    probability_before: float
    probability_after: float
    target_probability: float
    probability_reduction: float
    normalized_cost: float
    changed_features: tuple[str, ...]
    no_op_changes: tuple[str, ...]
    immutable_changes: tuple[str, ...]
    direction_violations: tuple[str, ...]
    unnecessary_features: tuple[str, ...]
    deltas: tuple[FeatureDelta, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in (
            "reason_codes",
            "changed_features",
            "no_op_changes",
            "immutable_changes",
            "direction_violations",
            "unnecessary_features",
            "deltas",
        ):
            payload[field] = list(payload[field])
        return payload


# Direction is the only domain claim made here. Costs merely put heterogeneous
# changes on comparable scales and must be calibrated for a real portfolio.
_ACTION_RULES: dict[str, tuple[str, float]] = {
    "income": ("increase", 20_000.0),
    "debt_ratio": ("decrease", 0.10),
    "late_payments": ("decrease", 1.0),
    "utilization": ("decrease", 0.10),
}


def audit_counterfactual(
    model: ProbabilityModel,
    row: pd.DataFrame,
    changes: Mapping[str, Any],
    *,
    policy: RecourseAuditPolicy | None = None,
) -> RecourseAuditReport:
    """Audit a proposed action set against the model and a recourse policy.

    Minimality is inclusion-minimality over the supplied action set: each action
    is removed once and the candidate is rescored. It is not a proof of a global
    minimum over all possible values or features.
    """

    active_policy = policy or RecourseAuditPolicy()
    _validate_policy(active_policy)
    _validate_row(row)
    if not isinstance(changes, Mapping):
        raise RecourseAuditError("INVALID_CHANGES", "changes must be a mapping")

    unknown = sorted(
        (feature for feature in changes if feature not in FEATURES), key=str
    )
    if unknown:
        raise RecourseAuditError(
            "UNKNOWN_CHANGE_FEATURE",
            f"unknown change features: {', '.join(map(str, unknown))}",
        )

    candidate = row.copy(deep=True)
    position = candidate.index[0]
    for feature, value in changes.items():
        try:
            candidate.loc[position, feature] = value
        except (TypeError, ValueError) as exc:
            raise RecourseAuditError(
                "INVALID_FEATURE_VALUE", f"{feature} has an incompatible value"
            ) from exc
    _validate_row(candidate)

    before = _predict(model, row)
    after = _predict(model, candidate)
    tolerance = active_policy.comparison_tolerance
    changed = tuple(
        feature
        for feature in FEATURES
        if feature in changes
        and not _values_equal(
            row.iloc[0][feature], candidate.iloc[0][feature], tolerance
        )
    )
    no_op_changes = tuple(
        feature for feature in FEATURES if feature in changes and feature not in changed
    )
    immutable = tuple(feature for feature in changed if feature not in _ACTION_RULES)
    direction_violations = tuple(
        feature
        for feature in changed
        if feature in _ACTION_RULES
        and not _direction_is_valid(
            _as_float(row.iloc[0][feature]),
            _as_float(candidate.iloc[0][feature]),
            _ACTION_RULES[feature][0],
            tolerance,
        )
    )

    deltas = tuple(
        _feature_delta(row, candidate, feature)
        for feature in changed
        if feature in _ACTION_RULES
    )
    normalized_cost = sum(delta.normalized_cost for delta in deltas)
    unnecessary = _find_unnecessary_features(
        model,
        row,
        candidate,
        changed,
        active_policy.target_probability,
    )

    reasons: list[str] = []
    if before <= active_policy.target_probability:
        reasons.append("ORIGINAL_ALREADY_MEETS_TARGET")
    if not changed:
        reasons.append("NO_EFFECTIVE_CHANGE")
    if no_op_changes:
        reasons.append("NO_OP_CHANGE")
    if len(changed) > active_policy.max_changed_features:
        reasons.append("TOO_MANY_FEATURES_CHANGED")
    if immutable:
        reasons.append("IMMUTABLE_FEATURE_CHANGED")
    if direction_violations:
        reasons.append("NON_ACTIONABLE_DIRECTION")
    if after > active_policy.target_probability:
        reasons.append("TARGET_NOT_REACHED")
    if before - after < active_policy.min_probability_reduction:
        reasons.append("INSUFFICIENT_PROBABILITY_REDUCTION")
    if normalized_cost > active_policy.max_normalized_cost + tolerance:
        reasons.append("ACTION_COST_EXCEEDED")
    if unnecessary:
        reasons.append("NON_MINIMAL_ACTION_SET")

    return RecourseAuditReport(
        accepted=not reasons,
        reason_codes=tuple(reasons),
        probability_before=before,
        probability_after=after,
        target_probability=active_policy.target_probability,
        probability_reduction=before - after,
        normalized_cost=normalized_cost,
        changed_features=changed,
        no_op_changes=no_op_changes,
        immutable_changes=immutable,
        direction_violations=direction_violations,
        unnecessary_features=unnecessary,
        deltas=deltas,
    )


def _validate_policy(policy: RecourseAuditPolicy) -> None:
    finite_fields = {
        "target_probability": policy.target_probability,
        "min_probability_reduction": policy.min_probability_reduction,
        "max_normalized_cost": policy.max_normalized_cost,
        "comparison_tolerance": policy.comparison_tolerance,
    }
    if any(
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in finite_fields.values()
    ):
        raise RecourseAuditError(
            "INVALID_POLICY", "policy numeric fields must be finite"
        )
    if not 0.0 < policy.target_probability < 1.0:
        raise RecourseAuditError(
            "INVALID_POLICY", "target_probability must be between zero and one"
        )
    if (
        policy.min_probability_reduction < 0
        or policy.max_normalized_cost < 0
        or policy.comparison_tolerance < 0
    ):
        raise RecourseAuditError("INVALID_POLICY", "policy budgets cannot be negative")
    if (
        not isinstance(policy.max_changed_features, int)
        or isinstance(policy.max_changed_features, bool)
        or policy.max_changed_features < 1
    ):
        raise RecourseAuditError(
            "INVALID_POLICY", "max_changed_features must be a positive integer"
        )


def _validate_row(row: pd.DataFrame) -> None:
    if not isinstance(row, pd.DataFrame) or len(row) != 1:
        raise RecourseAuditError("INVALID_ROW_SHAPE", "row must be a one-row DataFrame")
    if not row.columns.is_unique:
        raise RecourseAuditError("DUPLICATE_COLUMN", "row columns must be unique")
    missing = [feature for feature in FEATURES if feature not in row.columns]
    if missing:
        raise RecourseAuditError(
            "MISSING_FEATURE", f"missing features: {', '.join(missing)}"
        )

    values = row.iloc[0]
    for feature in ("income", "debt_ratio", "utilization"):
        _require_finite_real(values[feature], feature)
    for feature in ("late_payments", "account_age_months", "age"):
        value = values[feature]
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise RecourseAuditError(
                "INVALID_FEATURE_VALUE", f"{feature} must be an integer"
            )
    if float(values["income"]) <= 0:
        raise RecourseAuditError("INVALID_FEATURE_VALUE", "income must be positive")
    for feature in ("debt_ratio", "utilization"):
        if not 0.0 <= float(values[feature]) <= 1.0:
            raise RecourseAuditError(
                "INVALID_FEATURE_VALUE", f"{feature} must be between zero and one"
            )
    if int(values["late_payments"]) < 0 or int(values["account_age_months"]) < 0:
        raise RecourseAuditError(
            "INVALID_FEATURE_VALUE", "count features cannot be negative"
        )
    if not 18 <= int(values["age"]) <= 100:
        raise RecourseAuditError(
            "INVALID_FEATURE_VALUE", "age must be between 18 and 100"
        )
    for feature in ("employment", "housing"):
        if not isinstance(values[feature], str) or not values[feature].strip():
            raise RecourseAuditError(
                "INVALID_FEATURE_VALUE", f"{feature} must be a non-empty string"
            )


def _require_finite_real(value: Any, feature: str) -> None:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise RecourseAuditError(
            "INVALID_FEATURE_VALUE", f"{feature} must be finite and numeric"
        )


def _predict(model: ProbabilityModel, row: pd.DataFrame) -> float:
    try:
        probability = model.predict_probability(row)
    except Exception as exc:
        raise RecourseAuditError(
            "MODEL_PREDICTION_FAILED", "model could not score recourse evidence"
        ) from exc
    if (
        not isinstance(probability, Real)
        or isinstance(probability, bool)
        or not math.isfinite(float(probability))
    ):
        raise RecourseAuditError(
            "INVALID_MODEL_PROBABILITY", "model probability must be finite"
        )
    result = float(probability)
    if not 0.0 <= result <= 1.0:
        raise RecourseAuditError(
            "INVALID_MODEL_PROBABILITY",
            "model probability must be between zero and one",
        )
    return result


def _values_equal(before: Any, after: Any, tolerance: float) -> bool:
    if (
        isinstance(before, Real)
        and not isinstance(before, bool)
        and isinstance(after, Real)
        and not isinstance(after, bool)
    ):
        return math.isclose(float(before), float(after), rel_tol=0.0, abs_tol=tolerance)
    return bool(before == after)


def _as_float(value: Any) -> float:
    return float(value)


def _direction_is_valid(
    before: float, after: float, direction: str, tolerance: float
) -> bool:
    if direction == "increase":
        return after > before + tolerance
    return after < before - tolerance


def _feature_delta(
    row: pd.DataFrame, candidate: pd.DataFrame, feature: str
) -> FeatureDelta:
    before = float(row.iloc[0][feature])
    after = float(candidate.iloc[0][feature])
    delta = after - before
    return FeatureDelta(
        feature=feature,
        before=before,
        after=after,
        delta=delta,
        normalized_cost=abs(delta) / _ACTION_RULES[feature][1],
    )


def _find_unnecessary_features(
    model: ProbabilityModel,
    row: pd.DataFrame,
    candidate: pd.DataFrame,
    changed: tuple[str, ...],
    target_probability: float,
) -> tuple[str, ...]:
    unnecessary: list[str] = []
    for feature in changed:
        without_feature = candidate.copy(deep=True)
        without_feature.loc[without_feature.index[0], feature] = row.iloc[0][feature]
        if _predict(model, without_feature) <= target_probability:
            unnecessary.append(feature)
    return tuple(unnecessary)
