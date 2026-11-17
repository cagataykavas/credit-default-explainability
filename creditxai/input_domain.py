from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


class InputDomainError(ValueError):
    """Raised when numeric scoring evidence cannot be safely evaluated."""


@dataclass(frozen=True)
class NumericDomainProfile:
    feature: str
    lower_quantile: float
    upper_quantile: float
    reviewed_lower: float
    reviewed_upper: float
    observed_minimum: float
    observed_maximum: float
    finite_training_values: int


@dataclass(frozen=True)
class NumericDomainFinding:
    feature: str
    code: str


@dataclass(frozen=True)
class NumericDomainAudit:
    schema_version: int
    status: str
    checked_features: int
    missing_features: tuple[str, ...]
    findings: tuple[NumericDomainFinding, ...]
    reviewed_quantiles: tuple[float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "missing_features": list(self.missing_features),
            "findings": [asdict(finding) for finding in self.findings],
            "reviewed_quantiles": list(self.reviewed_quantiles),
        }


def fit_numeric_domain_profiles(
    frame: pd.DataFrame,
    features: list[str],
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> dict[str, NumericDomainProfile]:
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise InputDomainError("quantiles must satisfy 0 <= lower < upper <= 1")
    if not features or len(features) != len(set(features)):
        raise InputDomainError("numeric features must be non-empty and unique")

    profiles: dict[str, NumericDomainProfile] = {}
    for feature in features:
        if feature not in frame:
            raise InputDomainError(f"missing numeric training feature: {feature}")
        try:
            values = pd.to_numeric(frame[feature], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise InputDomainError(
                f"training feature {feature!r} must be numeric"
            ) from exc
        if np.isinf(values).any():
            raise InputDomainError(
                f"training feature {feature!r} contains infinite values"
            )
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise InputDomainError(f"training feature {feature!r} has no finite values")
        profiles[feature] = NumericDomainProfile(
            feature=feature,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            reviewed_lower=float(np.quantile(finite, lower_quantile)),
            reviewed_upper=float(np.quantile(finite, upper_quantile)),
            observed_minimum=float(np.min(finite)),
            observed_maximum=float(np.max(finite)),
            finite_training_values=len(finite),
        )
    return profiles


def _validate_profile(feature: str, profile: NumericDomainProfile) -> None:
    if not isinstance(profile, NumericDomainProfile) or profile.feature != feature:
        raise InputDomainError(f"invalid numeric domain profile for {feature!r}")
    if not 0.0 <= profile.lower_quantile < profile.upper_quantile <= 1.0:
        raise InputDomainError(
            f"numeric domain profile for {feature!r} has invalid quantiles"
        )
    bounds = (
        profile.observed_minimum,
        profile.reviewed_lower,
        profile.reviewed_upper,
        profile.observed_maximum,
    )
    if not all(math.isfinite(value) for value in bounds):
        raise InputDomainError(f"numeric domain profile for {feature!r} is not finite")
    if not bounds[0] <= bounds[1] <= bounds[2] <= bounds[3]:
        raise InputDomainError(
            f"numeric domain profile for {feature!r} has invalid bounds"
        )
    if (
        isinstance(profile.finite_training_values, bool)
        or not isinstance(profile.finite_training_values, int)
        or profile.finite_training_values < 1
    ):
        raise InputDomainError(
            f"numeric domain profile for {feature!r} has invalid support"
        )


def audit_numeric_input_domain(
    row: pd.DataFrame,
    profiles: dict[str, NumericDomainProfile],
) -> NumericDomainAudit:
    """Surface numeric extrapolation risk before emitting a local explanation."""

    if len(row) != 1:
        raise InputDomainError("numeric domain audit requires exactly one scoring row")
    if not profiles:
        raise InputDomainError("numeric domain profiles are required")

    findings: list[NumericDomainFinding] = []
    missing: list[str] = []
    checked = 0
    quantile_policies: set[tuple[float, float]] = set()
    for feature, profile in profiles.items():
        _validate_profile(feature, profile)
        quantile_policies.add((profile.lower_quantile, profile.upper_quantile))
        if feature not in row:
            raise InputDomainError(f"scoring row is missing numeric feature: {feature}")
        value = row.iloc[0][feature]
        try:
            is_missing = bool(pd.isna(value))
        except (TypeError, ValueError) as exc:
            raise InputDomainError(
                f"scoring feature {feature!r} must be a scalar"
            ) from exc
        if is_missing:
            missing.append(feature)
            continue
        if isinstance(value, (bool, np.bool_)):
            raise InputDomainError(
                f"scoring feature {feature!r} must be numeric, not boolean"
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise InputDomainError(
                f"scoring feature {feature!r} must be numeric"
            ) from exc
        if not math.isfinite(numeric):
            raise InputDomainError(f"scoring feature {feature!r} must be finite")
        checked += 1
        if numeric < profile.observed_minimum or numeric > profile.observed_maximum:
            findings.append(
                NumericDomainFinding(feature, "outside_observed_training_range")
            )
        elif numeric < profile.reviewed_lower or numeric > profile.reviewed_upper:
            findings.append(
                NumericDomainFinding(feature, "outside_reviewed_quantile_range")
            )

    if len(quantile_policies) != 1:
        raise InputDomainError(
            "numeric domain profiles use inconsistent quantile policies"
        )
    reviewed_quantiles = next(iter(quantile_policies))

    if findings:
        status = "review_recommended"
    elif missing:
        status = "partial_missing_values"
    else:
        status = "within_reviewed_domain"
    return NumericDomainAudit(
        schema_version=1,
        status=status,
        checked_features=checked,
        missing_features=tuple(sorted(missing)),
        findings=tuple(sorted(findings, key=lambda finding: finding.feature)),
        reviewed_quantiles=reviewed_quantiles,
    )
