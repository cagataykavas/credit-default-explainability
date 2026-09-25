"""Exact baseline-coalition attribution with additivity and repeatability evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import comb, isfinite
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .data import FEATURES


class ProbabilityModel(Protocol):
    def baseline_row(self) -> dict[str, Any]: ...

    def predict_probability(self, row: pd.DataFrame) -> float: ...


@dataclass(frozen=True)
class AdditivityPolicy:
    max_features: int = 10
    max_coalitions: int = 1_024
    max_string_chars: int = 256
    max_additivity_error: float = 1e-10
    max_repeatability_error: float = 1e-12

    def __post_init__(self) -> None:
        integer_limits = {
            "max_features": (self.max_features, 1, 16),
            "max_coalitions": (self.max_coalitions, 2, 65_536),
            "max_string_chars": (self.max_string_chars, 1, 4_096),
        }
        for name, (value, minimum, maximum) in integer_limits.items():
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        for name, value in {
            "max_additivity_error": self.max_additivity_error,
            "max_repeatability_error": self.max_repeatability_error,
        }.items():
            if type(value) not in {int, float} or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ExactContribution:
    feature: str
    shapley_value: float
    leave_one_out_delta: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdditivityAudit:
    passed: bool
    findings: tuple[str, ...]
    feature_count: int
    coalition_count: int
    baseline_probability: float
    observed_probability: float
    probability_delta: float
    shapley_sum: float
    additivity_residual: float
    leave_one_out_sum: float
    leave_one_out_residual: float
    baseline_repeatability_error: float
    observed_repeatability_error: float
    contributions: tuple[ExactContribution, ...]
    evidence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "findings": list(self.findings),
            "contributions": [item.to_dict() for item in self.contributions],
        }


def audit_exact_additivity(
    model: ProbabilityModel,
    row: pd.DataFrame,
    *,
    features: tuple[str, ...] = tuple(FEATURES),
    policy: AdditivityPolicy | None = None,
) -> AdditivityAudit:
    """Compute exact Shapley values for the model probability against its baseline.

    The implementation enumerates every baseline/observed coalition, so it is bounded
    to small tabular feature sets and fails closed when the configured budget is exceeded.
    """

    active_policy = policy or AdditivityPolicy()
    feature_names, baseline = _validate_inputs(model, row, features, active_policy)
    coalition_count = 1 << len(feature_names)
    observed = {name: _scalar(row.iloc[0][name]) for name in feature_names}

    predictions: dict[int, float] = {}
    for mask in range(coalition_count):
        values = {
            name: observed[name] if mask & (1 << index) else _scalar(baseline[name])
            for index, name in enumerate(feature_names)
        }
        predictions[mask] = _predict(model, pd.DataFrame([values], columns=feature_names))

    empty_mask = 0
    full_mask = coalition_count - 1
    baseline_probability = predictions[empty_mask]
    observed_probability = predictions[full_mask]
    baseline_repeat = _predict(
        model,
        pd.DataFrame(
            [{name: _scalar(baseline[name]) for name in feature_names}], columns=feature_names
        ),
    )
    observed_repeat = _predict(
        model,
        pd.DataFrame([observed], columns=feature_names),
    )

    contributions: list[ExactContribution] = []
    for index, feature in enumerate(feature_names):
        feature_bit = 1 << index
        shapley_value = 0.0
        for mask in range(coalition_count):
            if mask & feature_bit:
                continue
            coalition_size = mask.bit_count()
            weight = 1.0 / (len(feature_names) * comb(len(feature_names) - 1, coalition_size))
            shapley_value += weight * (predictions[mask | feature_bit] - predictions[mask])
        leave_one_out = observed_probability - predictions[full_mask ^ feature_bit]
        contributions.append(
            ExactContribution(
                feature=feature,
                shapley_value=float(shapley_value),
                leave_one_out_delta=float(leave_one_out),
                direction=_direction(shapley_value),
            )
        )

    ordered = tuple(
        sorted(contributions, key=lambda item: (-abs(item.shapley_value), item.feature))
    )
    probability_delta = observed_probability - baseline_probability
    shapley_sum = sum(item.shapley_value for item in contributions)
    additivity_residual = probability_delta - shapley_sum
    leave_one_out_sum = sum(item.leave_one_out_delta for item in contributions)
    leave_one_out_residual = probability_delta - leave_one_out_sum
    baseline_repeatability_error = abs(baseline_probability - baseline_repeat)
    observed_repeatability_error = abs(observed_probability - observed_repeat)

    findings: list[str] = []
    if abs(additivity_residual) > active_policy.max_additivity_error:
        findings.append("ADDITIVITY_RESIDUAL_EXCEEDED")
    if baseline_repeatability_error > active_policy.max_repeatability_error:
        findings.append("BASELINE_PREDICTION_DRIFT")
    if observed_repeatability_error > active_policy.max_repeatability_error:
        findings.append("OBSERVED_PREDICTION_DRIFT")

    evidence = {
        "features": list(feature_names),
        "baseline_probability": baseline_probability,
        "observed_probability": observed_probability,
        "contributions": [item.to_dict() for item in ordered],
        "policy": asdict(active_policy),
    }
    digest = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return AdditivityAudit(
        passed=not findings,
        findings=tuple(findings),
        feature_count=len(feature_names),
        coalition_count=coalition_count,
        baseline_probability=baseline_probability,
        observed_probability=observed_probability,
        probability_delta=probability_delta,
        shapley_sum=shapley_sum,
        additivity_residual=additivity_residual,
        leave_one_out_sum=leave_one_out_sum,
        leave_one_out_residual=leave_one_out_residual,
        baseline_repeatability_error=baseline_repeatability_error,
        observed_repeatability_error=observed_repeatability_error,
        contributions=ordered,
        evidence_sha256=digest,
    )


def _validate_inputs(
    model: ProbabilityModel,
    row: pd.DataFrame,
    features: tuple[str, ...],
    policy: AdditivityPolicy,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    if len(row) != 1:
        raise ValueError("row must contain exactly one observation")
    if not features or len(features) > policy.max_features:
        raise ValueError("feature count is empty or exceeds the configured budget")
    if len(features) != len(set(features)):
        raise ValueError("feature names must be unique")
    if any(not isinstance(name, str) or not name or len(name) > 256 for name in features):
        raise ValueError("feature names must be non-empty bounded strings")
    coalition_count = 1 << len(features)
    if coalition_count > policy.max_coalitions:
        raise ValueError("coalition count exceeds the configured budget")

    missing_observed = set(features) - set(row.columns)
    baseline = dict(model.baseline_row())
    missing_baseline = set(features) - set(baseline)
    if missing_observed or missing_baseline:
        raise ValueError("observed row or baseline is missing required features")
    for name in features:
        _validate_value(_scalar(row.iloc[0][name]), policy)
        _validate_value(_scalar(baseline[name]), policy)
    return tuple(features), baseline


def _validate_value(value: Any, policy: AdditivityPolicy) -> None:
    if type(value) in {int, float}:
        if not isfinite(value):
            raise ValueError("feature values must be finite")
        return
    if isinstance(value, str):
        if (
            not value
            or len(value) > policy.max_string_chars
            or any(ord(char) < 32 for char in value)
        ):
            raise ValueError("categorical values must be non-empty bounded strings")
        return
    raise ValueError("feature values must be finite numbers or bounded strings")


def _predict(model: ProbabilityModel, row: pd.DataFrame) -> float:
    value = model.predict_probability(row)
    if type(value) not in {int, float} or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("model probabilities must be finite numbers in [0, 1]")
    return float(value)


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _direction(value: float) -> str:
    if value > 0:
        return "increases_risk"
    if value < 0:
        return "decreases_risk"
    return "neutral"
