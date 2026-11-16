"""Audit local explanation robustness under single-feature missingness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .data import FEATURES
from .explain import local_perturbation_attribution


class ExplainableProbabilityModel(Protocol):
    def baseline_row(self) -> dict[str, Any]: ...

    def predict_probability(self, row: pd.DataFrame) -> float: ...


@dataclass(frozen=True)
class MissingnessPolicy:
    """Fail-closed thresholds for a bounded single-feature audit."""

    top_k: int = 3
    max_probability_drift: float = 0.20
    min_top_k_overlap: float = 0.50
    min_direction_agreement: float = 0.75
    direction_epsilon: float = 1e-9
    max_scenarios: int = 16
    max_predictions: int = 10_000

    def __post_init__(self) -> None:
        integer_limits = {
            "top_k": (self.top_k, 1, len(FEATURES) - 1),
            "max_scenarios": (self.max_scenarios, 1, 64),
            "max_predictions": (self.max_predictions, 1, 1_000_000),
        }
        for name, (value, minimum, maximum) in integer_limits.items():
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        for name, value in {
            "max_probability_drift": self.max_probability_drift,
            "min_top_k_overlap": self.min_top_k_overlap,
            "min_direction_agreement": self.min_direction_agreement,
            "direction_epsilon": self.direction_epsilon,
        }.items():
            if (
                type(value) not in {int, float}
                or not isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be a finite number in [0, 1]")


@dataclass(frozen=True)
class MissingnessScenario:
    missing_feature: str
    probability: float
    absolute_probability_drift: float
    top_k_overlap: float
    direction_agreement: float
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "findings": list(self.findings)}


@dataclass(frozen=True)
class MissingnessAudit:
    passed: bool
    findings: tuple[str, ...]
    original_probability: float
    scenario_count: int
    worst_probability_drift: float
    worst_top_k_overlap: float
    worst_direction_agreement: float
    scenarios: tuple[MissingnessScenario, ...]
    evidence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "findings": list(self.findings),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }


def audit_explanation_missingness(
    model: ExplainableProbabilityModel,
    row: pd.DataFrame,
    *,
    mask_features: tuple[str, ...] = tuple(FEATURES),
    policy: MissingnessPolicy | None = None,
) -> MissingnessAudit:
    """Measure explanation stability after masking one input feature at a time.

    Rankings and direction agreement are compared only across the features that remain
    observed in each scenario. The audit reports sensitivity to the model's configured
    imputers; it does not simulate every production missing-data mechanism.
    """

    active_policy = policy or MissingnessPolicy()
    ordered_masks = _validate_inputs(row, mask_features, active_policy)
    estimated_predictions = (len(ordered_masks) + 1) * (len(FEATURES) + 2)
    if estimated_predictions > active_policy.max_predictions:
        raise ValueError("audit exceeds the configured prediction budget")

    original_probability = _predict(model, row)
    original = _contribution_map(model, row)
    scenarios: list[MissingnessScenario] = []
    aggregate_findings: list[str] = []

    for missing_feature in ordered_masks:
        masked = row.copy(deep=True)
        masked.loc[masked.index[0], missing_feature] = np.nan
        probability = _predict(model, masked)
        perturbed = _contribution_map(model, masked)
        remaining = tuple(feature for feature in FEATURES if feature != missing_feature)
        top_count = min(active_policy.top_k, len(remaining))
        original_top = set(_ranked_features(original, remaining)[:top_count])
        perturbed_top = set(_ranked_features(perturbed, remaining)[:top_count])
        top_k_overlap = len(original_top & perturbed_top) / top_count
        direction_agreement = sum(
            _direction(original[feature], active_policy.direction_epsilon)
            == _direction(perturbed[feature], active_policy.direction_epsilon)
            for feature in remaining
        ) / len(remaining)
        probability_drift = abs(probability - original_probability)

        findings: list[str] = []
        if probability_drift > active_policy.max_probability_drift:
            findings.append("PROBABILITY_DRIFT_EXCEEDED")
        if top_k_overlap < active_policy.min_top_k_overlap:
            findings.append("TOP_K_OVERLAP_BELOW_MINIMUM")
        if direction_agreement < active_policy.min_direction_agreement:
            findings.append("DIRECTION_AGREEMENT_BELOW_MINIMUM")
        aggregate_findings.extend(f"{code}:{missing_feature}" for code in findings)
        scenarios.append(
            MissingnessScenario(
                missing_feature=missing_feature,
                probability=probability,
                absolute_probability_drift=probability_drift,
                top_k_overlap=top_k_overlap,
                direction_agreement=direction_agreement,
                findings=tuple(findings),
            )
        )

    evidence = {
        "schema_version": 1,
        "row_sha256": _row_digest(row),
        "policy": asdict(active_policy),
        "original_probability": original_probability,
        "scenarios": [scenario.to_dict() for scenario in scenarios],
    }
    evidence_sha256 = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return MissingnessAudit(
        passed=not aggregate_findings,
        findings=tuple(aggregate_findings),
        original_probability=original_probability,
        scenario_count=len(scenarios),
        worst_probability_drift=max(
            item.absolute_probability_drift for item in scenarios
        ),
        worst_top_k_overlap=min(item.top_k_overlap for item in scenarios),
        worst_direction_agreement=min(item.direction_agreement for item in scenarios),
        scenarios=tuple(scenarios),
        evidence_sha256=evidence_sha256,
    )


def _validate_inputs(
    row: pd.DataFrame,
    mask_features: tuple[str, ...],
    policy: MissingnessPolicy,
) -> tuple[str, ...]:
    if len(row) != 1:
        raise ValueError("row must contain exactly one observation")
    missing_columns = set(FEATURES) - set(row.columns)
    if missing_columns:
        raise ValueError("row is missing required model features")
    if not mask_features or len(mask_features) > policy.max_scenarios:
        raise ValueError("mask feature count is empty or exceeds the configured budget")
    if len(mask_features) != len(set(mask_features)):
        raise ValueError("mask features must be unique")
    mask_set = set(mask_features)
    unknown = mask_set - set(FEATURES)
    if unknown:
        raise ValueError("mask features must belong to the model feature schema")
    for feature in FEATURES:
        value = row.iloc[0][feature]
        if pd.isna(value):
            raise ValueError("the reference row must not contain missing values")
        if isinstance(value, str):
            if not value or len(value) > 256 or any(ord(char) < 32 for char in value):
                raise ValueError("categorical values must be non-empty bounded strings")
        elif (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not isfinite(float(value))
        ):
            raise ValueError("numeric values must be finite")
    return tuple(feature for feature in FEATURES if feature in mask_set)


def _predict(model: ExplainableProbabilityModel, row: pd.DataFrame) -> float:
    probability = model.predict_probability(row)
    if (
        not isinstance(probability, Real)
        or isinstance(probability, bool)
        or not isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("model probabilities must be finite numbers in [0, 1]")
    return float(probability)


def _contribution_map(
    model: ExplainableProbabilityModel, row: pd.DataFrame
) -> dict[str, float]:
    contributions = local_perturbation_attribution(model, row)
    if {item.feature for item in contributions} != set(FEATURES):
        raise ValueError(
            "explanation contributions do not match the model feature schema"
        )
    result: dict[str, float] = {}
    for item in contributions:
        value = item.probability_delta
        if not isfinite(value):
            raise ValueError("explanation contributions must be finite")
        result[item.feature] = value
    return result


def _ranked_features(values: dict[str, float], features: tuple[str, ...]) -> list[str]:
    return sorted(features, key=lambda feature: (-abs(values[feature]), feature))


def _direction(value: float, epsilon: float) -> int:
    return 1 if value > epsilon else -1 if value < -epsilon else 0


def _row_digest(row: pd.DataFrame) -> str:
    values = {feature: _json_scalar(row.iloc[0][feature]) for feature in FEATURES}
    payload = json.dumps(
        values, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
