from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from .data import FEATURES
from .model import CreditRiskModel

DEFAULT_DIRECTION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    observed: Any
    baseline: Any
    probability_delta: float
    direction: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def local_perturbation_attribution(
    model: CreditRiskModel,
    row: pd.DataFrame,
    *,
    direction_tolerance: float = DEFAULT_DIRECTION_TOLERANCE,
) -> list[FeatureContribution]:
    """Estimate local contribution by replacing one feature at a time with a baseline.

    This is intentionally model-agnostic and easy to audit. It is not claimed to be
    a Shapley-value estimator: feature dependence and interactions can make one-at-a-
    time perturbations differ from causal or SHAP interpretations.
    """
    direction_tolerance = _validated_direction_tolerance(direction_tolerance)
    original_probability = _validated_probability(
        model.predict_probability(row), "original prediction"
    )
    baseline = model.baseline_row()
    contributions: list[FeatureContribution] = []
    for feature in FEATURES:
        perturbed = row.copy()
        perturbed.loc[perturbed.index[0], feature] = baseline[feature]
        probability_without_feature_value = _validated_probability(
            model.predict_probability(perturbed), f"{feature} baseline prediction"
        )
        delta = original_probability - probability_without_feature_value
        contributions.append(
            FeatureContribution(
                feature=feature,
                observed=_python_scalar(row.iloc[0][feature]),
                baseline=_python_scalar(baseline[feature]),
                probability_delta=float(delta),
                direction=_direction(delta, direction_tolerance),
            )
        )
    return sorted(
        contributions, key=lambda item: abs(item.probability_delta), reverse=True
    )


def reason_codes(contributions: list[FeatureContribution], top_k: int = 4) -> list[str]:
    labels = {
        "income": "income level",
        "debt_ratio": "debt-to-income burden",
        "late_payments": "late-payment history",
        "utilization": "credit utilization",
        "account_age_months": "account history length",
        "age": "age feature in the synthetic model",
        "employment": "employment category",
        "housing": "housing category",
    }
    positive = [item for item in contributions if item.direction == "increases_risk"]
    return [labels.get(item.feature, item.feature) for item in positive[:top_k]]


def explain_decision(
    model: CreditRiskModel,
    row: pd.DataFrame,
    threshold: float = 0.35,
    *,
    direction_tolerance: float = DEFAULT_DIRECTION_TOLERANCE,
) -> dict[str, Any]:
    direction_tolerance = _validated_direction_tolerance(direction_tolerance)
    probability = _validated_probability(
        model.predict_probability(row), "original prediction"
    )
    contributions = local_perturbation_attribution(
        model, row, direction_tolerance=direction_tolerance
    )
    return {
        "default_probability": probability,
        "threshold": threshold,
        "decision": "manual_review" if probability >= threshold else "lower_risk_path",
        "reason_codes": reason_codes(contributions),
        "local_attribution_method": "one_feature_baseline_perturbation",
        "direction_tolerance": direction_tolerance,
        "contributions": [item.as_dict() for item in contributions],
        "caveat": (
            "Perturbation deltas describe model sensitivity around this row; they are not causal explanations "
            "and should not be interpreted as legal adverse-action reasons without domain/legal validation."
        ),
    }


def _direction(delta: float, tolerance: float) -> str:
    if abs(delta) <= tolerance:
        return "neutral"
    return "increases_risk" if delta > 0 else "decreases_risk"


def _validated_direction_tolerance(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("direction_tolerance must be a finite number between 0 and 1")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("direction_tolerance must be a finite number between 0 and 1")
    return float(value)


def _validated_probability(value: float, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{context} must be a finite probability")
    probability = float(value)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"{context} must be a finite probability")
    return probability
