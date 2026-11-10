from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .data import FEATURES

RiskDirection = Literal["increasing_risk", "decreasing_risk"]

# These expectations describe the synthetic data-generating process, not lending policy.
# Age and categorical features are deliberately outside the governed default set.
DEFAULT_RISK_DIRECTIONS: dict[str, RiskDirection] = {
    "income": "decreasing_risk",
    "debt_ratio": "increasing_risk",
    "late_payments": "increasing_risk",
    "utilization": "increasing_risk",
    "account_age_months": "decreasing_risk",
}


class ProbabilityScorer(Protocol):
    def predict_probability(self, row: pd.DataFrame) -> float: ...


@dataclass(frozen=True)
class FeatureMonotonicity:
    feature: str
    expected_direction: RiskDirection
    grid_values: list[float]
    contexts_evaluated: int
    comparisons: int
    violations: int
    violation_rate: float
    worst_violation: float
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_configuration(
    cohort: pd.DataFrame,
    expectations: dict[str, RiskDirection],
    quantiles: tuple[float, ...],
    max_contexts: int,
    tolerance: float,
) -> None:
    if cohort.empty:
        raise ValueError("cohort must contain at least one row")
    missing = [feature for feature in FEATURES if feature not in cohort.columns]
    if missing:
        raise ValueError(f"cohort is missing model features: {missing}")
    if not expectations:
        raise ValueError("expectations must not be empty")
    unknown = sorted(set(expectations) - set(FEATURES))
    if unknown:
        raise ValueError(f"expectations contain unknown features: {unknown}")
    non_numeric = sorted(feature for feature in expectations if not is_numeric_dtype(cohort[feature]))
    if non_numeric:
        raise ValueError(f"expectations contain non-numeric features: {non_numeric}")
    allowed_directions = {"increasing_risk", "decreasing_risk"}
    invalid = sorted(feature for feature, direction in expectations.items() if direction not in allowed_directions)
    if invalid:
        raise ValueError(f"expectations contain invalid directions for: {invalid}")
    if not 1 <= max_contexts <= 256:
        raise ValueError("max_contexts must be between 1 and 256")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if not 2 <= len(quantiles) <= 20:
        raise ValueError("quantiles must contain between 2 and 20 values")
    if any(not np.isfinite(value) or not 0 < value < 1 for value in quantiles):
        raise ValueError("quantiles must be finite values strictly between 0 and 1")
    if any(left >= right for left, right in pairwise(quantiles)):
        raise ValueError("quantiles must be strictly increasing")


def audit_monotonic_responses(
    model: ProbabilityScorer,
    cohort: pd.DataFrame,
    *,
    expectations: dict[str, RiskDirection] | None = None,
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
    max_contexts: int = 32,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Audit governed score directions over held-out, one-feature perturbation grids.

    Each audited feature is swept over cohort-derived quantiles while every other
    value is fixed to a sampled row. Opposing adjacent score movements larger than
    ``tolerance`` are violations. This checks model behavior, not causality or the
    validity of the declared domain expectations.
    """
    governed = dict(DEFAULT_RISK_DIRECTIONS if expectations is None else expectations)
    _validate_configuration(cohort, governed, quantiles, max_contexts, tolerance)

    context_count = min(len(cohort), max_contexts)
    positions = np.linspace(0, len(cohort) - 1, num=context_count, dtype=int)
    contexts = cohort.iloc[positions][FEATURES].reset_index(drop=True)
    results: list[FeatureMonotonicity] = []

    for feature, direction in governed.items():
        values = pd.to_numeric(cohort[feature], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"audited feature {feature!r} contains non-finite values")
        grid = np.unique(np.quantile(values, quantiles, method="nearest")).astype(float)
        if len(grid) < 2:
            raise ValueError(f"audited feature {feature!r} needs at least two distinct quantile values")

        violations = 0
        worst_violation = 0.0
        comparisons = 0
        for _, context in contexts.iterrows():
            probabilities: list[float] = []
            for value in grid:
                trial_values = context.to_dict()
                trial_values[feature] = float(value)
                trial = pd.DataFrame([trial_values], columns=FEATURES)
                probability = float(model.predict_probability(trial))
                if not np.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("model returned a non-finite probability or a value outside [0, 1]")
                probabilities.append(probability)

            deltas = np.diff(probabilities)
            opposing = -deltas if direction == "increasing_risk" else deltas
            feature_violations = opposing > tolerance
            violations += int(feature_violations.sum())
            comparisons += len(deltas)
            if feature_violations.any():
                worst_violation = max(worst_violation, float(opposing[feature_violations].max()))

        results.append(
            FeatureMonotonicity(
                feature=feature,
                expected_direction=direction,
                grid_values=[float(value) for value in grid],
                contexts_evaluated=context_count,
                comparisons=comparisons,
                violations=violations,
                violation_rate=float(violations / comparisons),
                worst_violation=worst_violation,
                passed=violations == 0,
            )
        )

    return {
        "passed": all(result.passed for result in results),
        "contexts_evaluated": context_count,
        "tolerance": tolerance,
        "quantiles": list(quantiles),
        "features": [result.as_dict() for result in results],
        "caveat": (
            "This audit checks score direction under controlled, one-feature perturbations. "
            "It does not prove causal monotonicity, validate lending policy, or guarantee behavior outside the cohort range."
        ),
    }
