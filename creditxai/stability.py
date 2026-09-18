from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean, pstdev

import numpy as np
import pandas as pd

from .explain import local_perturbation_attribution
from .model import NUMERIC, CreditRiskModel

BOUNDED_FEATURES = {"debt_ratio", "utilization"}
INTEGER_FEATURES = {"late_payments", "account_age_months", "age"}


@dataclass(frozen=True)
class StabilityAudit:
    samples: int
    seed: int
    relative_noise: float
    top_k: int
    mean_top_k_overlap: float
    minimum_top_k_overlap: float
    mean_direction_agreement: float
    minimum_direction_agreement: float
    probability_standard_deviation: float
    maximum_probability_drift: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def audit_local_explanation_stability(
    model: CreditRiskModel,
    row: pd.DataFrame,
    *,
    samples: int = 50,
    relative_noise: float = 0.01,
    top_k: int = 4,
    seed: int = 42,
) -> StabilityAudit:
    """Measure local attribution robustness under small valid numeric perturbations.

    This is a diagnostic, not a proof that an explanation is correct. It asks whether
    nearby rows receive similar top features and attribution directions.
    """
    if len(row) != 1:
        raise ValueError("row must contain exactly one observation")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not math.isfinite(relative_noise) or relative_noise <= 0:
        raise ValueError("relative_noise must be positive and finite")
    if not 1 <= top_k <= len(NUMERIC) + 2:
        raise ValueError("top_k is outside the supported feature range")

    baseline_probability = model.predict_probability(row)
    baseline = local_perturbation_attribution(model, row)
    baseline_top = baseline[:top_k]
    baseline_names = {item.feature for item in baseline_top}
    baseline_directions = {
        item.feature: _sign(item.probability_delta) for item in baseline_top
    }

    rng = np.random.default_rng(seed)
    overlaps: list[float] = []
    direction_agreements: list[float] = []
    probabilities: list[float] = []

    for _ in range(samples):
        neighbor = _numeric_neighbor(row, rng, relative_noise)
        probability = model.predict_probability(neighbor)
        contributions = local_perturbation_attribution(model, neighbor)
        candidate_top = {item.feature for item in contributions[:top_k]}
        candidate_directions = {
            item.feature: _sign(item.probability_delta) for item in contributions
        }

        overlaps.append(len(baseline_names & candidate_top) / top_k)
        comparable = [
            feature
            for feature, direction in baseline_directions.items()
            if direction != 0 and candidate_directions[feature] != 0
        ]
        direction_agreements.append(
            mean(
                candidate_directions[feature] == baseline_directions[feature]
                for feature in comparable
            )
            if comparable
            else 1.0
        )
        probabilities.append(probability)

    return StabilityAudit(
        samples=samples,
        seed=seed,
        relative_noise=relative_noise,
        top_k=top_k,
        mean_top_k_overlap=round(mean(overlaps), 6),
        minimum_top_k_overlap=round(min(overlaps), 6),
        mean_direction_agreement=round(mean(direction_agreements), 6),
        minimum_direction_agreement=round(min(direction_agreements), 6),
        probability_standard_deviation=round(pstdev(probabilities), 6),
        maximum_probability_drift=round(
            max(abs(value - baseline_probability) for value in probabilities),
            6,
        ),
    )


def _numeric_neighbor(
    row: pd.DataFrame,
    rng: np.random.Generator,
    relative_noise: float,
) -> pd.DataFrame:
    neighbor = row.copy()
    index = neighbor.index[0]
    for feature in NUMERIC:
        original = float(neighbor.at[index, feature])
        scale = max(abs(original), 1.0) * relative_noise
        value = original + float(rng.normal(0.0, scale))
        if feature in BOUNDED_FEATURES:
            value = float(np.clip(value, 0.0, 1.0))
        elif feature in INTEGER_FEATURES:
            value = float(max(0, round(value)))
        else:
            value = max(0.0, value)
        neighbor.at[index, feature] = value
    return neighbor


def _sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0
