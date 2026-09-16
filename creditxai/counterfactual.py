from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .model import CreditRiskModel


@dataclass(frozen=True)
class Counterfactual:
    probability_before: float
    probability_after: float
    changed_features: dict[str, Any]
    feasible: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_actionable_counterfactual(
    model: CreditRiskModel,
    row: pd.DataFrame,
    *,
    target_probability: float = 0.25,
) -> Counterfactual:
    """Greedy search over a small set of explicitly actionable synthetic features.

    Protected/identity features such as age are deliberately excluded from the search.
    This is an educational counterfactual mechanism, not prescriptive financial advice.
    """
    candidate = row.copy()
    before = model.predict_probability(candidate)
    changes: dict[str, Any] = {}
    if before <= target_probability:
        return Counterfactual(before, before, changes, True)

    original = candidate.iloc[0]
    proposals = [
        ("utilization", max(0.05, float(original["utilization"]) * 0.80)),
        ("debt_ratio", max(0.05, float(original["debt_ratio"]) * 0.82)),
        ("late_payments", max(0, int(original["late_payments"]) - 1)),
        ("income", float(original["income"]) * 1.10),
    ]

    for feature, proposed_value in proposals:
        trial = candidate.copy()
        trial.loc[trial.index[0], feature] = proposed_value
        if model.predict_probability(trial) < model.predict_probability(candidate):
            candidate = trial
            changes[feature] = proposed_value
        if model.predict_probability(candidate) <= target_probability:
            break

    after = model.predict_probability(candidate)
    return Counterfactual(before, after, changes, after <= target_probability)
