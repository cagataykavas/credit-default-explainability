from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import FEATURES
from .model import NUMERIC, CreditRiskModel
from .reason_policy import governed_reason_codes


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
) -> list[FeatureContribution]:
    """Estimate local contribution by replacing one feature at a time with a baseline.

    This is intentionally model-agnostic and easy to audit. It is not claimed to be
    a Shapley-value estimator: feature dependence and interactions can make one-at-a-
    time perturbations differ from causal or SHAP interpretations.
    """
    original_probability = model.predict_probability(row)
    baseline = model.baseline_row()
    contributions: list[FeatureContribution] = []
    for feature in FEATURES:
        perturbed = row.copy()
        if feature in NUMERIC:
            perturbed[feature] = perturbed[feature].astype(float)
        perturbed.loc[perturbed.index[0], feature] = baseline[feature]
        probability_without_feature_value = model.predict_probability(perturbed)
        delta = original_probability - probability_without_feature_value
        contributions.append(
            FeatureContribution(
                feature=feature,
                observed=_python_scalar(row.iloc[0][feature]),
                baseline=_python_scalar(baseline[feature]),
                probability_delta=float(delta),
                direction="increases_risk" if delta > 0 else "decreases_risk",
            )
        )
    return sorted(contributions, key=lambda item: abs(item.probability_delta), reverse=True)


def reason_codes(contributions: list[FeatureContribution], top_k: int = 4) -> list[str]:
    """Return governed labels for backward-compatible library callers."""
    selected = governed_reason_codes(contributions, top_k=top_k, materiality=0.0)
    return list(selected.reason_codes)


def explain_decision(model: CreditRiskModel, row: pd.DataFrame, threshold: float = 0.35) -> dict[str, Any]:
    probability = model.predict_probability(row)
    contributions = local_perturbation_attribution(model, row)
    decision = "manual_review" if probability >= threshold else "lower_risk_path"
    governed_reasons = governed_reason_codes(
        contributions,
        require_reason=decision == "manual_review",
    )
    return {
        "default_probability": probability,
        "threshold": threshold,
        "decision": decision,
        "reason_codes": list(governed_reasons.reason_codes),
        "reason_code_features": list(governed_reasons.source_features),
        "reason_code_policy": governed_reasons.policy_evidence(),
        "local_attribution_method": "one_feature_baseline_perturbation",
        "contributions": [item.as_dict() for item in contributions],
        "caveat": (
            "Perturbation deltas describe model sensitivity around this row; they are not causal explanations "
            "and the governed labels are not legal adverse-action reasons without domain/legal validation."
        ),
    }
