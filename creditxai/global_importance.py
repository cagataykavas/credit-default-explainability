from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from .data import FEATURES, TARGET
from .model import CreditRiskModel


def permutation_feature_importance(
    model: CreditRiskModel,
    frame: pd.DataFrame,
    *,
    repeats: int = 8,
    seed: int = 42,
) -> list[dict[str, float | str]]:
    result = permutation_importance(
        model.pipeline,
        frame[FEATURES],
        frame[TARGET],
        scoring="roc_auc",
        n_repeats=repeats,
        random_state=seed,
    )
    rows = [
        {
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for feature, mean, std in zip(FEATURES, result.importances_mean, result.importances_std)
    ]
    return sorted(rows, key=lambda item: float(item["importance_mean"]), reverse=True)
