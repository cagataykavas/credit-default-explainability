from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURES, TARGET

NUMERIC = ["income", "debt_ratio", "late_payments", "utilization", "account_age_months", "age"]
CATEGORICAL = ["employment", "housing"]


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    classifier = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=1200, class_weight="balanced"),
        method="sigmoid",
        cv=4,
    )
    return Pipeline([("features", preprocessor), ("classifier", classifier)])


@dataclass
class CreditRiskModel:
    pipeline: Pipeline
    training_medians: dict[str, float]
    category_modes: dict[str, str]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> tuple["CreditRiskModel", dict[str, float]]:
        split = int(len(frame) * 0.8)
        train = frame.iloc[:split].copy()
        test = frame.iloc[split:].copy()
        pipeline = build_pipeline()
        pipeline.fit(train[FEATURES], train[TARGET])
        probabilities = pipeline.predict_proba(test[FEATURES])[:, 1]
        metrics = {
            "roc_auc": float(roc_auc_score(test[TARGET], probabilities)),
            "average_precision": float(average_precision_score(test[TARGET], probabilities)),
            "brier_score": float(brier_score_loss(test[TARGET], probabilities)),
            "test_default_rate": float(test[TARGET].mean()),
        }
        medians = {name: float(train[name].median()) for name in NUMERIC}
        modes = {name: str(train[name].mode(dropna=True).iloc[0]) for name in CATEGORICAL}
        return cls(pipeline, medians, modes), metrics

    def predict_probability(self, row: pd.DataFrame) -> float:
        return float(self.pipeline.predict_proba(row[FEATURES])[:, 1][0])

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "CreditRiskModel":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("artifact is not a CreditRiskModel")
        return obj

    def baseline_row(self) -> dict[str, Any]:
        return {**self.training_medians, **self.category_modes}
