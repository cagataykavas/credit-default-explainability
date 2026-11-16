from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURES, TARGET

NUMERIC = [
    "income",
    "debt_ratio",
    "late_payments",
    "utilization",
    "account_age_months",
    "age",
]
CATEGORICAL = ["employment", "housing"]


@dataclass(frozen=True)
class CategoryDomainViolation:
    feature: str
    observed: str
    allowed: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "observed": self.observed,
            "allowed": list(self.allowed),
        }


class InputDomainError(ValueError):
    def __init__(self, violations: tuple[CategoryDomainViolation, ...]) -> None:
        self.violations = violations
        features = ", ".join(item.feature for item in violations)
        super().__init__(f"unsupported categorical value for: {features}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": "unsupported_category",
            "message": str(self),
            "violations": [item.as_dict() for item in self.violations],
        }


def build_pipeline() -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ]
    )
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
    category_values: dict[str, tuple[str, ...]]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> tuple[CreditRiskModel, dict[str, float]]:
        split = int(len(frame) * 0.8)
        train = frame.iloc[:split].copy()
        test = frame.iloc[split:].copy()
        pipeline = build_pipeline()
        pipeline.fit(train[FEATURES], train[TARGET])
        probabilities = pipeline.predict_proba(test[FEATURES])[:, 1]
        metrics = {
            "roc_auc": float(roc_auc_score(test[TARGET], probabilities)),
            "average_precision": float(
                average_precision_score(test[TARGET], probabilities)
            ),
            "brier_score": float(brier_score_loss(test[TARGET], probabilities)),
            "test_default_rate": float(test[TARGET].mean()),
        }
        medians = {name: float(train[name].median()) for name in NUMERIC}
        modes = {
            name: str(train[name].mode(dropna=True).iloc[0]) for name in CATEGORICAL
        }
        category_values = {
            name: tuple(sorted(map(str, train[name].dropna().unique())))
            for name in CATEGORICAL
        }
        return cls(pipeline, medians, modes, category_values), metrics

    def predict_probability(self, row: pd.DataFrame) -> float:
        self.validate_input_domain(row)
        return float(self.pipeline.predict_proba(row[FEATURES])[:, 1][0])

    def validate_input_domain(self, row: pd.DataFrame) -> None:
        missing = [feature for feature in FEATURES if feature not in row.columns]
        if missing:
            raise ValueError(f"missing model features: {', '.join(missing)}")
        if len(row) != 1:
            raise ValueError("credit scoring requires exactly one row")

        violations: list[CategoryDomainViolation] = []
        for feature in CATEGORICAL:
            observed = row.iloc[0][feature]
            if pd.isna(observed):
                continue
            value = str(observed)
            allowed = self.category_values.get(feature, ())
            if value not in allowed:
                violations.append(CategoryDomainViolation(feature, value, allowed))
        if violations:
            raise InputDomainError(tuple(violations))

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> CreditRiskModel:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("artifact is not a CreditRiskModel")
        if not getattr(obj, "category_values", None):
            raise TypeError(
                "artifact lacks categorical-domain metadata; retraining is required"
            )
        return obj

    def baseline_row(self) -> dict[str, Any]:
        return {**self.training_medians, **self.category_modes}
