from __future__ import annotations

import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import app
from creditxai.counterfactual import find_actionable_counterfactual
from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.explain import (
    explain_decision,
    local_perturbation_attribution,
    reason_codes,
)
from creditxai.model import CreditRiskModel


def test_model_probability_and_explanation():
    frame = synthetic_credit_data(rows=2200, seed=42)
    model, metrics = CreditRiskModel.fit(frame)
    row = frame.drop(columns="default").iloc[[100]][FEATURES]
    probability = model.predict_probability(row)
    explanation = explain_decision(model, row)
    assert 0.0 <= probability <= 1.0
    assert metrics["roc_auc"] > 0.5
    assert len(explanation["contributions"]) == len(FEATURES)


def test_counterfactual_never_changes_age_or_categories():
    frame = synthetic_credit_data(rows=2200, seed=8)
    model, _ = CreditRiskModel.fit(frame)
    row = (
        frame.drop(columns="default")
        .sort_values(["utilization", "debt_ratio"], ascending=False)
        .iloc[[0]][FEATURES]
    )
    result = find_actionable_counterfactual(model, row)
    assert "age" not in result.changed_features
    assert "employment" not in result.changed_features
    assert "housing" not in result.changed_features
    assert result.probability_after <= result.probability_before + 1e-12


def test_demo_api_returns_explanation_and_counterfactual():
    response = TestClient(app).get("/demo")
    assert response.status_code == 200
    payload = response.json()
    assert "explanation" in payload
    assert "counterfactual" in payload


class ControlledProbabilityModel:
    def __init__(self) -> None:
        self.baseline = {
            "income": 90.0,
            "debt_ratio": 0.3,
            "late_payments": 1,
            "utilization": 0.4,
            "account_age_months": 12,
            "age": 30,
            "employment": "student",
            "housing": "owned",
        }

    def baseline_row(self) -> dict[str, object]:
        return dict(self.baseline)

    def predict_probability(self, row: pd.DataFrame) -> float:
        changed = [
            feature
            for feature in FEATURES
            if row.iloc[0][feature] == self.baseline[feature]
        ]
        if not changed:
            return 0.5
        return {
            "income": 0.4999995,
            "debt_ratio": 0.48,
            "utilization": 0.52,
        }.get(changed[0], 0.5)


def controlled_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "income": 100.0,
                "debt_ratio": 0.4,
                "late_payments": 2,
                "utilization": 0.5,
                "account_age_months": 24,
                "age": 35,
                "employment": "salaried",
                "housing": "rent",
            }
        ]
    )


def test_local_attribution_treats_negligible_deltas_as_neutral():
    contributions = local_perturbation_attribution(
        ControlledProbabilityModel(), controlled_row()
    )
    by_feature = {item.feature: item for item in contributions}

    assert by_feature["income"].probability_delta == pytest.approx(5e-7)
    assert by_feature["income"].direction == "neutral"
    assert by_feature["debt_ratio"].direction == "increases_risk"
    assert by_feature["utilization"].direction == "decreases_risk"
    assert reason_codes(contributions) == ["debt-to-income burden"]


@pytest.mark.parametrize("tolerance", [-1.0, 1.1, math.nan, math.inf])
def test_local_attribution_rejects_invalid_direction_tolerance(tolerance):
    with pytest.raises(ValueError, match="direction_tolerance"):
        local_perturbation_attribution(
            ControlledProbabilityModel(),
            controlled_row(),
            direction_tolerance=tolerance,
        )


@pytest.mark.parametrize("tolerance", [True, "0.01"])
def test_local_attribution_rejects_non_numeric_direction_tolerance(tolerance):
    with pytest.raises(TypeError, match="direction_tolerance"):
        local_perturbation_attribution(
            ControlledProbabilityModel(),
            controlled_row(),
            direction_tolerance=tolerance,
        )


@pytest.mark.parametrize("probability", [math.nan, -0.01, 1.01])
def test_local_attribution_rejects_invalid_model_output(probability):
    model = ControlledProbabilityModel()
    model.predict_probability = lambda row: probability

    with pytest.raises(ValueError, match="original prediction"):
        local_perturbation_attribution(model, controlled_row())
