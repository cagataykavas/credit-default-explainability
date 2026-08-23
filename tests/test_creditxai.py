from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app
from creditxai.counterfactual import find_actionable_counterfactual
from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.explain import explain_decision
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
    row = frame.drop(columns="default").sort_values(["utilization", "debt_ratio"], ascending=False).iloc[[0]][FEATURES]
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
