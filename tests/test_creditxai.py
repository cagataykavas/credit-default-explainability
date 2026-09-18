from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app
from creditxai.counterfactual import find_actionable_counterfactual
from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.explain import explain_decision
from creditxai.model import CreditRiskModel
from creditxai.stability import audit_local_explanation_stability


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


def test_local_explanation_stability_audit_is_deterministic():
    frame = synthetic_credit_data(rows=2200, seed=17)
    model, _ = CreditRiskModel.fit(frame)
    row = frame.drop(columns="default").iloc[[250]][FEATURES]

    first = audit_local_explanation_stability(
        model,
        row,
        samples=20,
        relative_noise=0.005,
        top_k=4,
        seed=9,
    )
    second = audit_local_explanation_stability(
        model,
        row,
        samples=20,
        relative_noise=0.005,
        top_k=4,
        seed=9,
    )

    assert first == second
    assert 0.0 <= first.minimum_top_k_overlap <= first.mean_top_k_overlap <= 1.0
    assert 0.0 <= first.minimum_direction_agreement <= first.mean_direction_agreement <= 1.0
    assert first.probability_standard_deviation >= 0.0
    assert first.maximum_probability_drift >= 0.0


def test_stability_audit_rejects_invalid_sampling_policy():
    frame = synthetic_credit_data(rows=2200, seed=19)
    model, _ = CreditRiskModel.fit(frame)
    row = frame.drop(columns="default").iloc[[10]][FEATURES]

    for kwargs in (
        {"samples": 0},
        {"relative_noise": 0.0},
        {"top_k": 0},
        {"top_k": len(FEATURES) + 1},
    ):
        try:
            audit_local_explanation_stability(model, row, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
