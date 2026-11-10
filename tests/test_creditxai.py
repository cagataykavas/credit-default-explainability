from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app
from creditxai.counterfactual import find_actionable_counterfactual
from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.explain import explain_decision
from creditxai.model import CreditRiskModel
from creditxai.monotonicity import audit_monotonic_responses
from creditxai.report import render_report


class LinearRiskScorer:
    def __init__(self, slope: float, feature: str = "utilization") -> None:
        self.slope = slope
        self.feature = feature

    def predict_probability(self, row):
        return 0.5 + self.slope * float(row.iloc[0][self.feature])


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


def test_monotonicity_audit_passes_consistent_score_direction():
    cohort = synthetic_credit_data(rows=100, seed=5).drop(columns="default")
    report = audit_monotonic_responses(
        LinearRiskScorer(slope=0.2),
        cohort,
        expectations={"utilization": "increasing_risk"},
        max_contexts=8,
    )
    feature = report["features"][0]
    assert report["passed"] is True
    assert feature["comparisons"] == 32
    assert feature["violations"] == 0


def test_monotonicity_audit_reports_opposing_score_movements():
    cohort = synthetic_credit_data(rows=100, seed=5).drop(columns="default")
    report = audit_monotonic_responses(
        LinearRiskScorer(slope=-0.2),
        cohort,
        expectations={"utilization": "increasing_risk"},
        max_contexts=4,
    )
    feature = report["features"][0]
    assert report["passed"] is False
    assert feature["violations"] == feature["comparisons"]
    assert feature["worst_violation"] > 0


def test_monotonicity_audit_handles_integer_feature_quantiles():
    cohort = synthetic_credit_data(rows=100, seed=5).drop(columns="default")
    report = audit_monotonic_responses(
        LinearRiskScorer(slope=0.02, feature="late_payments"),
        cohort,
        expectations={"late_payments": "increasing_risk"},
        max_contexts=4,
    )
    assert report["passed"] is True


def test_monotonicity_audit_rejects_unknown_governed_feature():
    cohort = synthetic_credit_data(rows=100, seed=5).drop(columns="default")
    try:
        audit_monotonic_responses(
            LinearRiskScorer(slope=0.2),
            cohort,
            expectations={"unknown": "increasing_risk"},
        )
    except ValueError as error:
        assert "unknown features" in str(error)
    else:
        raise AssertionError("unknown governed feature was accepted")


def test_html_report_includes_monotonicity_evidence(tmp_path):
    result = {
        "metrics": {"roc_auc": 0.7, "brier_score": 0.1},
        "global_importance": [{"feature": "utilization", "importance_mean": 0.2, "importance_std": 0.01}],
        "explanation": {
            "default_probability": 0.4,
            "decision": "manual_review",
            "reason_codes": ["credit utilization"],
            "contributions": [
                {
                    "feature": "utilization",
                    "observed": 0.8,
                    "probability_delta": 0.1,
                    "direction": "increases_risk",
                }
            ],
            "caveat": "synthetic",
        },
        "counterfactual": {"probability_after": 0.2, "changed_features": {}},
        "monotonicity_audit": {
            "passed": True,
            "contexts_evaluated": 4,
            "features": [
                {
                    "feature": "utilization",
                    "expected_direction": "increasing_risk",
                    "comparisons": 16,
                    "violations": 0,
                    "worst_violation": 0.0,
                    "passed": True,
                }
            ],
        },
    }
    output = render_report(result, tmp_path / "report.html")
    rendered = output.read_text(encoding="utf-8")
    assert "Governed monotonic-response audit" in rendered
    assert "16</td><td>0</td>" in rendered
