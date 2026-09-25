from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest
from creditxai.additivity import AdditivityPolicy, audit_exact_additivity
from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.model import CreditRiskModel


class InteractionModel:
    def baseline_row(self) -> dict[str, float]:
        return {"a": 0.0, "b": 0.0}

    def predict_probability(self, row: pd.DataFrame) -> float:
        a = float(row.iloc[0]["a"])
        b = float(row.iloc[0]["b"])
        return 0.1 + 0.2 * a + 0.3 * b + 0.1 * a * b


class CategoryModel:
    def baseline_row(self) -> dict[str, object]:
        return {"amount": 0.0, "segment": "standard"}

    def predict_probability(self, row: pd.DataFrame) -> float:
        amount = float(row.iloc[0]["amount"])
        segment = str(row.iloc[0]["segment"])
        return 0.1 + 0.2 * amount + (0.15 if segment == "priority" else 0.0)


class DriftingModel(InteractionModel):
    def __init__(self) -> None:
        self.calls = 0

    def predict_probability(self, row: pd.DataFrame) -> float:
        self.calls += 1
        return super().predict_probability(row) + self.calls * 1e-5


def test_exact_values_reconcile_probability_with_interaction() -> None:
    report = audit_exact_additivity(
        InteractionModel(), pd.DataFrame([{"a": 1.0, "b": 1.0}]), features=("a", "b")
    )
    values = {item.feature: item for item in report.contributions}

    assert report.passed
    assert report.coalition_count == 4
    assert report.baseline_probability == pytest.approx(0.1)
    assert report.observed_probability == pytest.approx(0.7)
    assert report.probability_delta == pytest.approx(0.6)
    assert values["a"].shapley_value == pytest.approx(0.25)
    assert values["b"].shapley_value == pytest.approx(0.35)
    assert report.shapley_sum == pytest.approx(report.probability_delta)
    assert abs(report.additivity_residual) < 1e-12


def test_leave_one_out_residual_exposes_interaction_double_counting() -> None:
    report = audit_exact_additivity(
        InteractionModel(), pd.DataFrame([{"a": 1.0, "b": 1.0}]), features=("a", "b")
    )

    assert report.leave_one_out_sum == pytest.approx(0.7)
    assert report.leave_one_out_residual == pytest.approx(-0.1)


def test_mixed_numeric_and_categorical_features_are_supported() -> None:
    report = audit_exact_additivity(
        CategoryModel(),
        pd.DataFrame([{"amount": 1.0, "segment": "priority"}]),
        features=("amount", "segment"),
    )
    values = {item.feature: item.shapley_value for item in report.contributions}

    assert report.passed
    assert values == pytest.approx({"amount": 0.2, "segment": 0.15})


def test_real_credit_model_produces_additive_finite_evidence() -> None:
    frame = synthetic_credit_data(rows=1_200, seed=19)
    model, _ = CreditRiskModel.fit(frame)
    row = frame.drop(columns="default").iloc[[1_100]][FEATURES]

    report = audit_exact_additivity(model, row)

    assert report.passed
    assert report.feature_count == len(FEATURES)
    assert report.coalition_count == 256
    assert abs(report.additivity_residual) <= 1e-10
    assert len(report.contributions) == len(FEATURES)


def test_report_is_deterministic_and_json_ready() -> None:
    row = pd.DataFrame([{"a": 1.0, "b": 1.0}])

    first = audit_exact_additivity(InteractionModel(), row, features=("a", "b"))
    second = audit_exact_additivity(InteractionModel(), row, features=("a", "b"))

    assert first == second
    assert len(first.evidence_sha256) == 64
    json.dumps(first.to_dict(), allow_nan=False)


def test_prediction_drift_rejects_the_audit() -> None:
    report = audit_exact_additivity(
        DriftingModel(),
        pd.DataFrame([{"a": 1.0, "b": 1.0}]),
        features=("a", "b"),
        policy=AdditivityPolicy(max_additivity_error=1.0, max_repeatability_error=1e-8),
    )

    assert not report.passed
    assert "BASELINE_PREDICTION_DRIFT" in report.findings
    assert "OBSERVED_PREDICTION_DRIFT" in report.findings


@pytest.mark.parametrize(
    "row",
    [
        pd.DataFrame([{"a": float("nan"), "b": 1.0}]),
        pd.DataFrame([{"a": True, "b": 1.0}]),
        pd.DataFrame([{"a": 1.0}]),
        pd.DataFrame([{"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 0.0}]),
    ],
)
def test_malformed_observations_fail_closed(row: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        audit_exact_additivity(InteractionModel(), row, features=("a", "b"))


def test_invalid_feature_contracts_fail_closed() -> None:
    row = pd.DataFrame([{"a": 1.0, "b": 1.0}])

    with pytest.raises(ValueError, match="unique"):
        audit_exact_additivity(InteractionModel(), row, features=("a", "a"))
    with pytest.raises(ValueError, match="feature count"):
        audit_exact_additivity(InteractionModel(), row, features=())
    with pytest.raises(ValueError, match="coalition count"):
        audit_exact_additivity(
            InteractionModel(),
            row,
            features=("a", "b"),
            policy=AdditivityPolicy(max_coalitions=2),
        )


@pytest.mark.parametrize(
    "policy",
    [
        AdditivityPolicy(max_features=1),
        AdditivityPolicy(max_coalitions=2),
        AdditivityPolicy(max_string_chars=1),
        AdditivityPolicy(max_additivity_error=0.0),
        AdditivityPolicy(max_repeatability_error=0.0),
    ],
)
def test_policy_boundary_values_are_supported(policy: AdditivityPolicy) -> None:
    assert policy


@pytest.mark.parametrize(
    "changes",
    [
        {"max_features": True},
        {"max_coalitions": 1},
        {"max_string_chars": 0},
        {"max_additivity_error": float("nan")},
        {"max_repeatability_error": -1.0},
    ],
)
def test_invalid_policies_fail_closed(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(AdditivityPolicy(), **changes)


class InvalidProbabilityModel(InteractionModel):
    def __init__(self, value: object) -> None:
        self.value = value

    def predict_probability(self, row: pd.DataFrame) -> float:
        return self.value  # type: ignore[return-value]


@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.1, "0.5", True])
def test_invalid_model_probabilities_fail_closed(value: object) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        audit_exact_additivity(
            InvalidProbabilityModel(value),
            pd.DataFrame([{"a": 1.0, "b": 1.0}]),
            features=("a", "b"),
        )
