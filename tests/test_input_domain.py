from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import app
from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.explain import explain_decision
from creditxai.input_domain import (
    InputDomainError,
    audit_numeric_input_domain,
    fit_numeric_domain_profiles,
)
from creditxai.model import CreditRiskModel


@pytest.fixture(scope="module")
def fitted_model() -> CreditRiskModel:
    model, _ = CreditRiskModel.fit(synthetic_credit_data(rows=1800, seed=19))
    return model


def baseline_row(model: CreditRiskModel) -> pd.DataFrame:
    return pd.DataFrame([model.baseline_row()])[FEATURES]


def test_baseline_row_is_inside_reviewed_numeric_domain(
    fitted_model: CreditRiskModel,
) -> None:
    audit = audit_numeric_input_domain(
        baseline_row(fitted_model), fitted_model.numeric_domain_profiles
    )

    assert audit.status == "within_reviewed_domain"
    assert audit.checked_features == 6
    assert not audit.findings


def test_explanation_surfaces_outside_training_range_without_raw_value(
    fitted_model: CreditRiskModel,
) -> None:
    row = baseline_row(fitted_model)
    maximum = fitted_model.numeric_domain_profiles["income"].observed_maximum
    row.loc[row.index[0], "income"] = maximum * 10

    explanation = explain_decision(fitted_model, row)
    domain = explanation["input_domain"]

    assert domain["status"] == "review_recommended"
    assert domain["findings"] == [
        {"feature": "income", "code": "outside_observed_training_range"}
    ]
    assert str(maximum * 10) not in str(domain)


def test_distinguishes_quantile_warning_from_observed_range_warning(
    fitted_model: CreditRiskModel,
) -> None:
    row = baseline_row(fitted_model)
    profile = fitted_model.numeric_domain_profiles["income"]
    row.loc[row.index[0], "income"] = (
        profile.observed_maximum + profile.reviewed_upper
    ) / 2

    audit = audit_numeric_input_domain(row, fitted_model.numeric_domain_profiles)

    assert audit.findings[0].code == "outside_reviewed_quantile_range"


def test_non_finite_scoring_value_fails_before_prediction(
    fitted_model: CreditRiskModel,
) -> None:
    row = baseline_row(fitted_model)
    row.loc[row.index[0], "income"] = float("inf")

    with pytest.raises(InputDomainError, match="must be finite"):
        explain_decision(fitted_model, row)


def test_missing_value_is_explicitly_not_counted_as_domain_checked(
    fitted_model: CreditRiskModel,
) -> None:
    row = baseline_row(fitted_model)
    row.loc[row.index[0], "income"] = None

    audit = audit_numeric_input_domain(row, fitted_model.numeric_domain_profiles)

    assert audit.status == "partial_missing_values"
    assert audit.checked_features == 5
    assert audit.missing_features == ("income",)


def test_malformed_profile_fails_closed(fitted_model: CreditRiskModel) -> None:
    profiles = dict(fitted_model.numeric_domain_profiles)
    profiles["income"] = replace(
        profiles["income"],
        observed_minimum=profiles["income"].observed_maximum + 1,
    )

    with pytest.raises(InputDomainError, match="invalid bounds"):
        audit_numeric_input_domain(baseline_row(fitted_model), profiles)


def test_model_round_trip_retains_numeric_domain_profiles(
    fitted_model: CreditRiskModel, tmp_path
) -> None:
    path = tmp_path / "model.joblib"
    fitted_model.save(str(path))

    loaded = CreditRiskModel.load(str(path))

    assert loaded.numeric_domain_profiles == fitted_model.numeric_domain_profiles


def test_audit_reports_the_quantiles_used_to_fit_profiles() -> None:
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    profiles = fit_numeric_domain_profiles(
        frame, ["value"], lower_quantile=0.2, upper_quantile=0.8
    )

    audit = audit_numeric_input_domain(pd.DataFrame({"value": [3.0]}), profiles)

    assert audit.reviewed_quantiles == (0.2, 0.8)


def test_scoring_api_returns_numeric_extrapolation_warning() -> None:
    response = TestClient(app).post(
        "/score",
        json={
            "income": 1_000_000_000,
            "debt_ratio": 0.4,
            "late_payments": 1,
            "utilization": 0.5,
            "account_age_months": 48,
            "age": 36,
            "employment": "salaried",
            "housing": "rent",
        },
    )

    assert response.status_code == 200
    domain = response.json()["explanation"]["input_domain"]
    assert domain["status"] == "review_recommended"
    assert {finding["feature"] for finding in domain["findings"]} == {"income"}
