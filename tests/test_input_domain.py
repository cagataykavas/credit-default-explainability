from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import app
from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.model import CreditRiskModel, InputDomainError


@pytest.fixture(scope="module")
def fitted_model() -> tuple[CreditRiskModel, object]:
    frame = synthetic_credit_data(rows=2200, seed=17)
    model, _ = CreditRiskModel.fit(frame)
    row = frame.drop(columns="default").iloc[[100]][FEATURES].copy()
    return model, row


def test_model_records_sorted_category_vocabulary(fitted_model) -> None:
    model, _ = fitted_model

    assert model.category_values["employment"] == (
        "salaried",
        "self_employed",
        "student",
        "unemployed",
    )
    assert model.category_values["housing"] == ("mortgage", "owned", "rent")


def test_supported_categories_are_scored(fitted_model) -> None:
    model, row = fitted_model

    probability = model.predict_probability(row)

    assert 0.0 <= probability <= 1.0


def test_unknown_category_fails_closed_with_structured_evidence(fitted_model) -> None:
    model, row = fitted_model
    row = row.copy()
    row.loc[row.index[0], "employment"] = "contractor"

    with pytest.raises(InputDomainError) as captured:
        model.predict_probability(row)

    payload = captured.value.as_dict()
    assert payload["code"] == "unsupported_category"
    assert payload["violations"] == [
        {
            "feature": "employment",
            "observed": "contractor",
            "allowed": ["salaried", "self_employed", "student", "unemployed"],
        }
    ]


def test_all_unknown_categories_are_reported_together(fitted_model) -> None:
    model, row = fitted_model
    row = row.copy()
    row.loc[row.index[0], "employment"] = "contractor"
    row.loc[row.index[0], "housing"] = "company_flat"

    with pytest.raises(InputDomainError) as captured:
        model.predict_probability(row)

    assert [item.feature for item in captured.value.violations] == [
        "employment",
        "housing",
    ]


def test_missing_category_remains_compatible_with_fitted_imputer(fitted_model) -> None:
    model, row = fitted_model
    row = row.copy()
    row.loc[row.index[0], "employment"] = np.nan

    assert 0.0 <= model.predict_probability(row) <= 1.0


def test_multiple_rows_are_rejected_before_scoring(fitted_model) -> None:
    model, row = fitted_model

    with pytest.raises(ValueError, match="exactly one row"):
        model.predict_probability(row.copy().loc[[row.index[0], row.index[0]]])


def test_api_rejects_unknown_category_without_emitting_explanation() -> None:
    response = TestClient(app).post(
        "/score",
        json={
            "income": 42000,
            "debt_ratio": 0.63,
            "late_payments": 3,
            "utilization": 0.82,
            "account_age_months": 24,
            "age": 34,
            "employment": "contractor",
            "housing": "rent",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "unsupported_category"
    assert detail["violations"][0]["feature"] == "employment"
    assert "explanation" not in response.json()
