from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.missingness import MissingnessPolicy, audit_explanation_missingness
from creditxai.model import CreditRiskModel


class StableModel:
    _weights: ClassVar[dict[str, float]] = {
        "income": 0.06,
        "debt_ratio": 0.05,
        "late_payments": 0.04,
        "utilization": 0.03,
        "account_age_months": 0.02,
        "age": 0.01,
        "employment": 0.008,
        "housing": 0.004,
    }

    def baseline_row(self):
        return {
            **{feature: 0.0 for feature in FEATURES[:6]},
            "employment": "baseline",
            "housing": "baseline",
        }

    def predict_probability(self, row):
        total = 0.4
        for feature in FEATURES[:6]:
            value = row.iloc[0][feature]
            total += self._weights[feature] * (0.0 if pd.isna(value) else float(value))
        for feature in FEATURES[6:]:
            value = row.iloc[0][feature]
            total += self._weights[feature] * (
                0.0 if pd.isna(value) or value == "baseline" else 1.0
            )
        return total


class NonFiniteModel(StableModel):
    def predict_probability(self, row):
        if pd.isna(row.iloc[0]["income"]):
            return math.nan
        return super().predict_probability(row)


def _row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **{feature: 1.0 for feature in FEATURES[:6]},
                "employment": "observed",
                "housing": "observed",
            }
        ]
    )


def test_stable_explanations_pass_and_are_order_invariant():
    policy = MissingnessPolicy(max_probability_drift=0.1)
    first = audit_explanation_missingness(
        StableModel(),
        _row(),
        mask_features=("income", "housing", "utilization"),
        policy=policy,
    )
    second = audit_explanation_missingness(
        StableModel(),
        _row(),
        mask_features=("utilization", "income", "housing"),
        policy=policy,
    )

    assert first.passed
    assert first.findings == ()
    assert first.scenario_count == 3
    assert first.worst_top_k_overlap == 1.0
    assert first.worst_direction_agreement == 1.0
    assert first.evidence_sha256 == second.evidence_sha256
    assert len(first.evidence_sha256) == 64
    assert "row_sha256" not in first.to_dict()


def test_policy_failures_name_the_affected_feature():
    result = audit_explanation_missingness(
        StableModel(),
        _row(),
        mask_features=("income",),
        policy=MissingnessPolicy(max_probability_drift=0.01),
    )

    assert not result.passed
    assert result.findings == ("PROBABILITY_DRIFT_EXCEEDED:income",)
    assert result.scenarios[0].findings == ("PROBABILITY_DRIFT_EXCEEDED",)


def test_non_finite_model_output_fails_closed():
    with pytest.raises(ValueError, match="probabilities must be finite"):
        audit_explanation_missingness(
            NonFiniteModel(), _row(), mask_features=("income",)
        )


def test_prediction_budget_is_enforced_before_inference():
    with pytest.raises(ValueError, match="prediction budget"):
        audit_explanation_missingness(
            StableModel(),
            _row(),
            mask_features=("income",),
            policy=MissingnessPolicy(max_predictions=19),
        )


@pytest.mark.parametrize(
    ("row", "mask_features", "message"),
    [
        (_row().iloc[0:0], ("income",), "exactly one"),
        (_row().assign(income=np.nan), ("income",), "must not contain missing"),
        (_row(), ("income", "income"), "must be unique"),
        (_row(), ("unknown",), "feature schema"),
    ],
)
def test_invalid_inputs_are_rejected(row, mask_features, message):
    with pytest.raises(ValueError, match=message):
        audit_explanation_missingness(StableModel(), row, mask_features=mask_features)


def test_real_pipeline_emits_bounded_deterministic_evidence():
    frame = synthetic_credit_data(rows=2200, seed=21)
    model, _ = CreditRiskModel.fit(frame)
    row = frame.drop(columns="default").iloc[[100]][FEATURES]
    first = audit_explanation_missingness(model, row)
    second = audit_explanation_missingness(model, row)

    assert first.scenario_count == len(FEATURES)
    assert first.evidence_sha256 == second.evidence_sha256
    assert 0 <= first.worst_probability_drift <= 1
    assert 0 <= first.worst_top_k_overlap <= 1
    assert 0 <= first.worst_direction_agreement <= 1
    assert [item.missing_feature for item in first.scenarios] == FEATURES
