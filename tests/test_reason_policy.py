from __future__ import annotations

import math

import pytest

from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.explain import FeatureContribution, explain_decision
from creditxai.model import CreditRiskModel
from creditxai.reason_policy import (
    ELIGIBLE_REASON_FEATURES,
    EXCLUDED_REASON_FEATURES,
    governed_reason_codes,
)


def contributions(deltas: dict[str, float]) -> list[FeatureContribution]:
    return [
        FeatureContribution(
            feature=feature,
            observed=1,
            baseline=0,
            probability_delta=deltas.get(feature, 0.0),
            direction="increases_risk"
            if deltas.get(feature, 0.0) > 0
            else "decreases_risk",
        )
        for feature in FEATURES
    ]


def test_policy_selects_only_eligible_material_reasons() -> None:
    result = governed_reason_codes(
        contributions(
            {
                "age": 0.08,
                "employment": 0.04,
                "debt_ratio": 0.07,
                "late_payments": 0.03,
                "income": 0.0005,
            }
        ),
        top_k=2,
    )

    assert result.source_features == ("debt_ratio", "late_payments")
    assert result.reason_codes == ("debt-to-income burden", "late-payment history")
    assert result.excluded_material_features == ("age", "employment")
    assert result.requires_policy_review is True
    assert result.findings == ("excluded_feature_materially_increased_score",)


def test_policy_retains_all_features_in_attribution_evidence() -> None:
    rows = contributions({"age": 0.02, "utilization": 0.03})
    result = governed_reason_codes(rows)

    assert set(result.source_features) <= ELIGIBLE_REASON_FEATURES
    assert set(result.excluded_material_features) <= EXCLUDED_REASON_FEATURES
    assert {row.feature for row in rows} == set(FEATURES)


def test_manual_review_without_eligible_reason_requires_review() -> None:
    result = governed_reason_codes(
        contributions({"age": 0.0001}),
        require_reason=True,
    )

    assert result.reason_codes == ()
    assert result.findings == ("no_material_policy_eligible_reason",)
    assert result.requires_policy_review is True


def test_low_risk_explanation_does_not_require_a_reason() -> None:
    result = governed_reason_codes(contributions({}), require_reason=False)

    assert result.reason_codes == ()
    assert result.findings == ()
    assert result.requires_policy_review is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.pop(), "complete model feature schema"),
        (lambda rows: rows.append(rows[0]), "duplicates a feature"),
        (lambda rows: setattr(rows[0], "feature", "unknown"), "unknown feature"),
    ],
)
def test_policy_fails_closed_on_attribution_schema_drift(mutate, message) -> None:
    rows = contributions({})
    if message == "unknown feature":
        rows[0] = FeatureContribution("unknown", 1, 0, 0.0, "decreases_risk")
    else:
        mutate(rows)

    with pytest.raises(ValueError, match=message):
        governed_reason_codes(rows)


@pytest.mark.parametrize("delta", [math.nan, math.inf, -math.inf])
def test_policy_rejects_non_finite_attributions(delta: float) -> None:
    rows = contributions({})
    rows[0] = FeatureContribution(FEATURES[0], 1, 0, delta, "decreases_risk")

    with pytest.raises(ValueError, match="must be finite"):
        governed_reason_codes(rows)


def test_policy_rejects_contradictory_direction() -> None:
    rows = contributions({"income": 0.02})
    rows[0] = FeatureContribution("income", 1, 0, 0.02, "decreases_risk")

    with pytest.raises(ValueError, match="contradicts"):
        governed_reason_codes(rows)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"top_k": 0}, ValueError),
        ({"top_k": True}, ValueError),
        ({"materiality": math.nan}, ValueError),
        ({"materiality": "0.1"}, TypeError),
        ({"require_reason": 1}, TypeError),
    ],
)
def test_policy_rejects_invalid_configuration(kwargs, error) -> None:
    with pytest.raises(error):
        governed_reason_codes(contributions({}), **kwargs)


def test_explanation_integrates_policy_without_hiding_attributions() -> None:
    frame = synthetic_credit_data(rows=2200, seed=91)
    model, _ = CreditRiskModel.fit(frame)
    row = frame.drop(columns="default").iloc[[100]][FEATURES]

    explanation = explain_decision(model, row)

    assert set(explanation["reason_code_features"]) <= ELIGIBLE_REASON_FEATURES
    assert len(explanation["reason_codes"]) == len(explanation["reason_code_features"])
    assert explanation["reason_code_policy"]["policy_version"]
    assert set(explanation["reason_code_policy"]["excluded_material_features"]) <= (
        EXCLUDED_REASON_FEATURES
    )
    assert {row["feature"] for row in explanation["contributions"]} == set(FEATURES)
