from __future__ import annotations

import math

import pytest

from creditxai.data import FEATURES
from creditxai.explain import FeatureContribution
from creditxai.tolerance_audit import audit_direction_tolerances


def contributions(*, income_delta: float = 5e-7) -> list[FeatureContribution]:
    deltas = {
        "income": income_delta,
        "debt_ratio": 0.02,
        "late_payments": 0.01,
        "utilization": -0.02,
        "account_age_months": -0.01,
        "age": 0.0,
        "employment": 0.0,
        "housing": 0.0,
    }
    return [
        FeatureContribution(feature, 1, 0, deltas[feature], "ignored")
        for feature in FEATURES
    ]


def test_audit_identifies_tolerance_sensitive_reason_ranking():
    report = audit_direction_tolerances(
        contributions(),
        tolerances=(0.0, 1e-8, 1e-6, 1e-4),
    )

    assert report.risk_ranking_stable is False
    assert report.reason_codes_stable is False
    assert report.unstable_features == ("income",)
    assert report.snapshots[0].risk_feature_ranking == (
        "debt_ratio",
        "late_payments",
        "income",
    )
    assert report.snapshots[2].risk_feature_ranking == (
        "debt_ratio",
        "late_payments",
    )
    assert len(report.attribution_sha256) == 64
    assert len(report.audit_sha256) == 64


def test_audit_is_deterministic_and_input_order_independent():
    first = audit_direction_tolerances(contributions(income_delta=0.002))
    second = audit_direction_tolerances(
        list(reversed(contributions(income_delta=0.002)))
    )

    assert first == second
    assert first.risk_ranking_stable
    assert first.reason_codes_stable
    assert first.unstable_features == ()


@pytest.mark.parametrize(
    "tolerances",
    [
        (0.0,),
        (0.0, 0.0),
        (1e-4, 1e-6),
        (0.0, math.nan),
        (0.0, 1.1),
    ],
)
def test_audit_rejects_invalid_tolerance_policies(tolerances):
    with pytest.raises(ValueError, match="tolerances"):
        audit_direction_tolerances(contributions(), tolerances=tolerances)


def test_audit_rejects_non_numeric_tolerance_and_invalid_top_k():
    with pytest.raises(TypeError, match="tolerances"):
        audit_direction_tolerances(contributions(), tolerances=(0.0, "0.1"))
    with pytest.raises(ValueError, match="top_k"):
        audit_direction_tolerances(contributions(), top_k=0)


def test_audit_rejects_missing_duplicate_or_invalid_attribution_evidence():
    values = contributions()
    with pytest.raises(ValueError, match="every model feature"):
        audit_direction_tolerances(values[:-1])
    with pytest.raises(ValueError, match="every model feature"):
        audit_direction_tolerances([*values[:-1], values[0]])
    invalid = [*values[:-1], FeatureContribution("housing", 1, 0, math.inf, "ignored")]
    with pytest.raises(ValueError, match="probability deltas"):
        audit_direction_tolerances(invalid)
    malformed = [*values[:-1], FeatureContribution("housing", 1, 0, "bad", "ignored")]
    with pytest.raises(ValueError, match="probability deltas"):
        audit_direction_tolerances(malformed)
