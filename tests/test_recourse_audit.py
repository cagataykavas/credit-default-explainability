from __future__ import annotations

import math

import pandas as pd
import pytest

from creditxai.recourse_audit import (
    RecourseAuditError,
    RecourseAuditPolicy,
    audit_counterfactual,
)


class LinearRiskModel:
    def predict_probability(self, row: pd.DataFrame) -> float:
        item = row.iloc[0]
        return float(
            0.10
            + 0.50 * item["utilization"]
            + 0.30 * item["debt_ratio"]
            + 0.05 * item["late_payments"]
            - item["income"] / 500_000
        )


@pytest.fixture
def row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "income": 50_000.0,
                "debt_ratio": 0.60,
                "late_payments": 2,
                "utilization": 0.80,
                "account_age_months": 36,
                "age": 35,
                "employment": "salaried",
                "housing": "rent",
            }
        ]
    )


def test_accepts_effective_directional_and_inclusion_minimal_recourse(
    row: pd.DataFrame,
):
    report = audit_counterfactual(
        LinearRiskModel(),
        row,
        {"utilization": 0.20, "debt_ratio": 0.20},
        policy=RecourseAuditPolicy(target_probability=0.35),
    )

    assert report.accepted
    assert report.reason_codes == ()
    assert report.changed_features == ("debt_ratio", "utilization")
    assert report.probability_before == pytest.approx(0.68)
    assert report.probability_after == pytest.approx(0.26)
    assert report.normalized_cost == pytest.approx(10.0)
    assert report.as_dict()["deltas"][0]["feature"] == "debt_ratio"


def test_rejects_candidate_that_does_not_reach_target(row: pd.DataFrame):
    report = audit_counterfactual(LinearRiskModel(), row, {"utilization": 0.60})

    assert not report.accepted
    assert "TARGET_NOT_REACHED" in report.reason_codes


def test_rejects_immutable_feature_change(row: pd.DataFrame):
    report = audit_counterfactual(LinearRiskModel(), row, {"age": 34})

    assert report.immutable_changes == ("age",)
    assert "IMMUTABLE_FEATURE_CHANGED" in report.reason_codes


def test_rejects_action_in_wrong_direction(row: pd.DataFrame):
    report = audit_counterfactual(LinearRiskModel(), row, {"utilization": 0.90})

    assert report.direction_violations == ("utilization",)
    assert "NON_ACTIONABLE_DIRECTION" in report.reason_codes


def test_detects_redundant_action_with_leave_one_out_rescoring(row: pd.DataFrame):
    report = audit_counterfactual(
        LinearRiskModel(),
        row,
        {"income": 55_000.0, "utilization": 0.20, "debt_ratio": 0.20},
        policy=RecourseAuditPolicy(target_probability=0.35),
    )

    assert report.unnecessary_features == ("income",)
    assert "NON_MINIMAL_ACTION_SET" in report.reason_codes


def test_no_op_is_visible_and_rejected(row: pd.DataFrame):
    report = audit_counterfactual(LinearRiskModel(), row, {"income": 50_000.0})

    assert report.changed_features == ()
    assert report.no_op_changes == ("income",)
    assert report.reason_codes[:2] == ("NO_EFFECTIVE_CHANGE", "NO_OP_CHANGE")


def test_rejects_action_cost_over_budget(row: pd.DataFrame):
    report = audit_counterfactual(
        LinearRiskModel(),
        row,
        {"utilization": 0.20, "debt_ratio": 0.20},
        policy=RecourseAuditPolicy(target_probability=0.35, max_normalized_cost=9.0),
    )

    assert report.reason_codes == ("ACTION_COST_EXCEEDED",)


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"unknown": 1}, "UNKNOWN_CHANGE_FEATURE"),
        ({"utilization": math.nan}, "INVALID_FEATURE_VALUE"),
        ({"late_payments": 1.5}, "INVALID_FEATURE_VALUE"),
    ],
)
def test_malformed_changes_fail_closed(
    row: pd.DataFrame, changes: dict[str, object], code: str
):
    with pytest.raises(RecourseAuditError) as error:
        audit_counterfactual(LinearRiskModel(), row, changes)

    assert error.value.code == code


def test_invalid_row_shape_fails_closed(row: pd.DataFrame):
    with pytest.raises(RecourseAuditError) as error:
        audit_counterfactual(LinearRiskModel(), pd.concat([row, row]), {})

    assert error.value.code == "INVALID_ROW_SHAPE"


def test_non_finite_model_output_fails_closed(row: pd.DataFrame):
    class BrokenModel:
        def predict_probability(self, row: pd.DataFrame) -> float:
            return math.nan

    with pytest.raises(RecourseAuditError) as error:
        audit_counterfactual(BrokenModel(), row, {"utilization": 0.20})

    assert error.value.code == "INVALID_MODEL_PROBABILITY"


def test_invalid_policy_fails_closed(row: pd.DataFrame):
    with pytest.raises(RecourseAuditError) as error:
        audit_counterfactual(
            LinearRiskModel(),
            row,
            {},
            policy=RecourseAuditPolicy(target_probability=1.0),
        )

    assert error.value.code == "INVALID_POLICY"
