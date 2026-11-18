from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditxai.data import FEATURES, TARGET, synthetic_credit_data
from creditxai.global_importance import (
    DEFAULT_FEATURE_GROUPS,
    grouped_permutation_importance,
)
from creditxai.model import CreditRiskModel


@pytest.fixture(scope="module")
def fitted_model_and_test() -> tuple[CreditRiskModel, pd.DataFrame]:
    frame = synthetic_credit_data(rows=2200, seed=18)
    model, _ = CreditRiskModel.fit(frame)
    return model, frame.iloc[int(len(frame) * 0.8) :].copy()


def test_grouped_importance_is_reproducible_and_retains_repeat_evidence(
    fitted_model_and_test: tuple[CreditRiskModel, pd.DataFrame],
) -> None:
    model, test = fitted_model_and_test
    first = grouped_permutation_importance(model, test, repeats=4, seed=91)
    second = grouped_permutation_importance(model, test, repeats=4, seed=91)

    assert first == second
    assert first.rows == len(test)
    assert first.scoring == "roc_auc"
    assert len(first.group_spec_sha256) == 64
    assert {group.group for group in first.groups} == set(DEFAULT_FEATURE_GROUPS)
    assert set().union(*(set(group.features) for group in first.groups)) == set(
        FEATURES
    )
    assert all(len(group.repeat_drops) == 4 for group in first.groups)
    assert all(np.isfinite(group.importance_mean) for group in first.groups)


def test_group_fingerprint_is_independent_of_mapping_order(
    fitted_model_and_test: tuple[CreditRiskModel, pd.DataFrame],
) -> None:
    model, test = fitted_model_and_test
    reversed_groups = dict(reversed(DEFAULT_FEATURE_GROUPS.items()))

    first = grouped_permutation_importance(
        model, test, feature_groups=DEFAULT_FEATURE_GROUPS, repeats=2, seed=3
    )
    second = grouped_permutation_importance(
        model, test, feature_groups=reversed_groups, repeats=2, seed=3
    )

    assert first == second


def test_joint_permutation_preserves_within_group_rows() -> None:
    class RecordingPipeline:
        def __init__(self) -> None:
            self.frames: list[pd.DataFrame] = []

        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            self.frames.append(frame.copy())
            score = (frame["income"].to_numpy() - 1_000) / 19
            return np.column_stack((1 - score, score))

    pipeline = RecordingPipeline()
    rows = np.arange(20)
    income = 1_000 + rows
    frame = pd.DataFrame(
        {
            "income": income,
            "debt_ratio": income / 10_000,
            "late_payments": rows % 3,
            "utilization": income / 2_000,
            "account_age_months": 12 + rows,
            "age": 25 + rows,
            "employment": ["salaried"] * 20,
            "housing": ["rent"] * 20,
            TARGET: (rows >= 10).astype(int),
        }
    )

    report = grouped_permutation_importance(
        type("Model", (), {"pipeline": pipeline})(),  # structural test double
        frame,
        repeats=2,
        seed=4,
    )

    affordability_calls = [
        observed
        for observed in pipeline.frames[1:]
        if not observed["income"].reset_index(drop=True).equals(frame["income"])
    ]
    assert len(affordability_calls) == 2
    for observed in affordability_calls:
        assert observed["debt_ratio"].to_numpy() == pytest.approx(
            observed["income"].to_numpy() / 10_000
        )
        assert observed["utilization"].to_numpy() == pytest.approx(
            observed["income"].to_numpy() / 2_000
        )
    affordability = next(
        group for group in report.groups if group.group == "affordability"
    )
    assert affordability.importance_mean > 0


@pytest.mark.parametrize(
    "groups, message",
    [
        ({"partial": ("income",)}, "must cover every model feature"),
        (
            {
                "one": tuple(FEATURES),
                "overlap": ("income",),
            },
            "more than one group",
        ),
        (
            {
                "known": tuple(FEATURES),
                "unknown": ("not_a_feature",),
            },
            "unknown features",
        ),
    ],
)
def test_invalid_group_contract_fails_closed(
    fitted_model_and_test: tuple[CreditRiskModel, pd.DataFrame],
    groups: dict[str, tuple[str, ...]],
    message: str,
) -> None:
    model, test = fitted_model_and_test
    with pytest.raises(ValueError, match=message):
        grouped_permutation_importance(model, test, feature_groups=groups, repeats=2)


def test_invalid_evaluation_contract_fails_closed(
    fitted_model_and_test: tuple[CreditRiskModel, pd.DataFrame],
) -> None:
    model, test = fitted_model_and_test
    with pytest.raises(ValueError, match="at least two"):
        grouped_permutation_importance(model, test, repeats=1)
    with pytest.raises(ValueError, match="missing columns"):
        grouped_permutation_importance(model, test.drop(columns=["income"]), repeats=2)
    one_class = test.copy()
    one_class[TARGET] = 0
    with pytest.raises(ValueError, match="two non-missing"):
        grouped_permutation_importance(model, one_class, repeats=2)
