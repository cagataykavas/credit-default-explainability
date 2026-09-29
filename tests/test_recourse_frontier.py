from __future__ import annotations

import math

import pandas as pd
import pytest

from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.model import CreditRiskModel
from creditxai.recourse_frontier import (
    ACTION_DIRECTIONS,
    ActionGrid,
    RecourseSearchError,
    RecourseSearchPolicy,
    default_action_grids,
    generate_recourse_frontier,
)


def row(**overrides: object) -> pd.DataFrame:
    values: dict[str, object] = {
        "income": 50_000.0,
        "debt_ratio": 0.60,
        "late_payments": 2,
        "utilization": 0.80,
        "account_age_months": 36,
        "age": 35,
        "employment": "salaried",
        "housing": "rent",
    }
    values.update(overrides)
    return pd.DataFrame([values], columns=FEATURES)


class LinearRiskModel:
    def predict_probability(self, candidate: pd.DataFrame) -> float:
        record = candidate.iloc[0]
        probability = (
            0.30 * float(record["utilization"])
            + 0.25 * float(record["debt_ratio"])
            + 0.05 * float(record["late_payments"])
            - 0.000002 * (float(record["income"]) - 50_000.0)
        )
        return max(0.0, min(1.0, probability))


def grids() -> tuple[ActionGrid, ...]:
    return (
        ActionGrid("utilization", (0.60, 0.40, 0.20), 0.20),
        ActionGrid("debt_ratio", (0.40, 0.20), 0.20),
        ActionGrid("late_payments", (1, 0), 1.0),
        ActionGrid("income", (60_000.0, 75_000.0), 10_000.0),
    )


def test_generates_deterministic_pareto_recourse_options() -> None:
    policy = RecourseSearchPolicy(target_probability=0.30, max_model_calls=200)

    first = generate_recourse_frontier(
        LinearRiskModel(), row(), action_grids=grids(), policy=policy
    )
    second = generate_recourse_frontier(
        LinearRiskModel(), row(), action_grids=grids(), policy=policy
    )

    assert first == second
    assert first.reason_code == "FRONTIER_FOUND"
    assert first.options
    assert all(option.probability <= 0.30 for option in first.options)
    assert all(len(option.option_id) == 64 for option in first.options)
    assert not first.search_truncated


def test_option_identity_is_independent_of_action_grid_order() -> None:
    policy = RecourseSearchPolicy(target_probability=0.35)
    original = generate_recourse_frontier(
        LinearRiskModel(), row(), action_grids=grids(), policy=policy
    )
    reordered = generate_recourse_frontier(
        LinearRiskModel(), row(), action_grids=tuple(reversed(grids())), policy=policy
    )

    assert [option.option_id for option in original.options] == [
        option.option_id for option in reordered.options
    ]


def test_returned_options_are_pairwise_nondominated() -> None:
    result = generate_recourse_frontier(
        LinearRiskModel(),
        row(),
        action_grids=grids(),
        policy=RecourseSearchPolicy(target_probability=0.35),
    )

    objectives = [
        (option.normalized_cost, len(option.changes), option.probability)
        for option in result.options
    ]
    for index, left in enumerate(objectives):
        for other_index, right in enumerate(objectives):
            if index == other_index:
                continue
            dominates = all(a <= b + 1e-12 for a, b in zip(right, left)) and any(
                a < b - 1e-12 for a, b in zip(right, left)
            )
            assert not dominates


def test_only_allowlisted_directional_features_can_change() -> None:
    result = generate_recourse_frontier(
        LinearRiskModel(),
        row(),
        action_grids=grids(),
        policy=RecourseSearchPolicy(target_probability=0.35),
    )

    for option in result.options:
        for change in option.changes:
            assert change.feature in ACTION_DIRECTIONS
            if ACTION_DIRECTIONS[change.feature] == "increase":
                assert change.to_value > change.from_value
            else:
                assert change.to_value < change.from_value
        assert "age" not in option.changed_features
        assert "employment" not in option.changed_features
        assert "housing" not in option.changed_features


def test_changed_feature_budget_is_enforced() -> None:
    result = generate_recourse_frontier(
        LinearRiskModel(),
        row(),
        action_grids=grids(),
        policy=RecourseSearchPolicy(
            target_probability=0.35,
            max_changed_features=1,
        ),
    )

    assert all(len(option.changes) == 1 for option in result.options)
    assert result.search_space_size == 10


def test_model_call_budget_sets_explicit_truncation() -> None:
    result = generate_recourse_frontier(
        LinearRiskModel(),
        row(),
        action_grids=grids(),
        policy=RecourseSearchPolicy(
            target_probability=0.20,
            max_model_calls=5,
        ),
    )

    assert result.model_calls == 5
    assert result.search_truncated
    assert result.search_space_size > result.model_calls
    assert result.reason_code == "SEARCH_INCOMPLETE_NO_FEASIBLE_OPTION"
    assert not result.feasible


def test_result_budget_sets_separate_truncation() -> None:
    result = generate_recourse_frontier(
        LinearRiskModel(),
        row(),
        action_grids=grids(),
        policy=RecourseSearchPolicy(
            target_probability=0.40,
            max_results=1,
        ),
    )

    assert len(result.options) == 1
    assert result.result_truncated
    assert not result.search_truncated


def test_already_compliant_row_short_circuits_after_baseline() -> None:
    result = generate_recourse_frontier(
        LinearRiskModel(),
        row(utilization=0.10, debt_ratio=0.10, late_payments=0),
        policy=RecourseSearchPolicy(target_probability=0.25),
    )

    assert result.reason_code == "TARGET_ALREADY_MET"
    assert result.feasible
    assert result.options == ()
    assert result.model_calls == 1
    assert not result.search_truncated


def test_no_feasible_option_is_explicit() -> None:
    result = generate_recourse_frontier(
        LinearRiskModel(),
        row(),
        action_grids=(ActionGrid("income", (50_001.0,), 10_000.0),),
        policy=RecourseSearchPolicy(target_probability=0.05),
    )

    assert result.reason_code == "NO_FEASIBLE_OPTION"
    assert not result.feasible
    assert result.options == ()
    assert not result.search_truncated


def test_frontier_serializes_without_custom_encoder() -> None:
    result = generate_recourse_frontier(
        LinearRiskModel(),
        row(),
        action_grids=grids(),
        policy=RecourseSearchPolicy(target_probability=0.35),
    )

    payload = result.as_dict()
    assert payload["reason_code"] == "FRONTIER_FOUND"
    assert payload["options"][0]["option_id"] == result.options[0].option_id


@pytest.mark.parametrize(
    ("action_grids", "code"),
    [
        ((), "INVALID_ACTION_GRID_COUNT"),
        ((ActionGrid("age", (34,), 1.0),), "IMMUTABLE_OR_UNKNOWN_FEATURE"),
        (
            (
                ActionGrid("income", (60_000.0,), 1.0),
                ActionGrid("income", (70_000.0,), 1.0),
            ),
            "DUPLICATE_ACTION_FEATURE",
        ),
        ((ActionGrid("income", (), 1.0),), "INVALID_ACTION_VALUES"),
        ((ActionGrid("income", (50_000.0,), 1.0),), "DUPLICATE_OR_NOOP_ACTION"),
        (
            (ActionGrid("income", (60_000.0, 60_000.0), 1.0),),
            "DUPLICATE_OR_NOOP_ACTION",
        ),
        ((ActionGrid("income", (49_000.0,), 1.0),), "ACTION_DIRECTION_VIOLATION"),
        ((ActionGrid("utilization", (0.90,), 1.0),), "ACTION_DIRECTION_VIOLATION"),
        ((ActionGrid("utilization", (-0.10,), 1.0),), "ACTION_OUT_OF_DOMAIN"),
        ((ActionGrid("late_payments", (0.5,), 1.0),), "ACTION_OUT_OF_DOMAIN"),
        ((ActionGrid("income", (math.nan,), 1.0),), "INVALID_ACTION_VALUE"),
        ((ActionGrid("income", (60_000.0,), 0.0),), "INVALID_COST_SCALE"),
    ],
)
def test_malformed_action_grids_fail_closed(
    action_grids: tuple[ActionGrid, ...],
    code: str,
) -> None:
    with pytest.raises(RecourseSearchError) as error:
        generate_recourse_frontier(
            LinearRiskModel(),
            row(),
            action_grids=action_grids,
        )

    assert error.value.code == code


@pytest.mark.parametrize(
    ("policy", "code"),
    [
        (RecourseSearchPolicy(target_probability=1.0), "INVALID_TARGET_PROBABILITY"),
        (RecourseSearchPolicy(max_model_calls=1), "INVALID_MODEL_CALL_BUDGET"),
        (RecourseSearchPolicy(max_changed_features=0), "INVALID_CHANGE_BUDGET"),
        (RecourseSearchPolicy(max_results=0), "INVALID_RESULT_BUDGET"),
        (
            RecourseSearchPolicy(min_probability_improvement=math.nan),
            "INVALID_MINIMUM_IMPROVEMENT",
        ),
    ],
)
def test_invalid_policy_fails_closed(policy: RecourseSearchPolicy, code: str) -> None:
    with pytest.raises(RecourseSearchError) as error:
        generate_recourse_frontier(LinearRiskModel(), row(), policy=policy)

    assert error.value.code == code


@pytest.mark.parametrize(
    ("bad_row", "code"),
    [
        (pd.DataFrame(), "INVALID_ROW_SHAPE"),
        (pd.concat([row(), row()]), "INVALID_ROW_SHAPE"),
        (row().drop(columns="age"), "MISSING_MODEL_FEATURE"),
        (row(utilization=math.nan), "INVALID_ACTIONABLE_VALUE"),
        (row(late_payments=1.5), "INVALID_ACTIONABLE_VALUE"),
    ],
)
def test_invalid_rows_fail_closed(bad_row: pd.DataFrame, code: str) -> None:
    with pytest.raises(RecourseSearchError) as error:
        generate_recourse_frontier(LinearRiskModel(), bad_row)

    assert error.value.code == code


@pytest.mark.parametrize("probability", [math.nan, -0.1, 1.1, "bad"])
def test_invalid_model_probability_fails_closed(probability: object) -> None:
    class BrokenModel:
        def predict_probability(self, candidate: pd.DataFrame) -> object:
            return probability

    with pytest.raises(RecourseSearchError) as error:
        generate_recourse_frontier(BrokenModel(), row())  # type: ignore[arg-type]

    assert error.value.code == "INVALID_MODEL_PROBABILITY"


def test_model_exception_is_sanitized() -> None:
    class BrokenModel:
        def predict_probability(self, candidate: pd.DataFrame) -> float:
            raise RuntimeError("secret backend host and customer identifier")

    with pytest.raises(RecourseSearchError) as error:
        generate_recourse_frontier(BrokenModel(), row())

    assert error.value.code == "MODEL_PREDICTION_FAILED"
    assert "secret" not in str(error.value)


def test_default_grids_are_bounded_directional_and_unique() -> None:
    generated = default_action_grids(
        row(utilization=0.15, debt_ratio=0.05, late_payments=1)
    )

    assert 1 <= len(generated) <= 4
    for grid in generated:
        assert len(grid.values) <= 3
        assert len(grid.values) == len(set(grid.values))
        if grid.feature in {"utilization", "debt_ratio"}:
            assert all(0.0 <= value <= 1.0 for value in grid.values)


def test_default_grid_remains_available_at_numeric_floors() -> None:
    generated = default_action_grids(
        row(income=0.0, utilization=0.0, debt_ratio=0.0, late_payments=0)
    )

    assert [grid.feature for grid in generated] == ["income"]
    assert generated[0].values == (1_000.0,)


def test_real_credit_model_integration_returns_valid_frontier() -> None:
    frame = synthetic_credit_data(rows=2200, seed=91)
    model, _ = CreditRiskModel.fit(frame)
    candidate = (
        frame.drop(columns="default")
        .sort_values(["utilization", "debt_ratio", "late_payments"], ascending=False)
        .iloc[[0]][FEATURES]
    )
    before = model.predict_probability(candidate)
    target = max(0.01, before - 0.05)

    result = generate_recourse_frontier(
        model,
        candidate,
        policy=RecourseSearchPolicy(target_probability=target, max_results=5),
    )

    assert result.probability_before == pytest.approx(before)
    assert result.model_calls <= 300
    assert all(option.probability <= target for option in result.options)
    assert all(option.probability < before for option in result.options)
