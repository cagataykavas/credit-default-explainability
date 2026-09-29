from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Literal, Protocol

import pandas as pd

from .data import FEATURES

ACTION_DIRECTIONS: dict[str, Literal["increase", "decrease"]] = {
    "income": "increase",
    "debt_ratio": "decrease",
    "late_payments": "decrease",
    "utilization": "decrease",
}
RATIO_FEATURES = frozenset({"debt_ratio", "utilization"})
INTEGER_FEATURES = frozenset({"late_payments"})
MAX_ACTIONS = 4
MAX_VALUES_PER_ACTION = 32
MAX_MODEL_CALLS = 2_000


class RecourseSearchError(ValueError):
    """Fail-closed input or model error with a stable reason code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ProbabilityModel(Protocol):
    def predict_probability(self, row: pd.DataFrame) -> float: ...


@dataclass(frozen=True)
class ActionGrid:
    feature: str
    values: tuple[float | int, ...]
    cost_scale: float


@dataclass(frozen=True)
class RecourseSearchPolicy:
    target_probability: float = 0.25
    max_model_calls: int = 300
    max_changed_features: int = 3
    max_results: int = 10
    min_probability_improvement: float = 1e-6


@dataclass(frozen=True)
class FeatureChange:
    feature: str
    from_value: float | int
    to_value: float | int
    normalized_cost: float


@dataclass(frozen=True)
class RecourseOption:
    option_id: str
    probability: float
    normalized_cost: float
    changes: tuple[FeatureChange, ...]

    @property
    def changed_features(self) -> tuple[str, ...]:
        return tuple(change.feature for change in self.changes)


@dataclass(frozen=True)
class RecourseFrontier:
    probability_before: float
    target_probability: float
    options: tuple[RecourseOption, ...]
    model_calls: int
    search_space_size: int
    search_truncated: bool
    result_truncated: bool
    reason_code: Literal[
        "FRONTIER_FOUND",
        "TARGET_ALREADY_MET",
        "NO_FEASIBLE_OPTION",
        "SEARCH_INCOMPLETE_NO_FEASIBLE_OPTION",
    ]

    @property
    def feasible(self) -> bool:
        return self.reason_code in {"FRONTIER_FOUND", "TARGET_ALREADY_MET"}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Level:
    value: float | int
    cost: float


def _finite_number(value: Any, code: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise RecourseSearchError(code)
    result = float(value)
    if not math.isfinite(result):
        raise RecourseSearchError(code)
    return result


def _validate_policy(policy: RecourseSearchPolicy) -> None:
    target = _finite_number(policy.target_probability, "INVALID_TARGET_PROBABILITY")
    if not 0.0 < target < 1.0:
        raise RecourseSearchError("INVALID_TARGET_PROBABILITY")
    improvement = _finite_number(
        policy.min_probability_improvement,
        "INVALID_MINIMUM_IMPROVEMENT",
    )
    if improvement < 0.0:
        raise RecourseSearchError("INVALID_MINIMUM_IMPROVEMENT")
    integer_limits = (
        (policy.max_model_calls, 2, MAX_MODEL_CALLS, "INVALID_MODEL_CALL_BUDGET"),
        (policy.max_changed_features, 1, MAX_ACTIONS, "INVALID_CHANGE_BUDGET"),
        (policy.max_results, 1, 100, "INVALID_RESULT_BUDGET"),
    )
    for value, minimum, maximum, code in integer_limits:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not minimum <= value <= maximum
        ):
            raise RecourseSearchError(code)


def _validate_row(row: pd.DataFrame) -> dict[str, float | int]:
    if not isinstance(row, pd.DataFrame) or len(row) != 1:
        raise RecourseSearchError("INVALID_ROW_SHAPE")
    missing = set(FEATURES) - set(row.columns)
    if missing:
        raise RecourseSearchError("MISSING_MODEL_FEATURE")

    values: dict[str, float | int] = {}
    record = row.iloc[0]
    for feature in ACTION_DIRECTIONS:
        value = _finite_number(record[feature], "INVALID_ACTIONABLE_VALUE")
        if feature in RATIO_FEATURES and not 0.0 <= value <= 1.0:
            raise RecourseSearchError("INVALID_ACTIONABLE_VALUE")
        if feature == "income" and value < 0.0:
            raise RecourseSearchError("INVALID_ACTIONABLE_VALUE")
        if feature in INTEGER_FEATURES:
            if not value.is_integer() or value < 0.0:
                raise RecourseSearchError("INVALID_ACTIONABLE_VALUE")
            values[feature] = int(value)
        else:
            values[feature] = value
    return values


def _predict(model: ProbabilityModel, row: pd.DataFrame) -> float:
    try:
        probability = model.predict_probability(row)
    except Exception as error:
        raise RecourseSearchError("MODEL_PREDICTION_FAILED") from error
    value = _finite_number(probability, "INVALID_MODEL_PROBABILITY")
    if not 0.0 <= value <= 1.0:
        raise RecourseSearchError("INVALID_MODEL_PROBABILITY")
    return value


def default_action_grids(row: pd.DataFrame) -> tuple[ActionGrid, ...]:
    """Build a small deterministic action grid from the observed row."""

    original = _validate_row(row)
    candidates: dict[str, Sequence[float | int]] = {
        "utilization": tuple(
            max(0.0, original["utilization"] - delta) for delta in (0.10, 0.20, 0.30)
        ),
        "debt_ratio": tuple(
            max(0.0, original["debt_ratio"] - delta) for delta in (0.10, 0.20, 0.30)
        ),
        "late_payments": tuple(
            max(0, int(original["late_payments"]) - delta) for delta in (1, 2, 3)
        ),
        "income": tuple(
            original["income"] + max(float(original["income"]) * ratio, 1_000.0)
            for ratio in (0.10, 0.25, 0.50)
        ),
    }
    scales = {
        "utilization": 0.10,
        "debt_ratio": 0.10,
        "late_payments": 1.0,
        "income": max(float(original["income"]) * 0.10, 1_000.0),
    }
    grids: list[ActionGrid] = []
    for feature in ("utilization", "debt_ratio", "late_payments", "income"):
        unique = tuple(
            dict.fromkeys(
                value for value in candidates[feature] if value != original[feature]
            )
        )
        if unique:
            grids.append(ActionGrid(feature, unique, scales[feature]))
    return tuple(grids)


def _validate_grids(
    grids: Sequence[ActionGrid],
    original: dict[str, float | int],
) -> tuple[tuple[str, tuple[_Level, ...]], ...]:
    if not isinstance(grids, Sequence) or not grids or len(grids) > MAX_ACTIONS:
        raise RecourseSearchError("INVALID_ACTION_GRID_COUNT")
    seen: set[str] = set()
    validated: list[tuple[str, tuple[_Level, ...]]] = []
    for grid in grids:
        if not isinstance(grid, ActionGrid):
            raise RecourseSearchError("INVALID_ACTION_GRID")
        if grid.feature not in ACTION_DIRECTIONS:
            raise RecourseSearchError("IMMUTABLE_OR_UNKNOWN_FEATURE")
        if grid.feature in seen:
            raise RecourseSearchError("DUPLICATE_ACTION_FEATURE")
        seen.add(grid.feature)
        if (
            not isinstance(grid.values, tuple)
            or not grid.values
            or len(grid.values) > MAX_VALUES_PER_ACTION
        ):
            raise RecourseSearchError("INVALID_ACTION_VALUES")
        cost_scale = _finite_number(grid.cost_scale, "INVALID_COST_SCALE")
        if cost_scale <= 0.0:
            raise RecourseSearchError("INVALID_COST_SCALE")

        start = original[grid.feature]
        direction = ACTION_DIRECTIONS[grid.feature]
        levels: list[_Level] = [_Level(start, 0.0)]
        seen_values: set[float] = set()
        for raw_value in grid.values:
            value = _finite_number(raw_value, "INVALID_ACTION_VALUE")
            if value == float(start) or value in seen_values:
                raise RecourseSearchError("DUPLICATE_OR_NOOP_ACTION")
            seen_values.add(value)
            if grid.feature in RATIO_FEATURES and not 0.0 <= value <= 1.0:
                raise RecourseSearchError("ACTION_OUT_OF_DOMAIN")
            if grid.feature == "income" and value < 0.0:
                raise RecourseSearchError("ACTION_OUT_OF_DOMAIN")
            if grid.feature in INTEGER_FEATURES:
                if not value.is_integer() or value < 0.0:
                    raise RecourseSearchError("ACTION_OUT_OF_DOMAIN")
                normalized: float | int = int(value)
            else:
                normalized = value
            if direction == "increase" and value <= float(start):
                raise RecourseSearchError("ACTION_DIRECTION_VIOLATION")
            if direction == "decrease" and value >= float(start):
                raise RecourseSearchError("ACTION_DIRECTION_VIOLATION")
            levels.append(_Level(normalized, abs(value - float(start)) / cost_scale))
        levels[1:] = sorted(levels[1:], key=lambda level: (level.cost, level.value))
        validated.append((grid.feature, tuple(levels)))
    return tuple(validated)


def _search_space_size(
    levels: Sequence[tuple[str, tuple[_Level, ...]]], maximum: int
) -> int:
    counts = [0] * (len(levels) + 1)
    counts[0] = 1
    for _, feature_levels in levels:
        alternatives = len(feature_levels) - 1
        for changed in range(len(levels), 0, -1):
            counts[changed] += counts[changed - 1] * alternatives
    return sum(counts[: maximum + 1])


def _state_cost(
    state: tuple[int, ...], levels: Sequence[tuple[str, tuple[_Level, ...]]]
) -> float:
    return sum(levels[index][1][level].cost for index, level in enumerate(state))


def _changes_for_state(
    state: tuple[int, ...],
    levels: Sequence[tuple[str, tuple[_Level, ...]]],
) -> tuple[FeatureChange, ...]:
    changes = (
        FeatureChange(
            feature=feature,
            from_value=feature_levels[0].value,
            to_value=feature_levels[level].value,
            normalized_cost=feature_levels[level].cost,
        )
        for level, (feature, feature_levels) in zip(state, levels)
        if level > 0
    )
    return tuple(sorted(changes, key=lambda change: change.feature))


def _option_id(changes: tuple[FeatureChange, ...]) -> str:
    payload = [asdict(change) for change in changes]
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _dominates(left: RecourseOption, right: RecourseOption) -> bool:
    left_objectives = (left.normalized_cost, len(left.changes), left.probability)
    right_objectives = (right.normalized_cost, len(right.changes), right.probability)
    no_worse = all(a <= b + 1e-12 for a, b in zip(left_objectives, right_objectives))
    strictly_better = any(
        a < b - 1e-12 for a, b in zip(left_objectives, right_objectives)
    )
    return no_worse and strictly_better


def _pareto_frontier(options: Sequence[RecourseOption]) -> list[RecourseOption]:
    return [
        candidate
        for candidate in options
        if not any(
            other.option_id != candidate.option_id and _dominates(other, candidate)
            for other in options
        )
    ]


def generate_recourse_frontier(
    model: ProbabilityModel,
    row: pd.DataFrame,
    *,
    action_grids: Sequence[ActionGrid] | None = None,
    policy: RecourseSearchPolicy | None = None,
) -> RecourseFrontier:
    """Search a bounded action grid and return nondominated feasible recourse candidates."""

    effective_policy = policy or RecourseSearchPolicy()
    _validate_policy(effective_policy)
    original = _validate_row(row)
    supplied_grids = (
        action_grids if action_grids is not None else default_action_grids(row)
    )
    levels = _validate_grids(supplied_grids, original)
    space_size = _search_space_size(levels, effective_policy.max_changed_features)

    probability_before = _predict(model, row)
    if probability_before <= effective_policy.target_probability:
        return RecourseFrontier(
            probability_before=probability_before,
            target_probability=effective_policy.target_probability,
            options=(),
            model_calls=1,
            search_space_size=space_size,
            search_truncated=False,
            result_truncated=False,
            reason_code="TARGET_ALREADY_MET",
        )

    start = (0,) * len(levels)
    queue: list[tuple[float, int, tuple[int, ...]]] = [(0.0, 0, start)]
    enqueued = {start}
    model_calls = 1
    feasible: list[RecourseOption] = []

    while queue and model_calls < effective_policy.max_model_calls:
        _, _, state = heapq.heappop(queue)
        if state != start:
            changes = _changes_for_state(state, levels)
            candidate = row.copy()
            index = candidate.index[0]
            for change in changes:
                candidate.at[index, change.feature] = change.to_value
            probability = _predict(model, candidate)
            model_calls += 1
            if (
                probability <= effective_policy.target_probability
                and probability_before - probability
                >= effective_policy.min_probability_improvement
            ):
                feasible.append(
                    RecourseOption(
                        option_id=_option_id(changes),
                        probability=probability,
                        normalized_cost=sum(
                            change.normalized_cost for change in changes
                        ),
                        changes=changes,
                    )
                )

        for dimension, level in enumerate(state):
            if level + 1 >= len(levels[dimension][1]):
                continue
            neighbor = list(state)
            neighbor[dimension] += 1
            next_state = tuple(neighbor)
            changed_count = sum(value > 0 for value in next_state)
            if (
                changed_count > effective_policy.max_changed_features
                or next_state in enqueued
            ):
                continue
            enqueued.add(next_state)
            heapq.heappush(
                queue,
                (_state_cost(next_state, levels), changed_count, next_state),
            )

    search_truncated = bool(queue)
    frontier = _pareto_frontier(feasible)
    frontier.sort(
        key=lambda option: (
            len(option.changes),
            option.normalized_cost,
            option.probability,
            option.option_id,
        )
    )
    result_truncated = len(frontier) > effective_policy.max_results
    selected = tuple(frontier[: effective_policy.max_results])
    if selected:
        reason_code = "FRONTIER_FOUND"
    elif search_truncated:
        reason_code = "SEARCH_INCOMPLETE_NO_FEASIBLE_OPTION"
    else:
        reason_code = "NO_FEASIBLE_OPTION"
    return RecourseFrontier(
        probability_before=probability_before,
        target_probability=effective_policy.target_probability,
        options=selected,
        model_calls=model_calls,
        search_space_size=space_size,
        search_truncated=search_truncated,
        result_truncated=result_truncated,
        reason_code=reason_code,
    )
