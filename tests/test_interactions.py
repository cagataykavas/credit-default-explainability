from __future__ import annotations

import pandas as pd
import pytest

from creditxai.cli import main
from creditxai.interactions import (
    InteractionAuditError,
    InteractionPolicy,
    audit_pairwise_interactions,
)


class FormulaModel:
    def __init__(self, *, interaction: float = 0.0, invalid: bool = False) -> None:
        self.interaction = interaction
        self.invalid = invalid
        self.calls = 0

    def baseline_row(self) -> dict[str, float]:
        return {"x": 0.0, "y": 0.0}

    def predict_probability(self, row: pd.DataFrame) -> float:
        self.calls += 1
        if self.invalid:
            return float("nan")
        x = float(row.iloc[0]["x"])
        y = float(row.iloc[0]["y"])
        return 0.1 + 0.1 * x + 0.1 * y + self.interaction * x * y


def test_additive_model_has_zero_pair_residual() -> None:
    model = FormulaModel()

    audit = audit_pairwise_interactions(
        model,
        pd.DataFrame([{"x": 1.0, "y": 1.0}]),
        model_revision="additive-v1",
        features=("x", "y"),
        policy=InteractionPolicy(max_abs_interaction=0.01),
    )

    assert audit.accepted is True
    assert audit.maximum_abs_interaction == 0.0
    assert audit.interactions[0].direction == "neutral"
    assert audit.prediction_calls == 4
    assert model.calls == 4


def test_non_additive_pair_is_reported_and_gated() -> None:
    model = FormulaModel(interaction=0.08)

    audit = audit_pairwise_interactions(
        model,
        pd.DataFrame([{"x": 1.0, "y": 1.0}]),
        model_revision="interaction-v1",
        features=("x", "y"),
        policy=InteractionPolicy(max_abs_interaction=0.05),
    )

    assert audit.accepted is False
    assert audit.interactions[0].probability_residual == pytest.approx(0.08)
    assert audit.interactions[0].direction == "reinforcing"
    assert audit.findings == ("PAIR_INTERACTION_LIMIT_EXCEEDED:x:y",)


def test_audit_is_deterministic_and_binds_input() -> None:
    row = pd.DataFrame([{"x": 1.0, "y": 1.0}])
    first = audit_pairwise_interactions(
        FormulaModel(interaction=0.02),
        row,
        model_revision="model-v1",
        features=("x", "y"),
    )
    second = audit_pairwise_interactions(
        FormulaModel(interaction=0.02),
        row.copy(),
        model_revision="model-v1",
        features=("x", "y"),
    )
    changed = audit_pairwise_interactions(
        FormulaModel(interaction=0.02),
        pd.DataFrame([{"x": 1.5, "y": 1.0}]),
        model_revision="model-v1",
        features=("x", "y"),
    )

    assert first == second
    assert changed.input_sha256 != first.input_sha256
    assert changed.evidence_sha256 != first.evidence_sha256


def test_evidence_binds_the_reviewed_policy() -> None:
    row = pd.DataFrame([{"x": 1.0, "y": 1.0}])
    strict = audit_pairwise_interactions(
        FormulaModel(interaction=0.02),
        row,
        model_revision="model-v1",
        features=("x", "y"),
        policy=InteractionPolicy(max_abs_interaction=0.03),
    )
    loose = audit_pairwise_interactions(
        FormulaModel(interaction=0.02),
        row,
        model_revision="model-v1",
        features=("x", "y"),
        policy=InteractionPolicy(max_abs_interaction=0.04),
    )

    assert strict.accepted is True
    assert loose.accepted is True
    assert strict.policy.max_abs_interaction == 0.03
    assert strict.evidence_sha256 != loose.evidence_sha256


def test_prediction_budget_fails_before_model_execution() -> None:
    model = FormulaModel()

    with pytest.raises(InteractionAuditError, match="requires 4 predictions"):
        audit_pairwise_interactions(
            model,
            pd.DataFrame([{"x": 1.0, "y": 1.0}]),
            model_revision="model-v1",
            features=("x", "y"),
            policy=InteractionPolicy(max_prediction_calls=3),
        )

    assert model.calls == 0


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (pd.DataFrame([{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 2.0}]), "one-row"),
        (pd.DataFrame([{"y": 1.0, "x": 1.0}]), "feature order"),
    ],
)
def test_invalid_rows_fail_closed(row: pd.DataFrame, message: str) -> None:
    with pytest.raises(InteractionAuditError, match=message):
        audit_pairwise_interactions(
            FormulaModel(),
            row,
            model_revision="model-v1",
            features=("x", "y"),
        )


def test_invalid_model_probability_fails_closed() -> None:
    with pytest.raises(InteractionAuditError, match="finite"):
        audit_pairwise_interactions(
            FormulaModel(invalid=True),
            pd.DataFrame([{"x": 1.0, "y": 1.0}]),
            model_revision="model-v1",
            features=("x", "y"),
        )


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1", "not-a-number"])
def test_cli_rejects_invalid_interaction_threshold(value: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--max-pair-interaction", value])

    assert exc_info.value.code == 2
