from __future__ import annotations

import math

import pandas as pd
import pytest

from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.importance_stability import (
    ImportanceStabilityPolicy,
    audit_importance_stability,
    bootstrap_permutation_importance_runs,
)
from creditxai.model import CreditRiskModel


def test_consistent_global_importance_passes() -> None:
    runs = [
        {"debt_ratio": 0.40 + index * 0.001, "utilization": 0.20, "income": 0.05}
        for index in range(10)
    ]

    report = audit_importance_stability(
        runs,
        model_id="credit-risk-v1",
        cohort_id="review-2026-11",
        policy=ImportanceStabilityPolicy(top_k=2),
    )

    assert report.stable
    assert report.kendall_w == pytest.approx(1.0)
    assert report.consensus_top_k == ("debt_ratio", "utilization")
    assert len(report.evidence_digest) == 64
    assert report.as_dict()["schema_version"] == "importance-stability-report-v1"


def test_cyclic_rankings_fail_concordance_and_top_k_frequency() -> None:
    orderings = [
        ("a", "b", "c"),
        ("b", "c", "a"),
        ("c", "a", "b"),
    ]
    runs = [
        {feature: float(3 - rank) for rank, feature in enumerate(orderings[index % 3])}
        for index in range(12)
    ]

    report = audit_importance_stability(
        runs,
        model_id="credit-risk-v1",
        cohort_id="review-2026-11",
        policy=ImportanceStabilityPolicy(top_k=1, min_runs=10),
    )

    assert not report.stable
    assert report.kendall_w == pytest.approx(0.0)
    assert set(report.violations) == {
        "rank_concordance_below_threshold",
        "consensus_top_k_frequency_below_threshold",
    }


@pytest.mark.parametrize(
    ("runs", "message"),
    [
        ([{"a": 1.0, "b": 0.0}] * 9, "run count"),
        ([{"a": 1.0, "b": 0.0}] * 9 + [{"a": 1.0, "c": 0.0}], "same feature"),
        ([{"a": math.nan, "b": 0.0}] * 10, "finite"),
    ],
)
def test_invalid_importance_evidence_fails_closed(
    runs: list[dict[str, float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        audit_importance_stability(
            runs,
            model_id="credit-risk-v1",
            cohort_id="review-2026-11",
        )


def test_stratified_bootstrap_is_deterministic() -> None:
    frame = synthetic_credit_data(rows=1000, seed=11)
    model, _ = CreditRiskModel.fit(frame)
    holdout = frame.iloc[800:].copy()

    first = bootstrap_permutation_importance_runs(
        model,
        holdout,
        runs=5,
        repeats=1,
        seed=17,
    )
    second = bootstrap_permutation_importance_runs(
        model,
        holdout,
        runs=5,
        repeats=1,
        seed=17,
    )

    assert first == second
    assert all(set(run) == set(FEATURES) for run in first)
    report = audit_importance_stability(
        first,
        model_id="synthetic-credit-v1",
        cohort_id="heldout-seed-11",
        policy=ImportanceStabilityPolicy(
            top_k=3,
            min_runs=5,
            min_kendall_w=0.0,
            min_consensus_top_k_frequency=0.0,
        ),
    )
    assert report.runs == 5
    assert report.features == len(FEATURES)
    assert 0.0 <= report.kendall_w <= 1.0

    oversized_work = pd.concat([holdout] * 50, ignore_index=True)
    with pytest.raises(ValueError, match="work exceeds"):
        bootstrap_permutation_importance_runs(
            model,
            oversized_work,
            runs=200,
            repeats=20,
            seed=17,
        )
