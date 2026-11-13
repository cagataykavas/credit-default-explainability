from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean, stdev

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .data import FEATURES, TARGET
from .model import CreditRiskModel

MAX_BOOTSTRAP_ROWS = 10_000
MAX_SCORING_ROW_EVALUATIONS = 25_000_000


@dataclass(frozen=True)
class ImportanceStabilityPolicy:
    top_k: int = 3
    min_runs: int = 10
    max_runs: int = 200
    max_features: int = 256
    min_kendall_w: float = 0.70
    min_consensus_top_k_frequency: float = 0.80

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.min_runs < 2:
            raise ValueError("min_runs must be at least 2")
        if self.max_runs < self.min_runs:
            raise ValueError("max_runs must be at least min_runs")
        if self.max_features < 2:
            raise ValueError("max_features must be at least 2")
        if not 0.0 <= self.min_kendall_w <= 1.0:
            raise ValueError("min_kendall_w must be between 0 and 1")
        if not 0.0 <= self.min_consensus_top_k_frequency <= 1.0:
            raise ValueError("min_consensus_top_k_frequency must be between 0 and 1")


@dataclass(frozen=True)
class FeatureStability:
    feature: str
    importance_mean: float
    importance_std: float
    lower_quantile: float
    upper_quantile: float
    mean_rank: float
    rank_std: float
    top_k_frequency: float


@dataclass(frozen=True)
class ImportanceStabilityReport:
    model_id: str
    cohort_id: str
    runs: int
    features: int
    kendall_w: float
    consensus_top_k: tuple[str, ...]
    stable: bool
    violations: tuple[str, ...]
    feature_stability: tuple[FeatureStability, ...]
    evidence_digest: str
    policy: ImportanceStabilityPolicy

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = "importance-stability-report-v1"
        return payload


def _validate_identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f"{name} must be a non-empty string of at most 200 characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _rank_run(run: Mapping[str, float]) -> tuple[dict[str, float], int]:
    ordered = sorted(run.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    tie_correction = 0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for feature, _ in ordered[index:end]:
            ranks[feature] = average_rank
        tie_size = end - index
        if tie_size > 1:
            tie_correction += tie_size**3 - tie_size
        index = end
    return ranks, tie_correction


def _kendall_w(rankings: Sequence[dict[str, float]], tie_correction: int) -> float:
    run_count = len(rankings)
    features = sorted(rankings[0])
    feature_count = len(features)
    expected_rank_sum = run_count * (feature_count + 1) / 2.0
    squared_deviation = sum(
        (sum(ranking[feature] for ranking in rankings) - expected_rank_sum) ** 2
        for feature in features
    )
    denominator = (
        run_count**2 * (feature_count**3 - feature_count) - run_count * tie_correction
    )
    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, 12.0 * squared_deviation / denominator))


def _evidence_digest(
    runs: Sequence[Mapping[str, float]],
    *,
    model_id: str,
    cohort_id: str,
    policy: ImportanceStabilityPolicy,
) -> str:
    payload = {
        "schema_version": "importance-stability-evidence-v1",
        "model_id": model_id,
        "cohort_id": cohort_id,
        "policy": asdict(policy),
        "runs": [dict(sorted(run.items())) for run in runs],
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_importance_stability(
    importance_runs: Sequence[Mapping[str, float]],
    *,
    model_id: str,
    cohort_id: str,
    policy: ImportanceStabilityPolicy | None = None,
) -> ImportanceStabilityReport:
    active_policy = policy or ImportanceStabilityPolicy()
    checked_model_id = _validate_identifier(model_id, name="model_id")
    checked_cohort_id = _validate_identifier(cohort_id, name="cohort_id")
    if not active_policy.min_runs <= len(importance_runs) <= active_policy.max_runs:
        raise ValueError(
            f"importance run count must be between {active_policy.min_runs} "
            f"and {active_policy.max_runs}"
        )

    normalized: list[dict[str, float]] = []
    expected_features: set[str] | None = None
    for run in importance_runs:
        if not isinstance(run, Mapping):
            raise TypeError("each importance run must be a feature-to-score mapping")
        values: dict[str, float] = {}
        for feature, value in run.items():
            if not isinstance(feature, str) or not feature or len(feature) > 200:
                raise ValueError(
                    "feature names must be non-empty strings of at most 200 characters"
                )
            if isinstance(value, bool):
                raise TypeError("importance scores must be numeric and not boolean")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError("importance scores must be numeric") from exc
            if not math.isfinite(numeric):
                raise ValueError("importance scores must be finite")
            values[feature] = numeric
        if not 2 <= len(values) <= active_policy.max_features:
            raise ValueError(
                f"each run must contain between 2 and {active_policy.max_features} features"
            )
        feature_set = set(values)
        if expected_features is None:
            expected_features = feature_set
        elif feature_set != expected_features:
            raise ValueError("all importance runs must contain the same feature set")
        normalized.append(values)

    assert expected_features is not None
    if active_policy.top_k > len(expected_features):
        raise ValueError("top_k must not exceed the number of features")

    rankings: list[dict[str, float]] = []
    tie_correction = 0
    for run in normalized:
        ranking, run_ties = _rank_run(run)
        rankings.append(ranking)
        tie_correction += run_ties
    kendall_w = _kendall_w(rankings, tie_correction)

    feature_rows: list[FeatureStability] = []
    for feature in sorted(expected_features):
        scores = [run[feature] for run in normalized]
        ranks = [ranking[feature] for ranking in rankings]
        feature_rows.append(
            FeatureStability(
                feature=feature,
                importance_mean=float(fmean(scores)),
                importance_std=float(stdev(scores)),
                lower_quantile=float(np.quantile(scores, 0.025)),
                upper_quantile=float(np.quantile(scores, 0.975)),
                mean_rank=float(fmean(ranks)),
                rank_std=float(stdev(ranks)),
                top_k_frequency=sum(rank <= active_policy.top_k for rank in ranks)
                / len(ranks),
            )
        )
    feature_rows.sort(key=lambda row: (-row.importance_mean, row.feature))
    consensus = tuple(row.feature for row in feature_rows[: active_policy.top_k])
    by_feature = {row.feature: row for row in feature_rows}
    consensus_frequency = min(
        by_feature[feature].top_k_frequency for feature in consensus
    )

    violations: list[str] = []
    if kendall_w < active_policy.min_kendall_w:
        violations.append("rank_concordance_below_threshold")
    if consensus_frequency < active_policy.min_consensus_top_k_frequency:
        violations.append("consensus_top_k_frequency_below_threshold")
    return ImportanceStabilityReport(
        model_id=checked_model_id,
        cohort_id=checked_cohort_id,
        runs=len(normalized),
        features=len(expected_features),
        kendall_w=kendall_w,
        consensus_top_k=consensus,
        stable=not violations,
        violations=tuple(violations),
        feature_stability=tuple(feature_rows),
        evidence_digest=_evidence_digest(
            normalized,
            model_id=checked_model_id,
            cohort_id=checked_cohort_id,
            policy=active_policy,
        ),
        policy=active_policy,
    )


def bootstrap_permutation_importance_runs(
    model: CreditRiskModel,
    frame: pd.DataFrame,
    *,
    runs: int = 20,
    repeats: int = 2,
    seed: int = 42,
    max_rows: int = MAX_BOOTSTRAP_ROWS,
) -> list[dict[str, float]]:
    if not 2 <= runs <= 200:
        raise ValueError("runs must be between 2 and 200")
    if not 1 <= repeats <= 20:
        raise ValueError("repeats must be between 1 and 20")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not 20 <= max_rows <= MAX_BOOTSTRAP_ROWS:
        raise ValueError(f"max_rows must be between 20 and {MAX_BOOTSTRAP_ROWS}")
    if not 20 <= len(frame) <= max_rows:
        raise ValueError(f"frame must contain between 20 and {max_rows} rows")
    scoring_work = runs * repeats * len(frame) * len(FEATURES)
    if scoring_work > MAX_SCORING_ROW_EVALUATIONS:
        raise ValueError("bootstrap scoring work exceeds the audit limit")
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
    labels = frame[TARGET].to_numpy()
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("frame must contain both target classes")

    random = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in classes]
    importance_runs: list[dict[str, float]] = []
    for run_index in range(runs):
        sampled_indices = np.concatenate(
            [
                random.choice(indices, size=len(indices), replace=True)
                for indices in class_indices
            ]
        )
        random.shuffle(sampled_indices)
        sample = frame.iloc[sampled_indices]
        result = permutation_importance(
            model.pipeline,
            sample[FEATURES],
            sample[TARGET],
            scoring="roc_auc",
            n_repeats=repeats,
            random_state=seed + run_index,
        )
        importance_runs.append(
            {
                feature: float(value)
                for feature, value in zip(FEATURES, result.importances_mean)
            }
        )
    return importance_runs
