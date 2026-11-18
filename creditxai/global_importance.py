from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .data import FEATURES, TARGET
from .model import CreditRiskModel

DEFAULT_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "affordability": ("income", "debt_ratio", "utilization"),
    "credit_history": ("late_payments", "account_age_months"),
    "applicant_context": ("age", "employment", "housing"),
}


@dataclass(frozen=True, slots=True)
class GroupImportance:
    group: str
    features: tuple[str, ...]
    importance_mean: float
    importance_std: float
    repeat_drops: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class GroupedPermutationReport:
    scoring: str
    baseline_score: float
    rows: int
    repeats: int
    seed: int
    group_spec_sha256: str
    groups: tuple[GroupImportance, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "scoring": self.scoring,
            "baseline_score": self.baseline_score,
            "rows": self.rows,
            "repeats": self.repeats,
            "seed": self.seed,
            "group_spec_sha256": self.group_spec_sha256,
            "groups": [asdict(group) for group in self.groups],
        }


def _validated_groups(
    feature_groups: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not feature_groups:
        raise ValueError("feature_groups cannot be empty")
    governed = set(FEATURES)
    assigned: set[str] = set()
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for raw_name, raw_features in feature_groups.items():
        name = raw_name.strip()
        if not name:
            raise ValueError("feature group names cannot be blank")
        features = tuple(raw_features)
        if not features:
            raise ValueError(f"feature group {name!r} cannot be empty")
        if len(set(features)) != len(features):
            raise ValueError(f"feature group {name!r} contains a duplicate feature")
        unknown = set(features) - governed
        if unknown:
            raise ValueError(
                f"feature group {name!r} contains unknown features: {sorted(unknown)}"
            )
        overlap = assigned & set(features)
        if overlap:
            raise ValueError(
                f"features assigned to more than one group: {sorted(overlap)}"
            )
        assigned.update(features)
        normalized.append((name, tuple(sorted(features))))
    missing = governed - assigned
    if missing:
        raise ValueError(
            f"feature groups must cover every model feature; missing: {sorted(missing)}"
        )
    return tuple(sorted(normalized))


def _group_seed(seed: int, group: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{group}".encode(),
        digest_size=8,
        person=b"xai-group",
    ).digest()
    return int.from_bytes(digest, "big")


def grouped_permutation_importance(
    model: CreditRiskModel,
    frame: pd.DataFrame,
    *,
    feature_groups: Mapping[str, Sequence[str]] = DEFAULT_FEATURE_GROUPS,
    repeats: int = 8,
    seed: int = 42,
) -> GroupedPermutationReport:
    """Jointly permute governed raw-feature groups and retain repeat evidence.

    All columns in a group use the same row permutation. This preserves their
    within-row relationship while breaking the group's relationship to the target.
    It is a model-reliance audit, not a causal importance estimate.
    """
    if repeats < 2:
        raise ValueError("repeats must be at least two")
    required = set(FEATURES) | {TARGET}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"evaluation frame is missing columns: {sorted(missing_columns)}"
        )
    if len(frame) < 20:
        raise ValueError("grouped permutation evaluation requires at least 20 rows")
    labels = frame[TARGET]
    if labels.nunique(dropna=False) != 2 or labels.isna().any():
        raise ValueError("ROC-AUC evaluation requires two non-missing target classes")

    groups = _validated_groups(feature_groups)
    group_payload = json.dumps(groups, separators=(",", ":"), ensure_ascii=True)
    group_fingerprint = hashlib.sha256(group_payload.encode()).hexdigest()
    evaluation = frame.reset_index(drop=True)
    baseline_probabilities = model.pipeline.predict_proba(evaluation[FEATURES])[:, 1]
    baseline_score = float(roc_auc_score(labels, baseline_probabilities))

    results: list[GroupImportance] = []
    for group_name, features in groups:
        rng = np.random.default_rng(_group_seed(seed, group_name))
        drops: list[float] = []
        for _ in range(repeats):
            order = rng.permutation(len(evaluation))
            perturbed = evaluation.copy()
            perturbed.loc[:, list(features)] = evaluation.iloc[order][
                list(features)
            ].to_numpy()
            score = roc_auc_score(
                labels,
                model.pipeline.predict_proba(perturbed[FEATURES])[:, 1],
            )
            drops.append(float(baseline_score - score))
        results.append(
            GroupImportance(
                group=group_name,
                features=features,
                importance_mean=float(np.mean(drops)),
                importance_std=float(np.std(drops, ddof=1)),
                repeat_drops=tuple(drops),
            )
        )

    return GroupedPermutationReport(
        scoring="roc_auc",
        baseline_score=baseline_score,
        rows=len(evaluation),
        repeats=repeats,
        seed=seed,
        group_spec_sha256=group_fingerprint,
        groups=tuple(
            sorted(results, key=lambda item: (-item.importance_mean, item.group))
        ),
    )


def permutation_feature_importance(
    model: CreditRiskModel,
    frame: pd.DataFrame,
    *,
    repeats: int = 8,
    seed: int = 42,
) -> list[dict[str, float | str]]:
    result = permutation_importance(
        model.pipeline,
        frame[FEATURES],
        frame[TARGET],
        scoring="roc_auc",
        n_repeats=repeats,
        random_state=seed,
    )
    rows = [
        {
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for feature, mean, std in zip(
            FEATURES, result.importances_mean, result.importances_std
        )
    ]
    return sorted(rows, key=lambda item: float(item["importance_mean"]), reverse=True)
