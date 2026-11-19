from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import pairwise

from .data import FEATURES
from .explain import FeatureContribution, reason_codes

DEFAULT_TOLERANCES = (0.0, 1e-8, 1e-6, 1e-4)
MAX_TOLERANCES = 32


@dataclass(frozen=True)
class ToleranceSnapshot:
    tolerance: float
    neutral_features: tuple[str, ...]
    risk_feature_ranking: tuple[str, ...]
    reason_code_ranking: tuple[str, ...]


@dataclass(frozen=True)
class ToleranceSensitivityReport:
    tolerances: tuple[float, ...]
    top_k: int
    attribution_sha256: str
    audit_sha256: str
    risk_ranking_stable: bool
    reason_codes_stable: bool
    unstable_features: tuple[str, ...]
    snapshots: tuple[ToleranceSnapshot, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def audit_direction_tolerances(
    contributions: list[FeatureContribution],
    *,
    tolerances: tuple[float, ...] = DEFAULT_TOLERANCES,
    top_k: int = 4,
) -> ToleranceSensitivityReport:
    """Audit whether attribution directions and reason rankings survive tolerance changes.

    The raw probability deltas are held fixed. Only the neutral/directional classification
    changes, so a reported instability is attributable to the reviewed numerical policy
    rather than another model call or a changed applicant row.
    """
    selected_tolerances = _validate_tolerances(tolerances)
    _validate_contributions(contributions)
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= len(FEATURES)
    ):
        raise ValueError(f"top_k must be an integer between 1 and {len(FEATURES)}")

    ordered = sorted(
        contributions,
        key=lambda item: (-abs(item.probability_delta), item.feature),
    )
    snapshots: list[ToleranceSnapshot] = []
    directions_by_feature: dict[str, set[str]] = {
        feature: set() for feature in FEATURES
    }
    for tolerance in selected_tolerances:
        classified = [
            FeatureContribution(
                feature=item.feature,
                observed=item.observed,
                baseline=item.baseline,
                probability_delta=item.probability_delta,
                direction=_direction(item.probability_delta, tolerance),
            )
            for item in ordered
        ]
        for item in classified:
            directions_by_feature[item.feature].add(item.direction)
        snapshots.append(
            ToleranceSnapshot(
                tolerance=tolerance,
                neutral_features=tuple(
                    item.feature for item in classified if item.direction == "neutral"
                ),
                risk_feature_ranking=tuple(
                    item.feature
                    for item in classified
                    if item.direction == "increases_risk"
                ),
                reason_code_ranking=tuple(reason_codes(classified, top_k=top_k)),
            )
        )

    attribution_payload = [
        {
            "feature": item.feature,
            "probability_delta_hex": float(item.probability_delta).hex(),
        }
        for item in ordered
    ]
    attribution_sha256 = _digest(attribution_payload)
    audit_payload = {
        "tolerances_hex": [value.hex() for value in selected_tolerances],
        "top_k": top_k,
        "attribution_sha256": attribution_sha256,
        "snapshots": [asdict(item) for item in snapshots],
    }
    risk_rankings = {item.risk_feature_ranking for item in snapshots}
    reason_rankings = {item.reason_code_ranking for item in snapshots}
    return ToleranceSensitivityReport(
        tolerances=selected_tolerances,
        top_k=top_k,
        attribution_sha256=attribution_sha256,
        audit_sha256=_digest(audit_payload),
        risk_ranking_stable=len(risk_rankings) == 1,
        reason_codes_stable=len(reason_rankings) == 1,
        unstable_features=tuple(
            feature for feature in FEATURES if len(directions_by_feature[feature]) > 1
        ),
        snapshots=tuple(snapshots),
    )


def _validate_tolerances(values: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(values, tuple) or not 2 <= len(values) <= MAX_TOLERANCES:
        raise ValueError(f"tolerances must contain 2..{MAX_TOLERANCES} values")
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("tolerances must be finite numbers between 0 and 1")
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("tolerances must be finite numbers between 0 and 1")
        normalized.append(number)
    if any(left >= right for left, right in pairwise(normalized)):
        raise ValueError("tolerances must be unique and strictly increasing")
    return tuple(normalized)


def _validate_contributions(contributions: list[FeatureContribution]) -> None:
    if not isinstance(contributions, list) or len(contributions) != len(FEATURES):
        raise ValueError("contributions must contain every model feature exactly once")
    feature_names = [
        item.feature for item in contributions if isinstance(item, FeatureContribution)
    ]
    if len(feature_names) != len(contributions) or set(feature_names) != set(FEATURES):
        raise ValueError("contributions must contain every model feature exactly once")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("contribution feature names must be unique")
    for item in contributions:
        delta = item.probability_delta
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(float(delta))
            or not -1 <= float(delta) <= 1
        ):
            raise ValueError(
                "probability deltas must be finite values between -1 and 1"
            )


def _direction(delta: float, tolerance: float) -> str:
    if abs(delta) <= tolerance:
        return "neutral"
    return "increases_risk" if delta > 0 else "decreases_risk"


def _digest(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return sha256(canonical.encode("utf-8")).hexdigest()
