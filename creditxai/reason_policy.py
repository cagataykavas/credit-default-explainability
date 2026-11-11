from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Protocol

from .data import FEATURES

POLICY_VERSION = "synthetic-credit-reasons-v1"
DEFAULT_MATERIALITY = 0.001
ELIGIBLE_REASON_FEATURES = frozenset(
    {"income", "debt_ratio", "late_payments", "utilization", "account_age_months"}
)
EXCLUDED_REASON_FEATURES = frozenset({"age", "employment", "housing"})
REASON_LABELS = {
    "income": "income level",
    "debt_ratio": "debt-to-income burden",
    "late_payments": "late-payment history",
    "utilization": "credit utilization",
    "account_age_months": "account history length",
}


class ContributionLike(Protocol):
    feature: str
    probability_delta: float
    direction: str


@dataclass(frozen=True, slots=True)
class GovernedReasonCodes:
    policy_version: str
    materiality: float
    source_features: tuple[str, ...]
    reason_codes: tuple[str, ...]
    excluded_material_features: tuple[str, ...]
    requires_policy_review: bool
    findings: tuple[str, ...]

    def policy_evidence(self) -> dict[str, object]:
        evidence = asdict(self)
        evidence.pop("reason_codes")
        return evidence


def _validate_policy_definition() -> None:
    expected = set(FEATURES)
    if len(expected) != len(FEATURES):
        raise RuntimeError("model feature schema must not contain duplicates")
    eligible = set(ELIGIBLE_REASON_FEATURES)
    excluded = set(EXCLUDED_REASON_FEATURES)
    if eligible & excluded or eligible | excluded != expected:
        raise RuntimeError("reason-code policy must partition the model feature schema")
    if set(REASON_LABELS) != eligible:
        raise RuntimeError(
            "every eligible reason feature must have exactly one governed label"
        )


def governed_reason_codes(
    contributions: Sequence[ContributionLike],
    *,
    top_k: int = 4,
    materiality: float = DEFAULT_MATERIALITY,
    require_reason: bool = False,
) -> GovernedReasonCodes:
    """Select policy-eligible reasons while retaining excluded-feature evidence."""
    _validate_policy_definition()
    if type(top_k) is not int or not 1 <= top_k <= len(FEATURES):
        raise ValueError(f"top_k must be an integer between 1 and {len(FEATURES)}")
    if isinstance(materiality, bool) or not isinstance(materiality, (int, float)):
        raise TypeError("materiality must be a finite number between 0 and 1")
    materiality = float(materiality)
    if not isfinite(materiality) or not 0.0 <= materiality <= 1.0:
        raise ValueError("materiality must be a finite number between 0 and 1")
    if not isinstance(require_reason, bool):
        raise TypeError("require_reason must be a boolean")

    by_feature: dict[str, ContributionLike] = {}
    for position, contribution in enumerate(contributions, start=1):
        feature = getattr(contribution, "feature", None)
        delta = getattr(contribution, "probability_delta", None)
        direction = getattr(contribution, "direction", None)
        if not isinstance(feature, str) or feature not in FEATURES:
            raise ValueError(f"contribution {position} has an unknown feature")
        if feature in by_feature:
            raise ValueError(f"contribution {position} duplicates a feature")
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not isfinite(delta)
        ):
            raise ValueError(
                f"contribution {position} probability_delta must be finite"
            )
        expected_direction = "increases_risk" if delta > 0 else "decreases_risk"
        if direction != expected_direction:
            raise ValueError(
                f"contribution {position} direction contradicts probability_delta"
            )
        by_feature[feature] = contribution

    missing = sorted(set(FEATURES) - set(by_feature))
    if missing:
        raise ValueError("contributions do not cover the complete model feature schema")

    material_positive = sorted(
        (
            contribution
            for contribution in by_feature.values()
            if contribution.probability_delta > 0
            and contribution.probability_delta >= materiality
        ),
        key=lambda contribution: (
            -contribution.probability_delta,
            contribution.feature,
        ),
    )
    eligible = [
        contribution
        for contribution in material_positive
        if contribution.feature in ELIGIBLE_REASON_FEATURES
    ][:top_k]
    excluded = tuple(
        contribution.feature
        for contribution in material_positive
        if contribution.feature in EXCLUDED_REASON_FEATURES
    )

    findings: list[str] = []
    if excluded:
        findings.append("excluded_feature_materially_increased_score")
    if require_reason and not eligible:
        findings.append("no_material_policy_eligible_reason")

    return GovernedReasonCodes(
        policy_version=POLICY_VERSION,
        materiality=materiality,
        source_features=tuple(contribution.feature for contribution in eligible),
        reason_codes=tuple(
            REASON_LABELS[contribution.feature] for contribution in eligible
        ),
        excluded_material_features=excluded,
        requires_policy_review=bool(findings),
        findings=tuple(findings),
    )
