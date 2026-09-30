from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .data import FEATURES, synthetic_credit_data
from .explain import FeatureContribution, local_perturbation_attribution
from .model import CATEGORICAL, NUMERIC, CreditRiskModel

EXIT_ACCEPTED = 0
EXIT_REJECTED = 2
EXIT_MALFORMED = 3

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_IDENTIFIER_BYTES = 128


class FaithfulnessAuditError(ValueError):
    """The audit policy, cohort, model output, or attribution is malformed."""


class RiskModel(Protocol):
    def predict_probability(self, row: pd.DataFrame) -> float: ...

    def baseline_row(self) -> dict[str, Any]: ...


AttributionFunction = Callable[[RiskModel, pd.DataFrame], Sequence[FeatureContribution]]


@dataclass(frozen=True, slots=True)
class FaithfulnessPolicy:
    min_output_gap: float = 0.01
    min_eligible_case_fraction: float = 0.50
    top_k: int = 3
    min_top_k_recovery: float = 0.0
    min_ordering_advantage: float = 0.0
    max_prediction_drift: float = 1e-10
    max_failed_case_fraction: float = 0.10
    max_cases: int = 512
    max_features: int = 32
    max_prediction_calls: int = 50_000
    max_findings: int = 256

    def __post_init__(self) -> None:
        integer_fields = (
            "top_k",
            "max_cases",
            "max_features",
            "max_prediction_calls",
            "max_findings",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise FaithfulnessAuditError(f"{field} must be an integer")
        if not 1 <= self.top_k <= self.max_features:
            raise FaithfulnessAuditError("top_k must be within the feature budget")
        if not 1 <= self.max_cases <= 10_000:
            raise FaithfulnessAuditError("max_cases must be between 1 and 10000")
        if not 1 <= self.max_features <= 256:
            raise FaithfulnessAuditError("max_features must be between 1 and 256")
        if not 1 <= self.max_prediction_calls <= 10_000_000:
            raise FaithfulnessAuditError(
                "max_prediction_calls must be between 1 and 10000000"
            )
        if not 1 <= self.max_findings <= 4096:
            raise FaithfulnessAuditError("max_findings must be between 1 and 4096")
        bounded = (
            "min_output_gap",
            "min_eligible_case_fraction",
            "min_top_k_recovery",
            "max_prediction_drift",
            "max_failed_case_fraction",
        )
        for field in bounded:
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise FaithfulnessAuditError(f"{field} must be finite and in [0, 1]")
        value = self.min_ordering_advantage
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not -2.0 <= value <= 2.0
        ):
            raise FaithfulnessAuditError(
                "min_ordering_advantage must be finite and in [-2, 2]"
            )

    @property
    def digest(self) -> str:
        return _sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class CaseMetric:
    case_ref: str
    output_gap: float
    attribution_auc: float
    reverse_auc: float
    ordering_advantage: float
    top_k_recovery: float
    minimum_recovery: float
    maximum_recovery: float
    prediction_drift: float
    passed: bool


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    case_ref: str | None
    observed: float
    limit: float


@dataclass(frozen=True, slots=True)
class FaithfulnessReport:
    schema_version: int
    outcome: str
    model_sha256: str
    cohort_sha256: str
    explainer_sha256: str
    policy_sha256: str
    cases: int
    eligible_cases: int
    excluded_low_contrast_cases: int
    failed_cases: int
    prediction_calls: int
    metrics: tuple[CaseMetric, ...]
    findings: tuple[Finding, ...]
    findings_truncated: bool
    evidence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise FaithfulnessAuditError(
            f"{field} must be a canonical lowercase SHA-256 digest"
        )
    return value


def _case_reference(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise FaithfulnessAuditError("cohort indices must be strings or integers")
    encoded = str(value).encode("utf-8")
    if not encoded or len(encoded) > MAX_IDENTIFIER_BYTES:
        raise FaithfulnessAuditError("cohort index exceeds the identifier budget")
    return hashlib.sha256(encoded).hexdigest()


def _finite_probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FaithfulnessAuditError("model probability must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise FaithfulnessAuditError("model probability must be finite and in [0, 1]")
    return result


def _validate_numeric(feature: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FaithfulnessAuditError(f"{feature} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise FaithfulnessAuditError(f"{feature} must be finite")
    if feature == "income" and number <= 0:
        raise FaithfulnessAuditError("income must be positive")
    if feature in {"debt_ratio", "utilization"} and not 0.0 <= number <= 1.0:
        raise FaithfulnessAuditError(f"{feature} must be in [0, 1]")
    if feature in {"late_payments", "account_age_months", "age"} and (
        number < 0 or not number.is_integer()
    ):
        raise FaithfulnessAuditError(f"{feature} must be a non-negative integer")


def _validate_categorical(feature: str, value: object) -> None:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 128:
        raise FaithfulnessAuditError(f"{feature} must be a bounded non-empty string")


def _validate_row(values: dict[str, Any]) -> None:
    if set(values) != set(FEATURES):
        raise FaithfulnessAuditError(
            "row feature set does not match the model contract"
        )
    for feature in NUMERIC:
        _validate_numeric(feature, values[feature])
    for feature in CATEGORICAL:
        _validate_categorical(feature, values[feature])


class _CountingModel:
    def __init__(self, model: RiskModel, maximum: int) -> None:
        self._model = model
        self._maximum = maximum
        self.calls = 0

    def predict_probability(self, row: pd.DataFrame) -> float:
        if self.calls >= self._maximum:
            raise FaithfulnessAuditError("prediction-call budget exhausted")
        self.calls += 1
        return _finite_probability(self._model.predict_probability(row))

    def baseline_row(self) -> dict[str, Any]:
        values = self._model.baseline_row()
        if not isinstance(values, dict):
            raise FaithfulnessAuditError("model baseline must be an object")
        _validate_row(values)
        return dict(values)


def _baseline_frame(row: pd.DataFrame, baseline: dict[str, Any]) -> pd.DataFrame:
    output = row.copy(deep=True)
    for feature in FEATURES:
        output.loc[output.index[0], feature] = baseline[feature]
    return output


def _validate_attributions(
    values: Sequence[FeatureContribution],
) -> tuple[FeatureContribution, ...]:
    if isinstance(values, (str, bytes)) or len(values) != len(FEATURES):
        raise FaithfulnessAuditError(
            "explainer must return one attribution per feature"
        )
    output = tuple(values)
    if any(not isinstance(item, FeatureContribution) for item in output):
        raise FaithfulnessAuditError("explainer returned an invalid attribution object")
    observed = [item.feature for item in output]
    if len(set(observed)) != len(observed) or set(observed) != set(FEATURES):
        raise FaithfulnessAuditError(
            "attribution feature identities are incomplete or duplicated"
        )
    for item in output:
        if not math.isfinite(item.probability_delta):
            raise FaithfulnessAuditError("attribution values must be finite")
    return output


def _restoration_curve(
    model: _CountingModel,
    row: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    order: Sequence[str],
    observed_probability: float,
    output_gap: float,
) -> tuple[float, ...]:
    candidate = baseline_frame.copy(deep=True)
    values = [0.0]
    denominator = abs(output_gap)
    for feature in order:
        candidate.loc[candidate.index[0], feature] = row.iloc[0][feature]
        probability = model.predict_probability(candidate)
        values.append(1.0 - abs(observed_probability - probability) / denominator)
    return tuple(values)


def _trapezoid(values: Sequence[float]) -> float:
    intervals = len(values) - 1
    if intervals <= 0:
        raise FaithfulnessAuditError(
            "a restoration curve requires at least one interval"
        )
    return sum((left + right) / 2.0 for left, right in pairwise(values)) / intervals


def audit_sequential_ablation(
    model: RiskModel,
    cohort: pd.DataFrame,
    *,
    model_sha256: str,
    cohort_sha256: str,
    explainer_sha256: str,
    policy: FaithfulnessPolicy | None = None,
    attribution_fn: AttributionFunction = local_perturbation_attribution,
) -> FaithfulnessReport:
    """Audit whether attribution-ranked restoration beats a reverse-order control."""

    active_policy = policy or FaithfulnessPolicy()
    model_digest = _digest(model_sha256, "model_sha256")
    cohort_digest = _digest(cohort_sha256, "cohort_sha256")
    explainer_digest = _digest(explainer_sha256, "explainer_sha256")
    if not isinstance(cohort, pd.DataFrame):
        raise FaithfulnessAuditError("cohort must be a pandas DataFrame")
    if not 1 <= len(cohort) <= active_policy.max_cases:
        raise FaithfulnessAuditError("cohort violates the case budget")
    if tuple(cohort.columns) != tuple(FEATURES):
        raise FaithfulnessAuditError("cohort columns must exactly match FEATURES order")
    if len(FEATURES) > active_policy.max_features:
        raise FaithfulnessAuditError("model feature count exceeds the feature budget")
    if active_policy.top_k > len(FEATURES):
        raise FaithfulnessAuditError("top_k exceeds the model feature count")
    if not cohort.index.is_unique:
        raise FaithfulnessAuditError("cohort indices must be unique")

    counted_model = _CountingModel(model, active_policy.max_prediction_calls)
    baseline = counted_model.baseline_row()
    metrics: list[CaseMetric] = []
    findings: list[Finding] = []
    excluded = 0
    failed = 0

    for position in range(len(cohort)):
        row = cohort.iloc[[position]].copy(deep=True)
        values = row.iloc[0].to_dict()
        _validate_row(values)
        case_ref = _case_reference(cohort.index[position])
        baseline_frame = _baseline_frame(row, baseline)
        observed = counted_model.predict_probability(row)
        baseline_probability = counted_model.predict_probability(baseline_frame)
        output_gap = observed - baseline_probability
        if abs(output_gap) < active_policy.min_output_gap:
            excluded += 1
            continue

        attributions = _validate_attributions(attribution_fn(counted_model, row))
        order = tuple(item.feature for item in attributions)
        curve = _restoration_curve(
            counted_model,
            row,
            baseline_frame,
            order,
            observed,
            output_gap,
        )
        reverse_curve = _restoration_curve(
            counted_model,
            row,
            baseline_frame,
            tuple(reversed(order)),
            observed,
            output_gap,
        )
        repeated_observed = counted_model.predict_probability(row)
        repeated_baseline = counted_model.predict_probability(baseline_frame)
        drift = max(
            abs(observed - repeated_observed),
            abs(baseline_probability - repeated_baseline),
        )
        attribution_auc = _trapezoid(curve)
        reverse_auc = _trapezoid(reverse_curve)
        advantage = attribution_auc - reverse_auc
        top_k_recovery = curve[active_policy.top_k]
        case_findings: list[Finding] = []
        checks = (
            (
                "top_k_recovery_below_minimum",
                top_k_recovery,
                active_policy.min_top_k_recovery,
                lambda observed, limit: observed < limit,
            ),
            (
                "ordering_advantage_below_minimum",
                advantage,
                active_policy.min_ordering_advantage,
                lambda observed, limit: observed < limit,
            ),
            (
                "prediction_drift_above_maximum",
                drift,
                active_policy.max_prediction_drift,
                lambda observed, limit: observed > limit,
            ),
        )
        for code, observed_value, limit, violated in checks:
            if violated(observed_value, limit):
                case_findings.append(
                    Finding(
                        code=code,
                        case_ref=case_ref,
                        observed=round(observed_value, 8),
                        limit=limit,
                    )
                )
        passed = not case_findings
        failed += int(not passed)
        findings.extend(case_findings)
        metrics.append(
            CaseMetric(
                case_ref=case_ref,
                output_gap=round(abs(output_gap), 8),
                attribution_auc=round(attribution_auc, 8),
                reverse_auc=round(reverse_auc, 8),
                ordering_advantage=round(advantage, 8),
                top_k_recovery=round(top_k_recovery, 8),
                minimum_recovery=round(min(curve), 8),
                maximum_recovery=round(max(curve), 8),
                prediction_drift=round(drift, 12),
                passed=passed,
            )
        )

    eligible = len(metrics)
    eligible_fraction = eligible / len(cohort)
    failed_fraction = failed / eligible if eligible else 1.0
    release_findings: list[Finding] = []
    if eligible_fraction < active_policy.min_eligible_case_fraction:
        release_findings.append(
            Finding(
                code="eligible_case_fraction_below_minimum",
                case_ref=None,
                observed=round(eligible_fraction, 8),
                limit=active_policy.min_eligible_case_fraction,
            )
        )
    if failed_fraction > active_policy.max_failed_case_fraction:
        release_findings.append(
            Finding(
                code="failed_case_fraction_above_maximum",
                case_ref=None,
                observed=round(failed_fraction, 8),
                limit=active_policy.max_failed_case_fraction,
            )
        )
    findings.extend(release_findings)
    findings.sort(key=lambda item: (item.code, item.case_ref or ""))
    bounded_findings = tuple(findings[: active_policy.max_findings])
    metrics.sort(key=lambda item: item.case_ref)
    outcome = "accepted" if not release_findings else "rejected"
    report_document = {
        "cases": len(cohort),
        "cohort_sha256": cohort_digest,
        "eligible_cases": eligible,
        "evidence_schema_version": 1,
        "excluded_low_contrast_cases": excluded,
        "explainer_sha256": explainer_digest,
        "failed_cases": failed,
        "findings": [asdict(item) for item in bounded_findings],
        "findings_truncated": len(findings) > len(bounded_findings),
        "metrics": [asdict(item) for item in metrics],
        "model_sha256": model_digest,
        "outcome": outcome,
        "policy_sha256": active_policy.digest,
        "prediction_calls": counted_model.calls,
    }
    return FaithfulnessReport(
        schema_version=1,
        outcome=outcome,
        model_sha256=model_digest,
        cohort_sha256=cohort_digest,
        explainer_sha256=explainer_digest,
        policy_sha256=active_policy.digest,
        cases=len(cohort),
        eligible_cases=eligible,
        excluded_low_contrast_cases=excluded,
        failed_cases=failed,
        prediction_calls=counted_model.calls,
        metrics=tuple(metrics),
        findings=bounded_findings,
        findings_truncated=len(findings) > len(bounded_findings),
        evidence_sha256=_sha256_json(report_document),
    )


def write_report_atomic(path: Path, report: FaithfulnessReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                report.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _frame_digest(frame: pd.DataFrame) -> str:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            {
                key: value.item() if hasattr(value, "item") else value
                for key, value in row.items()
            }
        )
    return _sha256_json(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit sequential-ablation faithfulness for credit explanations."
    )
    parser.add_argument("--rows", type=int, default=2_200)
    parser.add_argument("--audit-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if not 500 <= args.rows <= 100_000:
            raise FaithfulnessAuditError("rows must be between 500 and 100000")
        if not 1 <= args.audit_rows <= 512:
            raise FaithfulnessAuditError("audit_rows must be between 1 and 512")
        frame = synthetic_credit_data(rows=args.rows, seed=args.seed)
        model, _ = CreditRiskModel.fit(frame)
        test = frame.iloc[int(len(frame) * 0.8) :]
        cohort = test.drop(columns="default").iloc[: args.audit_rows][FEATURES]
        if len(cohort) != args.audit_rows:
            raise FaithfulnessAuditError("held-out split is smaller than audit_rows")
        report = audit_sequential_ablation(
            model,
            cohort,
            model_sha256=_sha256_json(
                {
                    "model": "calibrated-logistic-credit-risk",
                    "seed": args.seed,
                    "training_data_sha256": _frame_digest(
                        frame.iloc[: int(len(frame) * 0.8)]
                    ),
                }
            ),
            cohort_sha256=_frame_digest(cohort),
            explainer_sha256=_sha256_json(
                {"method": "one-feature-baseline-perturbation", "version": 1}
            ),
        )
        write_report_atomic(args.output, report)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    except (OSError, FaithfulnessAuditError, ValueError, TypeError) as exc:
        print(
            json.dumps(
                {"outcome": "malformed", "reason": type(exc).__name__},
                sort_keys=True,
            )
        )
        return EXIT_MALFORMED
    return EXIT_ACCEPTED if report.outcome == "accepted" else EXIT_REJECTED


if __name__ == "__main__":
    raise SystemExit(main())
