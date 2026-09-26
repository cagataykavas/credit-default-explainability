"""Fail-closed cohort drift audit for tabular feature attributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

SCHEMA_VERSION = "credit-attribution-drift/v1"
REPORT_VERSION = "credit-attribution-drift-report/v1"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "monitor_id",
        "model_sha256",
        "explainer_sha256",
        "created_at",
        "reference_window",
        "current_window",
        "features",
        "prediction",
    }
)
_WINDOW_FIELDS = frozenset(
    {"manifest_sha256", "sample_count", "started_at", "ended_at"}
)
_FEATURE_FIELDS = frozenset(
    {
        "feature_id",
        "risk",
        "bin_edges_sha256",
        "reference_counts",
        "current_counts",
        "reference_mean_abs",
        "current_mean_abs",
        "reference_topk_rate",
        "current_topk_rate",
    }
)
_HISTOGRAM_FIELDS = frozenset(
    {"bin_edges_sha256", "reference_counts", "current_counts"}
)
_RISKS = frozenset({"standard", "critical"})
_MAX_FINDINGS = 256


def _invalid(message: str) -> NoReturn:
    raise ValueError(message)


def _integer(name: str, value: object, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        _invalid(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(name: str, value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        _invalid(f"{name} must be finite and in [{minimum}, {maximum}]")
    return number


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        _invalid(f"{name} must be a bounded identifier")
    return value


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _invalid(f"{name} must be a canonical lowercase SHA-256 digest")
    return value


def _timestamp(name: str, value: object) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        _invalid(f"{name} must be a bounded RFC 3339 timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{name} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _invalid(f"{name} must include a timezone offset")
    return parsed.astimezone(UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _invalid("JSON objects must not contain duplicate fields")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _invalid(f"non-finite JSON constant is not allowed: {value}")


@dataclass(frozen=True)
class DriftPolicy:
    feature_js_drift_threshold: float = 0.10
    topk_rate_drift_threshold: float = 0.15
    mean_abs_relative_drift_threshold: float = 0.50
    max_drifting_feature_fraction: float = 0.20
    max_weighted_mean_feature_js: float = 0.05
    max_single_feature_js: float = 0.25
    max_prediction_js: float = 0.15
    require_no_critical_drift: bool = True
    mean_abs_floor: float = 1e-12
    min_samples_per_window: int = 200
    max_artifact_age_seconds: float = 86_400.0
    max_future_skew_seconds: float = 300.0
    max_features: int = 1_000
    max_bins_per_histogram: int = 64
    max_input_bytes: int = 1_048_576

    def validate(self) -> None:
        for name in (
            "feature_js_drift_threshold",
            "topk_rate_drift_threshold",
            "max_drifting_feature_fraction",
            "max_weighted_mean_feature_js",
            "max_single_feature_js",
            "max_prediction_js",
        ):
            _number(name, getattr(self, name), 0.0, 1.0)
        _number(
            "mean_abs_relative_drift_threshold",
            self.mean_abs_relative_drift_threshold,
            0.0,
            1_000_000.0,
        )
        _number("mean_abs_floor", self.mean_abs_floor, 1e-18, 1.0)
        _integer("min_samples_per_window", self.min_samples_per_window, 1, 100_000_000)
        _number(
            "max_artifact_age_seconds",
            self.max_artifact_age_seconds,
            0.001,
            31_536_000.0,
        )
        _number("max_future_skew_seconds", self.max_future_skew_seconds, 0.0, 86_400.0)
        _integer("max_features", self.max_features, 1, 100_000)
        _integer("max_bins_per_histogram", self.max_bins_per_histogram, 2, 10_000)
        _integer("max_input_bytes", self.max_input_bytes, 1, 16_777_216)
        if not isinstance(self.require_no_critical_drift, bool):
            _invalid("require_no_critical_drift must be boolean")


@dataclass(frozen=True)
class Finding:
    code: str
    evidence: dict[str, object]
    blocking: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": "error" if self.blocking else "diagnostic",
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class DriftReport:
    accepted: bool
    artifact_sha256: str
    monitor_sha256: str
    configuration_sha256: str
    policy_sha256: str
    evaluated_at: str
    metrics: dict[str, int | float]
    findings: tuple[Finding, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": REPORT_VERSION,
            "accepted": self.accepted,
            "artifact_sha256": self.artifact_sha256,
            "monitor_sha256": self.monitor_sha256,
            "configuration_sha256": self.configuration_sha256,
            "policy_sha256": self.policy_sha256,
            "evaluated_at": self.evaluated_at,
            "metrics": self.metrics,
            "findings": [finding.as_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class _Window:
    manifest_sha256: str
    sample_count: int
    started_at: datetime
    ended_at: datetime


@dataclass(frozen=True)
class _Feature:
    feature_id: str
    risk: str
    bin_edges_sha256: str
    reference_counts: tuple[int, ...]
    current_counts: tuple[int, ...]
    reference_mean_abs: float
    current_mean_abs: float
    reference_topk_rate: float
    current_topk_rate: float


@dataclass(frozen=True)
class _Histogram:
    bin_edges_sha256: str
    reference_counts: tuple[int, ...]
    current_counts: tuple[int, ...]


def load_artifact(payload: bytes, *, max_bytes: int = 1_048_576) -> dict[str, Any]:
    """Load a strict and bounded monitoring artifact."""
    limit = _integer("max_bytes", max_bytes, 1, 16_777_216)
    if not payload or len(payload) > limit:
        _invalid("artifact byte size is outside the configured budget")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        _invalid("artifact root must be an object")
    return value


def _parse_window(name: str, value: object, policy: DriftPolicy) -> _Window:
    if not isinstance(value, dict) or set(value) != _WINDOW_FIELDS:
        _invalid(f"{name} must contain exactly the documented fields")
    started = _timestamp(f"{name}.started_at", value["started_at"])
    ended = _timestamp(f"{name}.ended_at", value["ended_at"])
    if started >= ended:
        _invalid(f"{name} must have positive duration")
    return _Window(
        manifest_sha256=_sha256(f"{name}.manifest_sha256", value["manifest_sha256"]),
        sample_count=_integer(
            f"{name}.sample_count",
            value["sample_count"],
            policy.min_samples_per_window,
            100_000_000,
        ),
        started_at=started,
        ended_at=ended,
    )


def _counts(
    name: str,
    value: object,
    *,
    sample_count: int,
    max_bins: int,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not 2 <= len(value) <= max_bins:
        _invalid(f"{name} must contain between 2 and {max_bins} bins")
    result = tuple(_integer(f"{name} count", item, 0, sample_count) for item in value)
    if sum(result) != sample_count:
        _invalid(f"{name} counts must sum to the window sample count")
    return result


def _parse_feature(
    value: object,
    *,
    reference_samples: int,
    current_samples: int,
    policy: DriftPolicy,
) -> _Feature:
    if not isinstance(value, dict) or set(value) != _FEATURE_FIELDS:
        _invalid("each feature must contain exactly the documented fields")
    reference_counts = _counts(
        "reference_counts",
        value["reference_counts"],
        sample_count=reference_samples,
        max_bins=policy.max_bins_per_histogram,
    )
    current_counts = _counts(
        "current_counts",
        value["current_counts"],
        sample_count=current_samples,
        max_bins=policy.max_bins_per_histogram,
    )
    if len(reference_counts) != len(current_counts):
        _invalid("feature histograms must use the same bin count")
    risk = value["risk"]
    if risk not in _RISKS:
        _invalid("risk must be standard or critical")
    return _Feature(
        feature_id=_identifier("feature_id", value["feature_id"]),
        risk=risk,
        bin_edges_sha256=_sha256("bin_edges_sha256", value["bin_edges_sha256"]),
        reference_counts=reference_counts,
        current_counts=current_counts,
        reference_mean_abs=_number(
            "reference_mean_abs", value["reference_mean_abs"], 0.0, 1e100
        ),
        current_mean_abs=_number(
            "current_mean_abs", value["current_mean_abs"], 0.0, 1e100
        ),
        reference_topk_rate=_number(
            "reference_topk_rate", value["reference_topk_rate"], 0.0, 1.0
        ),
        current_topk_rate=_number(
            "current_topk_rate", value["current_topk_rate"], 0.0, 1.0
        ),
    )


def _parse_histogram(
    value: object,
    *,
    reference_samples: int,
    current_samples: int,
    policy: DriftPolicy,
) -> _Histogram:
    if not isinstance(value, dict) or set(value) != _HISTOGRAM_FIELDS:
        _invalid("prediction must contain exactly the documented fields")
    reference_counts = _counts(
        "prediction.reference_counts",
        value["reference_counts"],
        sample_count=reference_samples,
        max_bins=policy.max_bins_per_histogram,
    )
    current_counts = _counts(
        "prediction.current_counts",
        value["current_counts"],
        sample_count=current_samples,
        max_bins=policy.max_bins_per_histogram,
    )
    if len(reference_counts) != len(current_counts):
        _invalid("prediction histograms must use the same bin count")
    return _Histogram(
        bin_edges_sha256=_sha256(
            "prediction.bin_edges_sha256", value["bin_edges_sha256"]
        ),
        reference_counts=reference_counts,
        current_counts=current_counts,
    )


def _jensen_shannon(reference: tuple[int, ...], current: tuple[int, ...]) -> float:
    reference_total = sum(reference)
    current_total = sum(current)
    divergence = 0.0
    for reference_count, current_count in zip(reference, current, strict=True):
        p = reference_count / reference_total
        q = current_count / current_total
        midpoint = (p + q) / 2.0
        if p > 0.0:
            divergence += 0.5 * p * math.log(p / midpoint)
        if q > 0.0:
            divergence += 0.5 * q * math.log(q / midpoint)
    return divergence / math.log(2.0)


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def audit_attribution_drift(
    document: dict[str, Any],
    *,
    evaluated_at: datetime,
    policy: DriftPolicy | None = None,
) -> DriftReport:
    """Compare reference and current attribution cohorts using predeclared bins."""
    active_policy = policy or DriftPolicy()
    active_policy.validate()
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        _invalid("evaluated_at must be timezone-aware")
    now = evaluated_at.astimezone(UTC)
    if not isinstance(document, dict) or set(document) != _DOCUMENT_FIELDS:
        _invalid("artifact must contain exactly the documented fields")
    if document["schema_version"] != SCHEMA_VERSION:
        _invalid("unsupported schema_version")

    monitor_id = _identifier("monitor_id", document["monitor_id"])
    model_sha256 = _sha256("model_sha256", document["model_sha256"])
    explainer_sha256 = _sha256("explainer_sha256", document["explainer_sha256"])
    created_at = _timestamp("created_at", document["created_at"])
    reference = _parse_window(
        "reference_window", document["reference_window"], active_policy
    )
    current = _parse_window("current_window", document["current_window"], active_policy)
    if reference.ended_at > current.started_at:
        _invalid("reference and current windows must be ordered and non-overlapping")
    if current.ended_at > created_at:
        _invalid("artifact cannot predate the current window")

    feature_values = document["features"]
    if not isinstance(feature_values, list) or not feature_values:
        _invalid("features must be a non-empty array")
    if len(feature_values) > active_policy.max_features:
        _invalid("features exceeds max_features")
    features = [
        _parse_feature(
            value,
            reference_samples=reference.sample_count,
            current_samples=current.sample_count,
            policy=active_policy,
        )
        for value in feature_values
    ]
    feature_ids = [feature.feature_id for feature in features]
    if len(set(feature_ids)) != len(feature_ids):
        _invalid("feature_id values must be unique")
    features.sort(key=lambda item: item.feature_id)
    prediction = _parse_histogram(
        document["prediction"],
        reference_samples=reference.sample_count,
        current_samples=current.sample_count,
        policy=active_policy,
    )

    canonical_document = {
        "schema_version": SCHEMA_VERSION,
        "monitor_id": monitor_id,
        "model_sha256": model_sha256,
        "explainer_sha256": explainer_sha256,
        "created_at": _utc(created_at),
        "reference_window": {
            "manifest_sha256": reference.manifest_sha256,
            "sample_count": reference.sample_count,
            "started_at": _utc(reference.started_at),
            "ended_at": _utc(reference.ended_at),
        },
        "current_window": {
            "manifest_sha256": current.manifest_sha256,
            "sample_count": current.sample_count,
            "started_at": _utc(current.started_at),
            "ended_at": _utc(current.ended_at),
        },
        "features": [
            {
                "feature_id": feature.feature_id,
                "risk": feature.risk,
                "bin_edges_sha256": feature.bin_edges_sha256,
                "reference_counts": feature.reference_counts,
                "current_counts": feature.current_counts,
                "reference_mean_abs": feature.reference_mean_abs,
                "current_mean_abs": feature.current_mean_abs,
                "reference_topk_rate": feature.reference_topk_rate,
                "current_topk_rate": feature.current_topk_rate,
            }
            for feature in features
        ],
        "prediction": {
            "bin_edges_sha256": prediction.bin_edges_sha256,
            "reference_counts": prediction.reference_counts,
            "current_counts": prediction.current_counts,
        },
    }
    artifact_sha256 = _digest(
        json.dumps(canonical_document, sort_keys=True, separators=(",", ":"))
    )
    findings: list[Finding] = []
    age_seconds = (now - created_at).total_seconds()
    if age_seconds > active_policy.max_artifact_age_seconds:
        findings.append(
            Finding(
                "STALE_ARTIFACT",
                {
                    "age_seconds": age_seconds,
                    "limit": active_policy.max_artifact_age_seconds,
                },
            )
        )
    if age_seconds < -active_policy.max_future_skew_seconds:
        findings.append(
            Finding(
                "FUTURE_ARTIFACT",
                {
                    "future_skew_seconds": -age_seconds,
                    "limit": active_policy.max_future_skew_seconds,
                },
            )
        )

    feature_js_values: list[float] = []
    feature_weights: list[float] = []
    topk_deltas: list[float] = []
    mean_abs_changes: list[float] = []
    drifting_features = 0
    critical_drifting_features = 0
    for feature in features:
        feature_js = _jensen_shannon(feature.reference_counts, feature.current_counts)
        topk_delta = abs(feature.current_topk_rate - feature.reference_topk_rate)
        mean_abs_change = abs(
            feature.current_mean_abs - feature.reference_mean_abs
        ) / max(feature.reference_mean_abs, active_policy.mean_abs_floor)
        drifted = (
            feature_js > active_policy.feature_js_drift_threshold
            or topk_delta > active_policy.topk_rate_drift_threshold
            or mean_abs_change > active_policy.mean_abs_relative_drift_threshold
        )
        feature_js_values.append(feature_js)
        feature_weights.append(feature.reference_mean_abs)
        topk_deltas.append(topk_delta)
        mean_abs_changes.append(mean_abs_change)
        drifting_features += int(drifted)
        critical_drifting_features += int(drifted and feature.risk == "critical")
        if drifted:
            identity = {
                "feature_sha256": _digest(feature.feature_id),
                "risk": feature.risk,
            }
            findings.append(
                Finding(
                    "FEATURE_ATTRIBUTION_DRIFT",
                    {
                        **identity,
                        "js_divergence": feature_js,
                        "topk_rate_delta": topk_delta,
                        "mean_abs_relative_change": mean_abs_change,
                    },
                    blocking=False,
                )
            )
            if feature.risk == "critical" and active_policy.require_no_critical_drift:
                findings.append(Finding("CRITICAL_FEATURE_DRIFT", identity))

    weight_total = sum(feature_weights)
    if weight_total <= active_policy.mean_abs_floor:
        weighted_mean_feature_js = sum(feature_js_values) / len(feature_js_values)
    else:
        weighted_mean_feature_js = (
            sum(
                value * weight
                for value, weight in zip(
                    feature_js_values, feature_weights, strict=True
                )
            )
            / weight_total
        )
    max_feature_js = max(feature_js_values)
    drifting_feature_fraction = drifting_features / len(features)
    prediction_js = _jensen_shannon(
        prediction.reference_counts, prediction.current_counts
    )

    if max_feature_js > active_policy.max_single_feature_js:
        findings.append(
            Finding(
                "MAX_FEATURE_JS",
                {
                    "observed": max_feature_js,
                    "limit": active_policy.max_single_feature_js,
                },
            )
        )
    if weighted_mean_feature_js > active_policy.max_weighted_mean_feature_js:
        findings.append(
            Finding(
                "WEIGHTED_MEAN_FEATURE_JS",
                {
                    "observed": weighted_mean_feature_js,
                    "limit": active_policy.max_weighted_mean_feature_js,
                },
            )
        )
    if drifting_feature_fraction > active_policy.max_drifting_feature_fraction:
        findings.append(
            Finding(
                "DRIFTING_FEATURE_FRACTION",
                {
                    "observed": drifting_feature_fraction,
                    "limit": active_policy.max_drifting_feature_fraction,
                },
            )
        )
    if prediction_js > active_policy.max_prediction_js:
        findings.append(
            Finding(
                "PREDICTION_DISTRIBUTION_DRIFT",
                {"observed": prediction_js, "limit": active_policy.max_prediction_js},
            )
        )

    findings.sort(
        key=lambda item: (item.code, json.dumps(item.evidence, sort_keys=True))
    )
    finding_count = len(findings)
    if finding_count > _MAX_FINDINGS:
        findings = findings[: _MAX_FINDINGS - 1]
        findings.append(
            Finding(
                "FINDINGS_TRUNCATED",
                {"observed": finding_count, "reported": _MAX_FINDINGS},
            )
        )
    configuration = {
        "model_sha256": model_sha256,
        "explainer_sha256": explainer_sha256,
        "reference_manifest_sha256": reference.manifest_sha256,
        "current_manifest_sha256": current.manifest_sha256,
        "prediction_bin_edges_sha256": prediction.bin_edges_sha256,
    }
    metrics: dict[str, int | float] = {
        "feature_count": len(features),
        "critical_feature_count": sum(
            feature.risk == "critical" for feature in features
        ),
        "drifting_feature_count": drifting_features,
        "critical_drifting_feature_count": critical_drifting_features,
        "drifting_feature_fraction": drifting_feature_fraction,
        "weighted_mean_feature_js": weighted_mean_feature_js,
        "max_feature_js": max_feature_js,
        "max_topk_rate_delta": max(topk_deltas),
        "max_mean_abs_relative_change": max(mean_abs_changes),
        "prediction_js": prediction_js,
        "reference_sample_count": reference.sample_count,
        "current_sample_count": current.sample_count,
        "artifact_age_seconds": age_seconds,
    }
    return DriftReport(
        accepted=not any(finding.blocking for finding in findings),
        artifact_sha256=artifact_sha256,
        monitor_sha256=_digest(monitor_id),
        configuration_sha256=_digest(
            json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        ),
        policy_sha256=_digest(
            json.dumps(asdict(active_policy), sort_keys=True, separators=(",", ":"))
        ),
        evaluated_at=_utc(now),
        metrics=metrics,
        findings=tuple(findings),
    )


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-bytes", type=int, default=1_048_576)
    args = parser.parse_args(argv)
    try:
        policy = DriftPolicy(max_input_bytes=args.max_bytes)
        if args.artifact.stat().st_size > policy.max_input_bytes:
            _invalid("artifact byte size is outside the configured budget")
        document = load_artifact(
            args.artifact.read_bytes(), max_bytes=policy.max_input_bytes
        )
        report = audit_attribution_drift(
            document, evaluated_at=datetime.now(UTC), policy=policy
        )
        payload = json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":"))
        exit_code = 0 if report.accepted else 2
    except (OSError, ValueError):
        payload = json.dumps(
            {
                "schema_version": REPORT_VERSION,
                "accepted": False,
                "error_code": "MALFORMED_ARTIFACT",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        exit_code = 3
    if args.output is None:
        print(payload)
    else:
        _write_atomic(args.output, payload)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
