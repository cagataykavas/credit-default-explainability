"""Governed subgroup gates for attribution-drift artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from creditxai.attribution_drift import (
    DriftPolicy,
    DriftReport,
    Finding,
    audit_attribution_drift,
    load_artifact,
)

SCHEMA_VERSION = "credit-attribution-drift-partition/v1"
REPORT_VERSION = "credit-attribution-drift-partition-report/v1"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_DOCUMENT_FIELDS = frozenset({"schema_version", "partition_id", "aggregate", "slices"})
_SLICE_FIELDS = frozenset({"slice_id", "artifact"})


def _invalid(message: str) -> NoReturn:
    raise ValueError(message)


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        _invalid(f"{name} must be a bounded identifier")
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        _invalid("window timestamps must be bounded RFC 3339 values")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError("window timestamps must be RFC 3339 values") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _invalid("window timestamps must include a timezone offset")
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class SubgroupDriftPolicy:
    """Policy for one mutually exclusive governed partition."""

    governed_slice_ids: tuple[str, ...]
    min_samples_per_slice: int = 50
    max_slices: int = 32
    require_complete_partition: bool = True
    max_input_bytes: int = 4_194_304
    drift_policy: DriftPolicy = field(default_factory=DriftPolicy)

    def validate(self) -> None:
        if not isinstance(self.governed_slice_ids, tuple):
            _invalid("governed_slice_ids must be a tuple")
        if not self.governed_slice_ids:
            _invalid("governed_slice_ids must not be empty")
        governed = [
            _identifier("governed_slice_id", value) for value in self.governed_slice_ids
        ]
        if len(set(governed)) != len(governed):
            _invalid("governed_slice_ids must be unique")
        if (
            isinstance(self.min_samples_per_slice, bool)
            or not isinstance(self.min_samples_per_slice, int)
            or not 1 <= self.min_samples_per_slice <= 100_000_000
        ):
            _invalid("min_samples_per_slice must be an integer in [1, 100000000]")
        if (
            isinstance(self.max_slices, bool)
            or not isinstance(self.max_slices, int)
            or not 1 <= self.max_slices <= 1_000
        ):
            _invalid("max_slices must be an integer in [1, 1000]")
        if len(governed) > self.max_slices:
            _invalid("governed_slice_ids exceeds max_slices")
        if not isinstance(self.require_complete_partition, bool):
            _invalid("require_complete_partition must be boolean")
        if (
            isinstance(self.max_input_bytes, bool)
            or not isinstance(self.max_input_bytes, int)
            or not 1 <= self.max_input_bytes <= 16_777_216
        ):
            _invalid("max_input_bytes must be an integer in [1, 16777216]")
        self.drift_policy.validate()


@dataclass(frozen=True)
class SubgroupResult:
    slice_sha256: str
    reference_sample_count: int
    current_sample_count: int
    report: DriftReport

    def as_dict(self) -> dict[str, object]:
        return {
            "slice_sha256": self.slice_sha256,
            "reference_sample_count": self.reference_sample_count,
            "current_sample_count": self.current_sample_count,
            "report": self.report.as_dict(),
        }


@dataclass(frozen=True)
class SubgroupDriftReport:
    accepted: bool
    artifact_sha256: str
    partition_sha256: str
    policy_sha256: str
    evaluated_at: str
    metrics: dict[str, int | float]
    aggregate_report: DriftReport
    subgroup_results: tuple[SubgroupResult, ...]
    findings: tuple[Finding, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": REPORT_VERSION,
            "accepted": self.accepted,
            "artifact_sha256": self.artifact_sha256,
            "partition_sha256": self.partition_sha256,
            "policy_sha256": self.policy_sha256,
            "evaluated_at": self.evaluated_at,
            "metrics": self.metrics,
            "aggregate_report": self.aggregate_report.as_dict(),
            "subgroup_results": [item.as_dict() for item in self.subgroup_results],
            "findings": [finding.as_dict() for finding in self.findings],
        }


def _feature_configuration(artifact: dict[str, Any]) -> tuple[tuple[object, ...], ...]:
    features = artifact.get("features")
    if not isinstance(features, list):
        _invalid("slice features must be an array")
    try:
        return tuple(
            sorted(
                (
                    item["feature_id"],
                    item["risk"],
                    item["bin_edges_sha256"],
                    len(item["reference_counts"]),
                    len(item["current_counts"]),
                )
                for item in features
            )
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("slice feature configuration is malformed") from exc


def _assert_compatible(aggregate: dict[str, Any], subgroup: dict[str, Any]) -> None:
    for field_name in ("schema_version", "model_sha256", "explainer_sha256"):
        if subgroup.get(field_name) != aggregate.get(field_name):
            _invalid(f"slice {field_name} must match the aggregate")
    for window_name in ("reference_window", "current_window"):
        aggregate_window = aggregate.get(window_name)
        subgroup_window = subgroup.get(window_name)
        if not isinstance(aggregate_window, dict) or not isinstance(
            subgroup_window, dict
        ):
            _invalid("slice windows must be objects")
        for field_name in ("started_at", "ended_at"):
            if _timestamp(subgroup_window.get(field_name)) != _timestamp(
                aggregate_window.get(field_name)
            ):
                _invalid(f"slice {window_name}.{field_name} must match the aggregate")
        aggregate_samples = aggregate_window.get("sample_count")
        subgroup_samples = subgroup_window.get("sample_count")
        if (
            isinstance(aggregate_samples, bool)
            or not isinstance(aggregate_samples, int)
            or isinstance(subgroup_samples, bool)
            or not isinstance(subgroup_samples, int)
            or subgroup_samples > aggregate_samples
        ):
            _invalid(f"slice {window_name} sample count exceeds the aggregate")
    if _feature_configuration(subgroup) != _feature_configuration(aggregate):
        _invalid("slice feature configuration must match the aggregate")
    aggregate_prediction = aggregate.get("prediction")
    subgroup_prediction = subgroup.get("prediction")
    if not isinstance(aggregate_prediction, dict) or not isinstance(
        subgroup_prediction, dict
    ):
        _invalid("slice predictions must be objects")
    prediction_shape = (
        "bin_edges_sha256",
        "reference_counts",
        "current_counts",
    )
    if subgroup_prediction.get(prediction_shape[0]) != aggregate_prediction.get(
        prediction_shape[0]
    ) or any(
        not isinstance(subgroup_prediction.get(name), list)
        or not isinstance(aggregate_prediction.get(name), list)
        or len(subgroup_prediction[name]) != len(aggregate_prediction[name])
        for name in prediction_shape[1:]
    ):
        _invalid("slice prediction bins must match the aggregate")


def audit_subgroup_drift(
    document: dict[str, Any],
    *,
    evaluated_at: datetime,
    policy: SubgroupDriftPolicy,
) -> SubgroupDriftReport:
    """Audit aggregate drift and every member of a governed subgroup partition."""
    policy.validate()
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        _invalid("evaluated_at must be timezone-aware")
    if not isinstance(document, dict) or set(document) != _DOCUMENT_FIELDS:
        _invalid("partition artifact must contain exactly the documented fields")
    if document["schema_version"] != SCHEMA_VERSION:
        _invalid("unsupported schema_version")
    partition_id = _identifier("partition_id", document["partition_id"])
    aggregate = document["aggregate"]
    if not isinstance(aggregate, dict):
        _invalid("aggregate must be an attribution-drift artifact")
    aggregate_report = audit_attribution_drift(
        aggregate, evaluated_at=evaluated_at, policy=policy.drift_policy
    )

    slice_values = document["slices"]
    if not isinstance(slice_values, list) or len(slice_values) > policy.max_slices:
        _invalid("slices must be an array within max_slices")
    parsed: dict[str, dict[str, Any]] = {}
    for value in slice_values:
        if not isinstance(value, dict) or set(value) != _SLICE_FIELDS:
            _invalid("each slice must contain exactly slice_id and artifact")
        slice_id = _identifier("slice_id", value["slice_id"])
        if slice_id in parsed:
            _invalid("slice_id values must be unique")
        artifact = value["artifact"]
        if not isinstance(artifact, dict):
            _invalid("slice artifact must be an object")
        _assert_compatible(aggregate, artifact)
        parsed[slice_id] = artifact

    governed = set(policy.governed_slice_ids)
    findings: list[Finding] = []
    for slice_id in sorted(governed - parsed.keys()):
        findings.append(
            Finding("MISSING_GOVERNED_SLICE", {"slice_sha256": _digest(slice_id)})
        )
    for slice_id in sorted(parsed.keys() - governed):
        findings.append(
            Finding("UNDECLARED_SLICE", {"slice_sha256": _digest(slice_id)})
        )
    if not aggregate_report.accepted:
        findings.append(Finding("AGGREGATE_DRIFT", {}))

    subgroup_policy = replace(policy.drift_policy, min_samples_per_window=1)
    results: list[SubgroupResult] = []
    reference_total = 0
    current_total = 0
    for slice_id in sorted(parsed):
        artifact = parsed[slice_id]
        reference_count = artifact["reference_window"]["sample_count"]
        current_count = artifact["current_window"]["sample_count"]
        reference_total += reference_count
        current_total += current_count
        slice_sha256 = _digest(slice_id)
        if min(reference_count, current_count) < policy.min_samples_per_slice:
            findings.append(
                Finding(
                    "INSUFFICIENT_SLICE_SUPPORT",
                    {
                        "slice_sha256": slice_sha256,
                        "reference_sample_count": reference_count,
                        "current_sample_count": current_count,
                        "minimum": policy.min_samples_per_slice,
                    },
                )
            )
        report = audit_attribution_drift(
            artifact, evaluated_at=evaluated_at, policy=subgroup_policy
        )
        if not report.accepted:
            findings.append(Finding("SUBGROUP_DRIFT", {"slice_sha256": slice_sha256}))
        results.append(
            SubgroupResult(slice_sha256, reference_count, current_count, report)
        )

    aggregate_reference = aggregate["reference_window"]["sample_count"]
    aggregate_current = aggregate["current_window"]["sample_count"]
    reference_coverage = reference_total / aggregate_reference
    current_coverage = current_total / aggregate_current
    if reference_total > aggregate_reference or current_total > aggregate_current:
        findings.append(
            Finding(
                "OVERLAPPING_OR_INVALID_PARTITION",
                {
                    "reference_coverage": reference_coverage,
                    "current_coverage": current_coverage,
                },
            )
        )
    elif policy.require_complete_partition and (
        reference_total != aggregate_reference or current_total != aggregate_current
    ):
        findings.append(
            Finding(
                "INCOMPLETE_PARTITION",
                {
                    "reference_coverage": reference_coverage,
                    "current_coverage": current_coverage,
                },
            )
        )

    results.sort(key=lambda item: item.slice_sha256)
    findings.sort(
        key=lambda item: (item.code, json.dumps(item.evidence, sort_keys=True))
    )
    canonical = {
        "schema_version": SCHEMA_VERSION,
        "partition_id": partition_id,
        "aggregate_artifact_sha256": aggregate_report.artifact_sha256,
        "slices": [
            {
                "slice_sha256": result.slice_sha256,
                "artifact_sha256": result.report.artifact_sha256,
            }
            for result in results
        ],
    }
    policy_payload = asdict(policy)
    return SubgroupDriftReport(
        accepted=not any(finding.blocking for finding in findings),
        artifact_sha256=_digest(
            json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        ),
        partition_sha256=_digest(partition_id),
        policy_sha256=_digest(
            json.dumps(policy_payload, sort_keys=True, separators=(",", ":"))
        ),
        evaluated_at=evaluated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        metrics={
            "governed_slice_count": len(governed),
            "observed_slice_count": len(parsed),
            "eligible_slice_count": sum(
                min(item.reference_sample_count, item.current_sample_count)
                >= policy.min_samples_per_slice
                for item in results
            ),
            "drifting_slice_count": sum(not item.report.accepted for item in results),
            "reference_coverage": reference_coverage,
            "current_coverage": current_coverage,
        },
        aggregate_report=aggregate_report,
        subgroup_results=tuple(results),
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
    parser.add_argument("--governed-slice", action="append", required=True)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        policy = SubgroupDriftPolicy(
            governed_slice_ids=tuple(args.governed_slice),
            min_samples_per_slice=args.min_samples,
            require_complete_partition=not args.allow_partial,
        )
        document = load_artifact(
            args.artifact.read_bytes(), max_bytes=policy.max_input_bytes
        )
        report = audit_subgroup_drift(
            document, evaluated_at=datetime.now(UTC), policy=policy
        )
        payload = json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":"))
        exit_code = 0 if report.accepted else 2
    except (OSError, ValueError):
        payload = json.dumps(
            {
                "schema_version": REPORT_VERSION,
                "accepted": False,
                "error_code": "MALFORMED_PARTITION_ARTIFACT",
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
