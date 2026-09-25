"""Fail-closed release audit for attribution sensitivity to reference baselines.

The module consumes explanation evidence produced elsewhere.  It deliberately
does not know how the model or attribution method works; that separation lets a
release gate independently check several baseline-relative explanations.
"""

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
from itertools import combinations
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
MAX_INPUT_BYTES = 256 * 1024
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class EvidenceError(ValueError):
    """Raised when explanation evidence cannot be safely audited."""


@dataclass(frozen=True)
class AuditPolicy:
    """Governed thresholds and resource bounds for the audit."""

    policy_id: str = "baseline-sensitivity-v1"
    min_cases: int = 1
    min_baselines_per_case: int = 3
    min_features: int = 3
    top_k: int = 3
    min_output_delta: float = 0.01
    max_completeness_error: float = 1e-6
    min_pairwise_cosine: float = 0.80
    min_pairwise_top_k_overlap: float = 2 / 3
    min_pairwise_sign_agreement: float = 0.75
    max_baseline_probability_span: float = 0.35
    max_failed_case_fraction: float = 0.0
    max_cases: int = 1_000
    max_baselines_per_case: int = 20
    max_features: int = 256
    max_pair_comparisons: int = 100_000

    def validate(self) -> None:
        _identifier(self.policy_id, "policy_id")
        for name in (
            "min_cases",
            "min_baselines_per_case",
            "min_features",
            "top_k",
            "max_cases",
            "max_baselines_per_case",
            "max_features",
            "max_pair_comparisons",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise EvidenceError(f"invalid policy field: {name}")
        for name in (
            "min_output_delta",
            "max_completeness_error",
            "min_pairwise_cosine",
            "min_pairwise_top_k_overlap",
            "min_pairwise_sign_agreement",
            "max_baseline_probability_span",
            "max_failed_case_fraction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise EvidenceError(f"invalid policy field: {name}")
        if self.min_cases > self.max_cases:
            raise EvidenceError("min_cases exceeds max_cases")
        if self.min_baselines_per_case > self.max_baselines_per_case:
            raise EvidenceError("minimum baseline count exceeds its budget")
        if self.min_features > self.max_features or self.top_k > self.max_features:
            raise EvidenceError("feature requirement exceeds its budget")
        if self.min_output_delta < 0 or self.max_completeness_error < 0:
            raise EvidenceError("error and contrast thresholds must be non-negative")
        if not -1 <= self.min_pairwise_cosine <= 1:
            raise EvidenceError("min_pairwise_cosine must be in [-1, 1]")
        for name in (
            "min_pairwise_top_k_overlap",
            "min_pairwise_sign_agreement",
            "max_baseline_probability_span",
            "max_failed_case_fraction",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise EvidenceError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class PairMetrics:
    baseline_ids: tuple[str, str]
    cosine_similarity: float
    top_k_overlap: float
    sign_agreement: float


@dataclass(frozen=True)
class CaseResult:
    case_id_hash: str
    accepted: bool
    reason_codes: tuple[str, ...]
    baseline_count: int
    feature_count: int
    baseline_probability_span: float
    max_completeness_error: float
    min_output_delta: float
    min_pairwise_cosine: float
    min_pairwise_top_k_overlap: float
    min_pairwise_sign_agreement: float
    worst_pair_hash: str


@dataclass(frozen=True)
class AuditReport:
    schema_version: str
    accepted: bool
    reason_codes: tuple[str, ...]
    policy: dict[str, Any]
    method_id: str
    model_id: str
    case_count: int
    failed_case_count: int
    failed_case_fraction: float
    pair_comparison_count: int
    cases: tuple[CaseResult, ...]
    evidence_sha256: str
    report_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise EvidenceError(f"invalid identifier: {field}")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{field} must be finite")
    return result


def _probability(value: Any, field: str) -> float:
    result = _finite_number(value, field)
    if not 0 <= result <= 1:
        raise EvidenceError(f"{field} must be in [0, 1]")
    return result


def _object(value: Any, field: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise EvidenceError(f"{field} has an invalid object shape")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvidenceError(f"{field} must be a list")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _redacted_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("duplicate JSON object key")
        result[key] = value
    return result


def load_evidence(
    path: str | Path, *, max_bytes: int = MAX_INPUT_BYTES
) -> dict[str, Any]:
    """Load bounded strict JSON, rejecting duplicates and non-finite constants."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise EvidenceError("max_bytes must be a positive integer")
    source = Path(path)
    if source.stat().st_size > max_bytes:
        raise EvidenceError("input exceeds byte budget")
    with source.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise EvidenceError("input exceeds byte budget")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError("input is not UTF-8") from exc
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EvidenceError(f"non-finite JSON constant: {token}")
            ),
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceError("invalid JSON evidence") from exc
    if not isinstance(value, dict):
        raise EvidenceError("evidence root must be an object")
    return value


def _parse_attributions(
    value: Any,
    *,
    field: str,
    policy: AuditPolicy,
) -> dict[str, float]:
    records = _list(value, field)
    if not policy.min_features <= len(records) <= policy.max_features:
        raise EvidenceError(f"{field} violates feature-count bounds")
    result: dict[str, float] = {}
    for index, raw in enumerate(records):
        item = _object(raw, f"{field}[{index}]", {"feature_id", "value"})
        feature_id = _identifier(item["feature_id"], f"{field}[{index}].feature_id")
        if feature_id in result:
            raise EvidenceError(f"duplicate feature_id in {field}")
        result[feature_id] = _finite_number(item["value"], f"{field}[{index}].value")
    return result


def _top_k(values: dict[str, float], k: int) -> tuple[str, ...]:
    return tuple(
        feature
        for feature, _ in sorted(
            values.items(), key=lambda item: (-abs(item[1]), item[0])
        )[:k]
    )


def _pair_metrics(
    left_id: str,
    left: dict[str, float],
    right_id: str,
    right: dict[str, float],
    top_k: int,
) -> PairMetrics:
    features = sorted(left)
    dot = sum(left[name] * right[name] for name in features)
    left_norm = math.sqrt(sum(left[name] ** 2 for name in features))
    right_norm = math.sqrt(sum(right[name] ** 2 for name in features))
    if left_norm == 0 or right_norm == 0:
        raise EvidenceError("attribution vectors must not be zero")
    cosine = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    effective_k = min(top_k, len(features))
    left_top = set(_top_k(left, effective_k))
    right_top = set(_top_k(right, effective_k))
    shared = left_top & right_top
    overlap = len(shared) / effective_k
    if shared:
        sign_agreement = sum(
            (left[name] > 0) == (right[name] > 0) for name in shared
        ) / len(shared)
    else:
        sign_agreement = 0.0
    return PairMetrics(
        baseline_ids=(left_id, right_id),
        cosine_similarity=cosine,
        top_k_overlap=overlap,
        sign_agreement=sign_agreement,
    )


def _audit_case(raw: Any, policy: AuditPolicy) -> tuple[CaseResult, int]:
    case = _object(raw, "case", {"case_id", "observed_probability", "baselines"})
    case_id = _identifier(case["case_id"], "case_id")
    observed = _probability(case["observed_probability"], "observed_probability")
    baselines = _list(case["baselines"], "baselines")
    if (
        not policy.min_baselines_per_case
        <= len(baselines)
        <= policy.max_baselines_per_case
    ):
        raise EvidenceError("case violates baseline-count bounds")

    parsed: list[tuple[str, float, dict[str, float]]] = []
    baseline_ids: set[str] = set()
    expected_features: set[str] | None = None
    completeness_errors: list[float] = []
    output_deltas: list[float] = []
    for index, raw_baseline in enumerate(baselines):
        baseline = _object(
            raw_baseline,
            f"baselines[{index}]",
            {"baseline_id", "baseline_probability", "attributions"},
        )
        baseline_id = _identifier(
            baseline["baseline_id"], f"baselines[{index}].baseline_id"
        )
        if baseline_id in baseline_ids:
            raise EvidenceError("duplicate baseline_id")
        baseline_ids.add(baseline_id)
        probability = _probability(
            baseline["baseline_probability"], f"baselines[{index}].baseline_probability"
        )
        attributions = _parse_attributions(
            baseline["attributions"],
            field=f"baselines[{index}].attributions",
            policy=policy,
        )
        feature_ids = set(attributions)
        if expected_features is None:
            expected_features = feature_ids
        elif feature_ids != expected_features:
            raise EvidenceError("feature IDs are not aligned across baselines")
        delta = observed - probability
        output_deltas.append(abs(delta))
        completeness_errors.append(abs(sum(attributions.values()) - delta))
        parsed.append((baseline_id, probability, attributions))

    pair_count = len(parsed) * (len(parsed) - 1) // 2
    metrics = [
        _pair_metrics(left[0], left[2], right[0], right[2], policy.top_k)
        for left, right in combinations(parsed, 2)
    ]
    worst = min(
        metrics,
        key=lambda item: (
            item.cosine_similarity,
            item.top_k_overlap,
            item.sign_agreement,
            item.baseline_ids,
        ),
    )
    probability_span = max(item[1] for item in parsed) - min(item[1] for item in parsed)
    max_error = max(completeness_errors)
    min_delta = min(output_deltas)
    min_cosine = min(item.cosine_similarity for item in metrics)
    min_overlap = min(item.top_k_overlap for item in metrics)
    min_sign = min(item.sign_agreement for item in metrics)
    reasons: list[str] = []
    if min_delta < policy.min_output_delta:
        reasons.append("WEAK_BASELINE_CONTRAST")
    if max_error > policy.max_completeness_error:
        reasons.append("COMPLETENESS_MISMATCH")
    if probability_span > policy.max_baseline_probability_span:
        reasons.append("BASELINE_OUTPUT_SPAN_EXCEEDED")
    if min_cosine < policy.min_pairwise_cosine:
        reasons.append("ATTRIBUTION_DIRECTION_UNSTABLE")
    if min_overlap < policy.min_pairwise_top_k_overlap:
        reasons.append("TOP_K_FEATURES_UNSTABLE")
    if min_sign < policy.min_pairwise_sign_agreement:
        reasons.append("SALIENT_FEATURE_SIGNS_UNSTABLE")
    feature_count = len(expected_features or ())
    return (
        CaseResult(
            case_id_hash=_redacted_id(case_id),
            accepted=not reasons,
            reason_codes=tuple(reasons),
            baseline_count=len(parsed),
            feature_count=feature_count,
            baseline_probability_span=probability_span,
            max_completeness_error=max_error,
            min_output_delta=min_delta,
            min_pairwise_cosine=min_cosine,
            min_pairwise_top_k_overlap=min_overlap,
            min_pairwise_sign_agreement=min_sign,
            worst_pair_hash=_digest(sorted(worst.baseline_ids)),
        ),
        pair_count,
    )


def audit_baseline_sensitivity(
    evidence: dict[str, Any], policy: AuditPolicy
) -> AuditReport:
    """Audit a batch and return deterministic, content-redacted release evidence."""

    policy.validate()
    root = _object(
        evidence, "evidence", {"schema_version", "method_id", "model_id", "cases"}
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError("unsupported schema_version")
    method_id = _identifier(root["method_id"], "method_id")
    model_id = _identifier(root["model_id"], "model_id")
    cases = _list(root["cases"], "cases")
    if not policy.min_cases <= len(cases) <= policy.max_cases:
        raise EvidenceError("evidence violates case-count bounds")

    results: list[CaseResult] = []
    seen_case_hashes: set[str] = set()
    pair_count = 0
    for raw_case in cases:
        result, comparisons = _audit_case(raw_case, policy)
        if result.case_id_hash in seen_case_hashes:
            raise EvidenceError("duplicate case_id")
        seen_case_hashes.add(result.case_id_hash)
        pair_count += comparisons
        if pair_count > policy.max_pair_comparisons:
            raise EvidenceError("pair-comparison budget exceeded")
        results.append(result)

    results.sort(key=lambda item: item.case_id_hash)
    failed = sum(not result.accepted for result in results)
    failed_fraction = failed / len(results)
    case_reason_codes = sorted(
        {code for result in results for code in result.reason_codes}
    )
    reason_codes: list[str] = []
    if failed_fraction > policy.max_failed_case_fraction:
        reason_codes.extend(case_reason_codes)
        reason_codes.append("FAILED_CASE_FRACTION_EXCEEDED")
    evidence_digest = _digest(evidence)
    report_core = {
        "schema_version": SCHEMA_VERSION,
        "accepted": not reason_codes,
        "reason_codes": tuple(reason_codes),
        "policy": asdict(policy),
        "method_id": method_id,
        "model_id": model_id,
        "case_count": len(results),
        "failed_case_count": failed,
        "failed_case_fraction": failed_fraction,
        "pair_comparison_count": pair_count,
        "cases": [asdict(result) for result in results],
        "evidence_sha256": evidence_digest,
    }
    return AuditReport(
        schema_version=SCHEMA_VERSION,
        accepted=not reason_codes,
        reason_codes=tuple(reason_codes),
        policy=asdict(policy),
        method_id=method_id,
        model_id=model_id,
        case_count=len(results),
        failed_case_count=failed,
        failed_case_fraction=failed_fraction,
        pair_comparison_count=pair_count,
        cases=tuple(results),
        evidence_sha256=evidence_digest,
        report_sha256=_digest(report_core),
    )


def _write_json(path: Path, report: AuditReport) -> None:
    payload = (
        json.dumps(report.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_baseline_sensitivity(load_evidence(args.evidence), AuditPolicy())
    except (EvidenceError, OSError):
        print(json.dumps({"accepted": False, "error_code": "MALFORMED_EVIDENCE"}))
        return 3
    if args.output:
        try:
            _write_json(args.output, report)
        except OSError:
            print(json.dumps({"accepted": False, "error_code": "OUTPUT_ERROR"}))
            return 3
    print(json.dumps(report.as_dict(), sort_keys=True, allow_nan=False))
    return 0 if report.accepted else 2


if __name__ == "__main__":
    sys.exit(main())
