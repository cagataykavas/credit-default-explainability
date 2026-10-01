from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .data import FEATURES
from .explain import FeatureContribution
from .model import CATEGORICAL, NUMERIC, CreditRiskModel

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")
MAX_INPUT_BYTES = 1024 * 1024
MAX_RAW_FEATURES = 128
MAX_TRANSFORMED_FEATURES = 4096
MAX_ATTRIBUTIONS = 128

FEATURE_METADATA = {
    "income": ("continuous", "currency_per_year", "credit.income.v1"),
    "debt_ratio": ("continuous", "ratio_0_1", "credit.debt_ratio.v1"),
    "late_payments": ("integer", "count", "credit.late_payments.v1"),
    "utilization": ("continuous", "ratio_0_1", "credit.utilization.v1"),
    "account_age_months": ("integer", "months", "credit.account_age.v1"),
    "age": ("integer", "years", "person.age.v1"),
    "employment": ("categorical", "category", "person.employment.v1"),
    "housing": ("categorical", "category", "person.housing.v1"),
}


class ExplanationContractError(ValueError):
    """Raised when an explanation contract artifact is malformed."""


@dataclass(frozen=True)
class RawFeatureContract:
    name: str
    position: int
    dtype: str
    unit: str
    semantic_id: str
    transform_state_digest: str
    transformed_outputs: tuple[str, ...]


@dataclass(frozen=True)
class ModelFeatureContract:
    schema_version: str
    model_digest: str
    pipeline_digest: str
    raw_features: tuple[RawFeatureContract, ...]
    transformed_feature_order: tuple[str, ...]

    @property
    def contract_digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_digest": self.model_digest,
            "pipeline_digest": self.pipeline_digest,
            "raw_features": [
                {**asdict(feature), "transformed_outputs": list(feature.transformed_outputs)}
                for feature in self.raw_features
            ],
            "transformed_feature_order": list(self.transformed_feature_order),
        }


@dataclass(frozen=True)
class AttributionBinding:
    feature: str
    raw_position: int
    attribution: float
    direction: str
    observed_value_digest: str
    baseline_value_digest: str


@dataclass(frozen=True)
class BoundExplanation:
    method: str
    model_digest: str
    pipeline_digest: str
    contract_digest: str
    raw_feature_order: tuple[str, ...]
    transformed_feature_order: tuple[str, ...]
    attributions: tuple[AttributionBinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "model_digest": self.model_digest,
            "pipeline_digest": self.pipeline_digest,
            "contract_digest": self.contract_digest,
            "raw_feature_order": list(self.raw_feature_order),
            "transformed_feature_order": list(self.transformed_feature_order),
            "attributions": [asdict(attribution) for attribution in self.attributions],
        }


@dataclass(frozen=True)
class ExplanationContractArtifact:
    generated_at: datetime
    contract: ModelFeatureContract
    explanation: BoundExplanation

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": _format_timestamp(self.generated_at),
            "contract": self.contract.to_dict(),
            "explanation": self.explanation.to_dict(),
        }


@dataclass(frozen=True)
class ContractPolicy:
    max_age_seconds: int = 86_400
    max_future_skew_seconds: int = 60
    allowed_methods: tuple[str, ...] = ("one_feature_baseline_perturbation",)

    def __post_init__(self) -> None:
        for name in ("max_age_seconds", "max_future_skew_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExplanationContractError(f"{name} must be a non-negative integer")
        if not self.allowed_methods or len(set(self.allowed_methods)) != len(self.allowed_methods):
            raise ExplanationContractError("allowed_methods must contain unique values")
        for method in self.allowed_methods:
            _validate_identifier(method, "allowed_methods")


@dataclass(frozen=True)
class ContractAuditReport:
    schema_version: str
    accepted: bool
    reasons: tuple[str, ...]
    evidence_digest: str
    model_digest: str
    pipeline_digest: str
    contract_digest: str
    generated_at: str
    raw_feature_count: int
    transformed_feature_count: int
    attribution_count: int
    verified_mapping_count: int

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "reasons": list(self.reasons)}


def _canonical_json(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ExplanationContractError("artifact contains a non-canonical value") from error
    return rendered.encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _python_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_python_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_python_value(item) for item in value]
    if isinstance(value, list):
        return [_python_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _python_value(item) for key, item in value.items()}
    return value


def _value_digest(value: object) -> str:
    return _digest(_python_value(value))


def _validate_digest(value: object, path: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ExplanationContractError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _validate_identifier(value: object, path: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        raise ExplanationContractError(f"{path} must be a bounded identifier")
    return value


def _validate_position(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExplanationContractError(f"{path} must be a non-negative integer")
    return value


def _validate_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExplanationContractError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ExplanationContractError(f"{path} must be finite")
    return result


def _require_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ExplanationContractError(f"{path} must be an object")
    return value


def _require_keys(value: dict[str, object], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise ExplanationContractError(f"{path} has missing or unexpected fields")


def _require_string_array(value: object, path: str, limit: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > limit:
        raise ExplanationContractError(f"{path} must be a non-empty bounded array")
    result = tuple(_validate_identifier(item, f"{path}[]") for item in value)
    if len(set(result)) != len(result):
        raise ExplanationContractError(f"{path} must contain unique values")
    return result


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ExplanationContractError("generated_at must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        raise ExplanationContractError("generated_at must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ExplanationContractError("generated_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExplanationContractError("generated_at must include a timezone")
    return parsed.astimezone(UTC)


def _pipeline_payload(
    raw_features: tuple[RawFeatureContract, ...],
    transformed_feature_order: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "raw_features": [
            {
                "name": feature.name,
                "position": feature.position,
                "dtype": feature.dtype,
                "unit": feature.unit,
                "semantic_id": feature.semantic_id,
                "transform_state_digest": feature.transform_state_digest,
                "transformed_outputs": list(feature.transformed_outputs),
            }
            for feature in raw_features
        ],
        "transformed_feature_order": list(transformed_feature_order),
    }


def build_credit_feature_contract(
    model: CreditRiskModel,
    *,
    model_digest: str,
) -> ModelFeatureContract:
    """Extract an immutable explanation contract from the fitted sklearn pipeline."""
    _validate_digest(model_digest, "model_digest")
    preprocessor = model.pipeline.named_steps["features"]
    if not hasattr(preprocessor, "transformers_"):
        raise ExplanationContractError("credit model preprocessor must be fitted")
    transformed_order = tuple(str(name) for name in preprocessor.get_feature_names_out())
    numeric = preprocessor.named_transformers_["numeric"]
    categorical = preprocessor.named_transformers_["categorical"]
    numeric_imputer = numeric.named_steps["impute"]
    scaler = numeric.named_steps["scale"]
    categorical_imputer = categorical.named_steps["impute"]
    encoder = categorical.named_steps["encode"]

    numeric_positions = {name: index for index, name in enumerate(NUMERIC)}
    categorical_positions = {name: index for index, name in enumerate(CATEGORICAL)}
    encoded_names = tuple(str(name) for name in encoder.get_feature_names_out(CATEGORICAL))
    raw_contracts: list[RawFeatureContract] = []
    for position, name in enumerate(FEATURES):
        dtype, unit, semantic_id = FEATURE_METADATA[name]
        if name in numeric_positions:
            index = numeric_positions[name]
            state = {
                "kind": "median_then_standard_scale",
                "impute": numeric_imputer.statistics_[index],
                "mean": scaler.mean_[index],
                "scale": scaler.scale_[index],
            }
            outputs = (f"numeric__{name}",)
        else:
            index = categorical_positions[name]
            categories = tuple(str(item) for item in encoder.categories_[index])
            state = {
                "kind": "mode_then_one_hot",
                "impute": categorical_imputer.statistics_[index],
                "categories": categories,
                "handle_unknown": encoder.handle_unknown,
            }
            outputs = tuple(
                f"categorical__{output}"
                for output in encoded_names
                if output.startswith(f"{name}_")
            )
        raw_contracts.append(
            RawFeatureContract(
                name=name,
                position=position,
                dtype=dtype,
                unit=unit,
                semantic_id=semantic_id,
                transform_state_digest=_value_digest(state),
                transformed_outputs=outputs,
            )
        )
    raw_tuple = tuple(raw_contracts)
    pipeline_digest = _digest(_pipeline_payload(raw_tuple, transformed_order))
    return ModelFeatureContract(
        schema_version="1.0",
        model_digest=model_digest,
        pipeline_digest=pipeline_digest,
        raw_features=raw_tuple,
        transformed_feature_order=transformed_order,
    )


def bind_explanation(
    contract: ModelFeatureContract,
    contributions: list[FeatureContribution],
    *,
    generated_at: datetime,
    method: str = "one_feature_baseline_perturbation",
) -> ExplanationContractArtifact:
    positions = {feature.name: feature.position for feature in contract.raw_features}
    bindings = tuple(
        AttributionBinding(
            feature=contribution.feature,
            raw_position=positions.get(contribution.feature, -1),
            attribution=float(contribution.probability_delta),
            direction=contribution.direction,
            observed_value_digest=_value_digest(contribution.observed),
            baseline_value_digest=_value_digest(contribution.baseline),
        )
        for contribution in contributions
    )
    explanation = BoundExplanation(
        method=method,
        model_digest=contract.model_digest,
        pipeline_digest=contract.pipeline_digest,
        contract_digest=contract.contract_digest,
        raw_feature_order=tuple(feature.name for feature in contract.raw_features),
        transformed_feature_order=contract.transformed_feature_order,
        attributions=bindings,
    )
    return ExplanationContractArtifact(
        generated_at=generated_at,
        contract=contract,
        explanation=explanation,
    )


def _parse_raw_feature(value: object, index: int) -> RawFeatureContract:
    path = f"contract.raw_features[{index}]"
    row = _require_object(value, path)
    _require_keys(
        row,
        {
            "name",
            "position",
            "dtype",
            "unit",
            "semantic_id",
            "transform_state_digest",
            "transformed_outputs",
        },
        path,
    )
    return RawFeatureContract(
        name=_validate_identifier(row["name"], f"{path}.name"),
        position=_validate_position(row["position"], f"{path}.position"),
        dtype=_validate_identifier(row["dtype"], f"{path}.dtype"),
        unit=_validate_identifier(row["unit"], f"{path}.unit"),
        semantic_id=_validate_identifier(row["semantic_id"], f"{path}.semantic_id"),
        transform_state_digest=_validate_digest(
            row["transform_state_digest"], f"{path}.transform_state_digest"
        ),
        transformed_outputs=_require_string_array(
            row["transformed_outputs"],
            f"{path}.transformed_outputs",
            MAX_TRANSFORMED_FEATURES,
        ),
    )


def _parse_contract(value: object) -> ModelFeatureContract:
    row = _require_object(value, "contract")
    _require_keys(
        row,
        {
            "schema_version",
            "model_digest",
            "pipeline_digest",
            "raw_features",
            "transformed_feature_order",
        },
        "contract",
    )
    schema_version = row["schema_version"]
    if schema_version != "1.0":
        raise ExplanationContractError("contract.schema_version must equal 1.0")
    raw_rows = row["raw_features"]
    if not isinstance(raw_rows, list) or not raw_rows or len(raw_rows) > MAX_RAW_FEATURES:
        raise ExplanationContractError("contract.raw_features must be a non-empty bounded array")
    raw_features = tuple(_parse_raw_feature(item, index) for index, item in enumerate(raw_rows))
    names = [feature.name for feature in raw_features]
    if len(set(names)) != len(names):
        raise ExplanationContractError("contract raw feature names must be unique")
    outputs = [output for feature in raw_features for output in feature.transformed_outputs]
    if len(set(outputs)) != len(outputs):
        raise ExplanationContractError("contract transformed outputs must map to one raw feature")
    transformed_order = _require_string_array(
        row["transformed_feature_order"],
        "contract.transformed_feature_order",
        MAX_TRANSFORMED_FEATURES,
    )
    if set(outputs) != set(transformed_order):
        raise ExplanationContractError("contract transformed output coverage is inconsistent")
    return ModelFeatureContract(
        schema_version="1.0",
        model_digest=_validate_digest(row["model_digest"], "contract.model_digest"),
        pipeline_digest=_validate_digest(row["pipeline_digest"], "contract.pipeline_digest"),
        raw_features=raw_features,
        transformed_feature_order=transformed_order,
    )


def _parse_attribution(value: object, index: int) -> AttributionBinding:
    path = f"explanation.attributions[{index}]"
    row = _require_object(value, path)
    _require_keys(
        row,
        {
            "feature",
            "raw_position",
            "attribution",
            "direction",
            "observed_value_digest",
            "baseline_value_digest",
        },
        path,
    )
    direction = row["direction"]
    if direction not in {"increases_risk", "decreases_risk", "neutral"}:
        raise ExplanationContractError(f"{path}.direction is invalid")
    return AttributionBinding(
        feature=_validate_identifier(row["feature"], f"{path}.feature"),
        raw_position=_validate_position(row["raw_position"], f"{path}.raw_position"),
        attribution=_validate_float(row["attribution"], f"{path}.attribution"),
        direction=direction,
        observed_value_digest=_validate_digest(
            row["observed_value_digest"], f"{path}.observed_value_digest"
        ),
        baseline_value_digest=_validate_digest(
            row["baseline_value_digest"], f"{path}.baseline_value_digest"
        ),
    )


def _parse_explanation(value: object) -> BoundExplanation:
    row = _require_object(value, "explanation")
    _require_keys(
        row,
        {
            "method",
            "model_digest",
            "pipeline_digest",
            "contract_digest",
            "raw_feature_order",
            "transformed_feature_order",
            "attributions",
        },
        "explanation",
    )
    attribution_rows = row["attributions"]
    if (
        not isinstance(attribution_rows, list)
        or not attribution_rows
        or len(attribution_rows) > MAX_ATTRIBUTIONS
    ):
        raise ExplanationContractError("explanation.attributions must be a non-empty bounded array")
    attributions = tuple(
        _parse_attribution(item, index) for index, item in enumerate(attribution_rows)
    )
    return BoundExplanation(
        method=_validate_identifier(row["method"], "explanation.method"),
        model_digest=_validate_digest(row["model_digest"], "explanation.model_digest"),
        pipeline_digest=_validate_digest(row["pipeline_digest"], "explanation.pipeline_digest"),
        contract_digest=_validate_digest(row["contract_digest"], "explanation.contract_digest"),
        raw_feature_order=_require_string_array(
            row["raw_feature_order"], "explanation.raw_feature_order", MAX_RAW_FEATURES
        ),
        transformed_feature_order=_require_string_array(
            row["transformed_feature_order"],
            "explanation.transformed_feature_order",
            MAX_TRANSFORMED_FEATURES,
        ),
        attributions=attributions,
    )


def parse_artifact(value: object) -> ExplanationContractArtifact:
    root = _require_object(value, "artifact")
    _require_keys(root, {"generated_at", "contract", "explanation"}, "artifact")
    return ExplanationContractArtifact(
        generated_at=_parse_timestamp(root["generated_at"]),
        contract=_parse_contract(root["contract"]),
        explanation=_parse_explanation(root["explanation"]),
    )


def _expected_direction(value: float) -> str:
    if value > 0:
        return "increases_risk"
    # The repository's current explainer deliberately labels zero deltas with the
    # non-positive branch; the audit binds to that producer contract exactly.
    return "decreases_risk"


def audit_explanation_contract(
    artifact: ExplanationContractArtifact | object,
    *,
    now: datetime,
    policy: ContractPolicy | None = None,
) -> ContractAuditReport:
    parsed = (
        artifact if isinstance(artifact, ExplanationContractArtifact) else parse_artifact(artifact)
    )
    if now.tzinfo is None or now.utcoffset() is None:
        raise ExplanationContractError("now must be timezone-aware")
    effective_policy = policy or ContractPolicy()
    contract = parsed.contract
    explanation = parsed.explanation
    reasons: set[str] = set()

    age_seconds = (now.astimezone(UTC) - parsed.generated_at).total_seconds()
    if age_seconds > effective_policy.max_age_seconds:
        reasons.add("stale_explanation")
    if age_seconds < -effective_policy.max_future_skew_seconds:
        reasons.add("future_generated_at")
    if explanation.method not in effective_policy.allowed_methods:
        reasons.add("unapproved_attribution_method")

    raw_order = tuple(feature.name for feature in contract.raw_features)
    expected_pipeline_digest = _digest(
        _pipeline_payload(contract.raw_features, contract.transformed_feature_order)
    )
    if contract.pipeline_digest != expected_pipeline_digest:
        reasons.add("contract_pipeline_digest_mismatch")
    if explanation.model_digest != contract.model_digest:
        reasons.add("model_digest_mismatch")
    if explanation.pipeline_digest != contract.pipeline_digest:
        reasons.add("pipeline_digest_mismatch")
    if explanation.contract_digest != contract.contract_digest:
        reasons.add("contract_digest_mismatch")
    if explanation.raw_feature_order != raw_order:
        reasons.add("raw_feature_order_mismatch")
    if explanation.transformed_feature_order != contract.transformed_feature_order:
        reasons.add("transformed_feature_order_mismatch")

    expected_positions = {feature.name: feature.position for feature in contract.raw_features}
    if tuple(feature.position for feature in contract.raw_features) != tuple(range(len(raw_order))):
        reasons.add("contract_raw_positions_invalid")
    seen: set[str] = set()
    verified = 0
    for attribution in explanation.attributions:
        if attribution.feature in seen:
            reasons.add("duplicate_attribution_feature")
        seen.add(attribution.feature)
        expected_position = expected_positions.get(attribution.feature)
        if expected_position is None:
            reasons.add("unknown_attribution_feature")
            continue
        if attribution.raw_position != expected_position:
            reasons.add("attribution_position_mismatch")
        else:
            verified += 1
        if attribution.direction != _expected_direction(attribution.attribution):
            reasons.add("attribution_direction_mismatch")
    if seen != set(raw_order):
        reasons.add("attribution_coverage_mismatch")

    canonical_artifact = parsed.to_dict()
    canonical_artifact["explanation"]["attributions"] = sorted(
        canonical_artifact["explanation"]["attributions"],
        key=lambda row: (row["raw_position"], row["feature"]),
    )
    ordered_reasons = tuple(sorted(reasons))
    return ContractAuditReport(
        schema_version="1.0",
        accepted=not ordered_reasons,
        reasons=ordered_reasons,
        evidence_digest=_digest(canonical_artifact),
        model_digest=contract.model_digest,
        pipeline_digest=contract.pipeline_digest,
        contract_digest=contract.contract_digest,
        generated_at=_format_timestamp(parsed.generated_at),
        raw_feature_count=len(contract.raw_features),
        transformed_feature_count=len(contract.transformed_feature_order),
        attribution_count=len(explanation.attributions),
        verified_mapping_count=verified,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExplanationContractError("JSON objects must not contain duplicate fields")
        result[key] = value
    return result


def load_artifact(path: Path) -> ExplanationContractArtifact:
    try:
        if path.stat().st_size > MAX_INPUT_BYTES:
            raise ExplanationContractError("input exceeds the JSON byte limit")
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ExplanationContractError("JSON must not contain non-finite numbers")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExplanationContractError("input is not valid UTF-8 JSON") from error
    return parse_artifact(payload)


def _write_json(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    if output is None:
        print(rendered, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit model-to-explanation feature lineage")
    parser.add_argument("artifact", type=Path, help="bound explanation JSON artifact")
    parser.add_argument("--output", type=Path, help="atomically write the audit report")
    parser.add_argument("--now", required=True, help="UTC-aware audit timestamp")
    parser.add_argument("--max-age-seconds", type=int, default=86_400)
    parser.add_argument("--max-future-skew-seconds", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = audit_explanation_contract(
            load_artifact(arguments.artifact),
            now=_parse_timestamp(arguments.now),
            policy=ContractPolicy(
                max_age_seconds=arguments.max_age_seconds,
                max_future_skew_seconds=arguments.max_future_skew_seconds,
            ),
        )
        _write_json(report.to_dict(), arguments.output)
    except (ExplanationContractError, OSError) as error:
        _write_json(
            {"accepted": False, "error": "malformed_artifact", "detail": str(error)},
            arguments.output,
        )
        return 2
    return 0 if report.accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
