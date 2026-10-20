from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.explain import local_perturbation_attribution
from creditxai.explanation_contract import (
    ContractPolicy,
    ExplanationContractError,
    audit_explanation_contract,
    bind_explanation,
    build_credit_feature_contract,
    parse_artifact,
    sign_promotion_receipt,
)
from creditxai.model import CreditRiskModel

MODEL_DIGEST = "a" * 64
NOW = datetime(2026, 10, 1, 9, 30, tzinfo=UTC)
PROMOTION_KEY = b"promotion-key-material-must-stay-private"
PROMOTION_KEY_ID = "credit-prod-2026-10"
PROMOTION_AUDIENCE = "credit-xai-production"


@pytest.fixture(scope="module")
def valid_payload() -> dict:
    frame = synthetic_credit_data(rows=2200, seed=91)
    model, _ = CreditRiskModel.fit(frame)
    row = frame.drop(columns="default").iloc[[123]][FEATURES]
    contributions = local_perturbation_attribution(model, row)
    contract = build_credit_feature_contract(model, model_digest=MODEL_DIGEST)
    artifact = bind_explanation(contract, contributions, generated_at=NOW)
    return artifact.to_dict()


@pytest.fixture
def signed_payload(valid_payload: dict) -> dict:
    parsed = parse_artifact(valid_payload)
    receipt = sign_promotion_receipt(
        parsed.contract,
        key_id=PROMOTION_KEY_ID,
        secret=PROMOTION_KEY,
        audience=PROMOTION_AUDIENCE,
        issued_at=NOW - timedelta(minutes=5),
        expires_at=NOW + timedelta(days=1),
    )
    payload = copy.deepcopy(valid_payload)
    payload["promotion"] = receipt.to_dict()
    return payload


def signed_policy(**overrides) -> ContractPolicy:
    values = {
        "require_signed_promotion": True,
        "required_promotion_audience": PROMOTION_AUDIENCE,
    }
    values.update(overrides)
    return ContractPolicy(**values)


def test_real_fitted_pipeline_contract_is_accepted(valid_payload: dict) -> None:
    report = audit_explanation_contract(valid_payload, now=NOW + timedelta(seconds=5))

    assert report.accepted
    assert report.reasons == ()
    assert report.raw_feature_count == len(FEATURES)
    assert report.attribution_count == len(FEATURES)
    assert report.verified_mapping_count == len(FEATURES)
    assert report.transformed_feature_count > len(FEATURES)


def test_builder_captures_raw_and_expanded_feature_order(valid_payload: dict) -> None:
    contract = valid_payload["contract"]

    assert [row["name"] for row in contract["raw_features"]] == FEATURES
    assert [row["position"] for row in contract["raw_features"]] == list(
        range(len(FEATURES))
    )
    assert contract["transformed_feature_order"][:2] == [
        "numeric__income",
        "numeric__debt_ratio",
    ]
    employment = next(
        row for row in contract["raw_features"] if row["name"] == "employment"
    )
    assert len(employment["transformed_outputs"]) == 4


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda value: value["explanation"].update(model_digest="b" * 64),
            "model_digest_mismatch",
        ),
        (
            lambda value: value["explanation"].update(pipeline_digest="b" * 64),
            "pipeline_digest_mismatch",
        ),
        (
            lambda value: value["explanation"].update(contract_digest="b" * 64),
            "contract_digest_mismatch",
        ),
        (
            lambda value: value["explanation"]["raw_feature_order"].reverse(),
            "raw_feature_order_mismatch",
        ),
        (
            lambda value: value["explanation"]["transformed_feature_order"].reverse(),
            "transformed_feature_order_mismatch",
        ),
        (
            lambda value: value["explanation"]["attributions"][0].update(
                raw_position=7
            ),
            "attribution_position_mismatch",
        ),
        (
            lambda value: value["explanation"]["attributions"][0].update(
                direction="decreases_risk"
                if value["explanation"]["attributions"][0]["attribution"] > 0
                else "increases_risk"
            ),
            "attribution_direction_mismatch",
        ),
        (
            lambda value: value["explanation"]["attributions"].pop(),
            "attribution_coverage_mismatch",
        ),
        (
            lambda value: value["explanation"].update(method="unreviewed_explainer"),
            "unapproved_attribution_method",
        ),
    ],
)
def test_semantic_binding_mismatches_are_policy_rejections(
    valid_payload: dict,
    mutate,
    reason: str,
) -> None:
    payload = copy.deepcopy(valid_payload)
    mutate(payload)

    report = audit_explanation_contract(payload, now=NOW)

    assert not report.accepted
    assert reason in report.reasons


def test_pipeline_state_tampering_is_detected(valid_payload: dict) -> None:
    payload = copy.deepcopy(valid_payload)
    payload["contract"]["raw_features"][0]["transform_state_digest"] = "c" * 64

    report = audit_explanation_contract(payload, now=NOW)

    assert "contract_pipeline_digest_mismatch" in report.reasons
    assert "contract_digest_mismatch" in report.reasons


def test_unknown_and_duplicate_attributions_fail_closed(valid_payload: dict) -> None:
    unknown = copy.deepcopy(valid_payload)
    unknown["explanation"]["attributions"][0]["feature"] = "unknown_feature"
    duplicate = copy.deepcopy(valid_payload)
    duplicate["explanation"]["attributions"][1]["feature"] = duplicate["explanation"][
        "attributions"
    ][0]["feature"]

    unknown_report = audit_explanation_contract(unknown, now=NOW)
    duplicate_report = audit_explanation_contract(duplicate, now=NOW)

    assert "unknown_attribution_feature" in unknown_report.reasons
    assert "duplicate_attribution_feature" in duplicate_report.reasons
    assert "attribution_coverage_mismatch" in duplicate_report.reasons


def test_stale_and_future_evidence_are_rejected(valid_payload: dict) -> None:
    stale = audit_explanation_contract(
        valid_payload,
        now=NOW + timedelta(seconds=101),
        policy=ContractPolicy(max_age_seconds=100),
    )
    future = audit_explanation_contract(
        valid_payload,
        now=NOW - timedelta(seconds=61),
        policy=ContractPolicy(max_future_skew_seconds=60),
    )

    assert stale.reasons == ("stale_explanation",)
    assert future.reasons == ("future_generated_at",)


def test_signed_promotion_is_verified_without_exposing_key_id(
    signed_payload: dict,
) -> None:
    report = audit_explanation_contract(
        signed_payload,
        now=NOW,
        policy=signed_policy(),
        trusted_promotion_keys={PROMOTION_KEY_ID: PROMOTION_KEY},
    )

    assert report.accepted
    assert report.promotion_verified
    assert report.promotion_key_reference is not None
    assert PROMOTION_KEY_ID not in json.dumps(report.to_dict())


def test_required_promotion_receipt_is_fail_closed(valid_payload: dict) -> None:
    report = audit_explanation_contract(
        valid_payload,
        now=NOW,
        policy=signed_policy(),
        trusted_promotion_keys={PROMOTION_KEY_ID: PROMOTION_KEY},
    )

    assert report.reasons == ("promotion_receipt_missing",)
    assert not report.promotion_verified


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda value: value["promotion"].update(signature="0" * 64),
            "promotion_signature_invalid",
        ),
        (
            lambda value: value["promotion"].update(audience="credit-xai-staging"),
            "promotion_audience_mismatch",
        ),
        (
            lambda value: value["promotion"].update(model_digest="b" * 64),
            "promotion_model_digest_mismatch",
        ),
    ],
)
def test_tampered_promotion_is_rejected(
    signed_payload: dict,
    mutate,
    reason: str,
) -> None:
    payload = copy.deepcopy(signed_payload)
    mutate(payload)

    report = audit_explanation_contract(
        payload,
        now=NOW,
        policy=signed_policy(),
        trusted_promotion_keys={PROMOTION_KEY_ID: PROMOTION_KEY},
    )

    assert not report.accepted
    assert reason in report.reasons


def test_unknown_signing_key_is_rejected(signed_payload: dict) -> None:
    report = audit_explanation_contract(
        signed_payload,
        now=NOW,
        policy=signed_policy(),
        trusted_promotion_keys={"replacement-key": b"r" * 32},
    )

    assert "promotion_key_untrusted" in report.reasons


@pytest.mark.parametrize(
    ("issued_at", "expires_at", "now", "policy", "reason"),
    [
        (
            NOW - timedelta(days=2),
            NOW - timedelta(seconds=1),
            NOW,
            signed_policy(),
            "promotion_receipt_expired",
        ),
        (
            NOW + timedelta(seconds=61),
            NOW + timedelta(days=1),
            NOW,
            signed_policy(),
            "promotion_receipt_future",
        ),
        (
            NOW - timedelta(seconds=1),
            NOW + timedelta(seconds=101),
            NOW,
            signed_policy(max_promotion_lifetime_seconds=100),
            "promotion_validity_too_long",
        ),
    ],
)
def test_promotion_time_bounds_are_enforced(
    valid_payload: dict,
    issued_at: datetime,
    expires_at: datetime,
    now: datetime,
    policy: ContractPolicy,
    reason: str,
) -> None:
    parsed = parse_artifact(valid_payload)
    receipt = sign_promotion_receipt(
        parsed.contract,
        key_id=PROMOTION_KEY_ID,
        secret=PROMOTION_KEY,
        audience=PROMOTION_AUDIENCE,
        issued_at=issued_at,
        expires_at=expires_at,
    )
    payload = copy.deepcopy(valid_payload)
    payload["promotion"] = receipt.to_dict()

    report = audit_explanation_contract(
        payload,
        now=now,
        policy=policy,
        trusted_promotion_keys={PROMOTION_KEY_ID: PROMOTION_KEY},
    )

    assert reason in report.reasons


def test_explanation_must_not_predate_promotion(valid_payload: dict) -> None:
    parsed = parse_artifact(valid_payload)
    receipt = sign_promotion_receipt(
        parsed.contract,
        key_id=PROMOTION_KEY_ID,
        secret=PROMOTION_KEY,
        audience=PROMOTION_AUDIENCE,
        issued_at=NOW + timedelta(seconds=30),
        expires_at=NOW + timedelta(days=1),
    )
    payload = copy.deepcopy(valid_payload)
    payload["promotion"] = receipt.to_dict()

    report = audit_explanation_contract(
        payload,
        now=NOW + timedelta(minutes=1),
        policy=signed_policy(max_future_skew_seconds=0),
        trusted_promotion_keys={PROMOTION_KEY_ID: PROMOTION_KEY},
    )

    assert "explanation_predates_promotion" in report.reasons


@pytest.mark.parametrize("secret", [b"short", b"x" * 4097])
def test_signing_rejects_unsafe_key_sizes(valid_payload: dict, secret: bytes) -> None:
    contract = parse_artifact(valid_payload).contract

    with pytest.raises(ExplanationContractError, match="32 to 4096 bytes"):
        sign_promotion_receipt(
            contract,
            key_id=PROMOTION_KEY_ID,
            secret=secret,
            audience=PROMOTION_AUDIENCE,
            issued_at=NOW,
            expires_at=NOW + timedelta(days=1),
        )


def test_attribution_order_does_not_change_evidence(valid_payload: dict) -> None:
    reversed_payload = copy.deepcopy(valid_payload)
    reversed_payload["explanation"]["attributions"].reverse()

    forward = audit_explanation_contract(valid_payload, now=NOW)
    reverse = audit_explanation_contract(reversed_payload, now=NOW)

    assert forward.to_dict() == reverse.to_dict()


def test_report_does_not_expose_values_or_feature_names(valid_payload: dict) -> None:
    report = audit_explanation_contract(valid_payload, now=NOW)
    rendered = json.dumps(report.to_dict())

    assert "income" not in rendered
    assert "employment" not in rendered
    attribution_value = str(
        valid_payload["explanation"]["attributions"][0]["attribution"]
    )
    assert attribution_value not in rendered


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(extra=True), "missing or unexpected"),
        (
            lambda value: value["contract"]["raw_features"].append(
                value["contract"]["raw_features"][0]
            ),
            "raw feature names",
        ),
        (
            lambda value: value["contract"]["raw_features"][0].update(position=True),
            "non-negative integer",
        ),
        (
            lambda value: value["explanation"]["attributions"][0].update(
                attribution=float("nan")
            ),
            "finite",
        ),
        (lambda value: value.update(generated_at="2026-10-01"), "timezone"),
    ],
)
def test_malformed_contracts_raise(valid_payload: dict, mutate, message: str) -> None:
    payload = copy.deepcopy(valid_payload)
    mutate(payload)

    with pytest.raises(ExplanationContractError, match=message):
        audit_explanation_contract(payload, now=NOW)


def test_transformed_output_must_have_single_raw_owner(valid_payload: dict) -> None:
    payload = copy.deepcopy(valid_payload)
    duplicate_output = payload["contract"]["raw_features"][0]["transformed_outputs"][0]
    payload["contract"]["raw_features"][1]["transformed_outputs"] = [duplicate_output]

    with pytest.raises(ExplanationContractError, match="one raw feature"):
        parse_artifact(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_age_seconds": -1},
        {"max_future_skew_seconds": True},
        {"max_promotion_lifetime_seconds": True},
        {"allowed_methods": ()},
        {"allowed_methods": ("same", "same")},
        {"require_signed_promotion": True},
        {"required_promotion_audience": "invalid audience"},
    ],
)
def test_invalid_policies_raise(kwargs: dict) -> None:
    with pytest.raises(ExplanationContractError):
        ContractPolicy(**kwargs)


def run_cli(
    tmp_path: Path,
    raw: str,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(raw, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "creditxai.explanation_contract",
            str(artifact_path),
            "--now",
            "2026-10-01T09:30:05Z",
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_cli_accepts_and_atomically_writes_report(
    valid_payload: dict, tmp_path: Path
) -> None:
    output = tmp_path / "reports" / "lineage.json"
    completed = run_cli(tmp_path, json.dumps(valid_payload), "--output", str(output))

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert json.loads(output.read_text())["accepted"] is True


def test_cli_returns_three_for_policy_rejection(
    valid_payload: dict, tmp_path: Path
) -> None:
    payload = copy.deepcopy(valid_payload)
    payload["explanation"]["model_digest"] = "b" * 64
    completed = run_cli(tmp_path, json.dumps(payload))

    assert completed.returncode == 3
    assert "model_digest_mismatch" in json.loads(completed.stdout)["reasons"]


def test_cli_verifies_key_loaded_from_environment(
    signed_payload: dict,
    tmp_path: Path,
) -> None:
    environment = {**os.environ, "CREDIT_XAI_PROMOTION_KEY": PROMOTION_KEY.decode()}
    completed = run_cli(
        tmp_path,
        json.dumps(signed_payload),
        "--require-signed-promotion",
        "--promotion-audience",
        PROMOTION_AUDIENCE,
        "--trusted-promotion-key-env",
        f"{PROMOTION_KEY_ID}=CREDIT_XAI_PROMOTION_KEY",
        environment=environment,
    )

    report = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert report["promotion_verified"] is True
    assert PROMOTION_KEY.decode() not in completed.stdout


def test_cli_missing_key_environment_variable_is_malformed(
    signed_payload: dict,
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment.pop("MISSING_PROMOTION_KEY", None)
    completed = run_cli(
        tmp_path,
        json.dumps(signed_payload),
        "--require-signed-promotion",
        "--promotion-audience",
        PROMOTION_AUDIENCE,
        "--trusted-promotion-key-env",
        f"{PROMOTION_KEY_ID}=MISSING_PROMOTION_KEY",
        environment=environment,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["error"] == "malformed_artifact"


@pytest.mark.parametrize(
    "raw",
    [
        '{"generated_at":"2026-10-01T09:30:00Z","contract":{},"contract":{}}',
        '{"generated_at":NaN,"contract":{},"explanation":{}}',
        "not-json",
    ],
)
def test_cli_returns_two_for_malformed_json(tmp_path: Path, raw: str) -> None:
    completed = run_cli(tmp_path, raw)

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["error"] == "malformed_artifact"
