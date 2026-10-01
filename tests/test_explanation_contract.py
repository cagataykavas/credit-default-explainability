from __future__ import annotations

import copy
import json
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
)
from creditxai.model import CreditRiskModel

MODEL_DIGEST = "a" * 64
NOW = datetime(2026, 10, 1, 9, 30, tzinfo=UTC)


@pytest.fixture(scope="module")
def valid_payload() -> dict:
    frame = synthetic_credit_data(rows=2200, seed=91)
    model, _ = CreditRiskModel.fit(frame)
    row = frame.drop(columns="default").iloc[[123]][FEATURES]
    contributions = local_perturbation_attribution(model, row)
    contract = build_credit_feature_contract(model, model_digest=MODEL_DIGEST)
    artifact = bind_explanation(contract, contributions, generated_at=NOW)
    return artifact.to_dict()


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
    assert [row["position"] for row in contract["raw_features"]] == list(range(len(FEATURES)))
    assert contract["transformed_feature_order"][:2] == ["numeric__income", "numeric__debt_ratio"]
    employment = next(row for row in contract["raw_features"] if row["name"] == "employment")
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
            lambda value: value["explanation"]["attributions"][0].update(raw_position=7),
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
    attribution_value = str(valid_payload["explanation"]["attributions"][0]["attribution"])
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
            lambda value: value["explanation"]["attributions"][0].update(attribution=float("nan")),
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
        {"allowed_methods": ()},
        {"allowed_methods": ("same", "same")},
    ],
)
def test_invalid_policies_raise(kwargs: dict) -> None:
    with pytest.raises(ExplanationContractError):
        ContractPolicy(**kwargs)


def run_cli(tmp_path: Path, raw: str, *arguments: str) -> subprocess.CompletedProcess[str]:
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
    )


def test_cli_accepts_and_atomically_writes_report(valid_payload: dict, tmp_path: Path) -> None:
    output = tmp_path / "reports" / "lineage.json"
    completed = run_cli(tmp_path, json.dumps(valid_payload), "--output", str(output))

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert json.loads(output.read_text())["accepted"] is True


def test_cli_returns_three_for_policy_rejection(valid_payload: dict, tmp_path: Path) -> None:
    payload = copy.deepcopy(valid_payload)
    payload["explanation"]["model_digest"] = "b" * 64
    completed = run_cli(tmp_path, json.dumps(payload))

    assert completed.returncode == 3
    assert "model_digest_mismatch" in json.loads(completed.stdout)["reasons"]


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
