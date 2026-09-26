from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from creditxai.attribution_drift import (
    DriftPolicy,
    audit_attribution_drift,
    load_artifact,
    main,
)

NOW = datetime(2026, 9, 26, 13, 0, tzinfo=UTC)
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def feature(feature_id="income-secret", risk="standard", reference=None, current=None):
    return {
        "feature_id": feature_id,
        "risk": risk,
        "bin_edges_sha256": HASH_D,
        "reference_counts": reference or [50, 100, 50],
        "current_counts": current or [48, 102, 50],
        "reference_mean_abs": 0.20,
        "current_mean_abs": 0.21,
        "reference_topk_rate": 0.40,
        "current_topk_rate": 0.42,
    }


def artifact():
    return {
        "schema_version": "credit-attribution-drift/v1",
        "monitor_id": "underwriting-monitor-secret",
        "model_sha256": HASH_A,
        "explainer_sha256": HASH_B,
        "created_at": "2026-09-26T12:50:00Z",
        "reference_window": {
            "manifest_sha256": HASH_B,
            "sample_count": 200,
            "started_at": "2026-09-01T00:00:00Z",
            "ended_at": "2026-09-08T00:00:00Z",
        },
        "current_window": {
            "manifest_sha256": HASH_C,
            "sample_count": 200,
            "started_at": "2026-09-15T00:00:00Z",
            "ended_at": "2026-09-26T12:00:00Z",
        },
        "features": [
            feature(),
            feature("debt-secret", "critical", [40, 120, 40], [42, 118, 40]),
            feature("age-secret", "standard", [60, 80, 60], [58, 82, 60]),
            feature("tenure-secret", "standard", [70, 60, 70], [68, 62, 70]),
            feature("utilization-secret", "critical", [45, 110, 45], [47, 108, 45]),
        ],
        "prediction": {
            "bin_edges_sha256": HASH_A,
            "reference_counts": [50, 100, 50],
            "current_counts": [49, 101, 50],
        },
    }


def codes(report):
    return {finding.code for finding in report.findings}


def test_stable_cohorts_are_accepted_with_content_free_evidence():
    report = audit_attribution_drift(artifact(), evaluated_at=NOW)

    assert report.accepted is True
    assert report.metrics["feature_count"] == 5
    assert report.metrics["critical_feature_count"] == 2
    assert report.metrics["drifting_feature_count"] == 0
    assert report.metrics["reference_sample_count"] == 200
    encoded = json.dumps(report.as_dict())
    assert "secret" not in encoded
    assert len(report.artifact_sha256) == 64
    assert len(report.policy_sha256) == 64


def test_attribution_drift_is_detected_when_predictions_are_stable():
    value = artifact()
    value["features"][0]["current_counts"] = [180, 10, 10]
    report = audit_attribution_drift(value, evaluated_at=NOW)

    assert "FEATURE_ATTRIBUTION_DRIFT" in codes(report)
    assert report.metrics["prediction_js"] < 0.01
    assert report.metrics["drifting_feature_count"] == 1


def test_critical_feature_drift_is_zero_tolerance_by_default():
    value = artifact()
    value["features"][1]["current_topk_rate"] = 0.90
    report = audit_attribution_drift(
        value,
        evaluated_at=NOW,
        policy=DriftPolicy(max_drifting_feature_fraction=1.0),
    )

    assert report.accepted is False
    assert "CRITICAL_FEATURE_DRIFT" in codes(report)


def test_standard_feature_diagnostics_respect_aggregate_tolerance():
    value = artifact()
    value["features"][0]["current_topk_rate"] = 0.80
    report = audit_attribution_drift(
        value,
        evaluated_at=NOW,
        policy=DriftPolicy(
            max_drifting_feature_fraction=0.25,
            require_no_critical_drift=False,
        ),
    )

    assert report.accepted is True
    finding = next(
        item for item in report.findings if item.code == "FEATURE_ATTRIBUTION_DRIFT"
    )
    assert finding.blocking is False
    assert finding.as_dict()["severity"] == "diagnostic"


def test_feature_fraction_and_single_feature_caps_are_independent():
    value = artifact()
    value["features"][0]["current_counts"] = [200, 0, 0]
    report = audit_attribution_drift(
        value,
        evaluated_at=NOW,
        policy=DriftPolicy(max_drifting_feature_fraction=0.19),
    )

    assert {"DRIFTING_FEATURE_FRACTION", "MAX_FEATURE_JS"} <= codes(report)


def test_weighted_mean_gate_uses_reference_salience():
    value = artifact()
    for item in value["features"]:
        item["current_counts"] = [190, 5, 5]
    report = audit_attribution_drift(value, evaluated_at=NOW)

    assert "WEIGHTED_MEAN_FEATURE_JS" in codes(report)
    assert report.metrics["weighted_mean_feature_js"] > 0.05


def test_zero_reference_salience_has_finite_relative_change():
    value = artifact()
    value["features"][0]["reference_mean_abs"] = 0.0
    value["features"][0]["current_mean_abs"] = 0.0
    report = audit_attribution_drift(value, evaluated_at=NOW)
    assert report.metrics["max_mean_abs_relative_change"] < 1.0

    value["features"][0]["current_mean_abs"] = 0.01
    changed = audit_attribution_drift(
        value,
        evaluated_at=NOW,
        policy=DriftPolicy(max_drifting_feature_fraction=1.0),
    )
    assert "FEATURE_ATTRIBUTION_DRIFT" in codes(changed)
    assert math_is_finite(changed.metrics["max_mean_abs_relative_change"])


def math_is_finite(value):
    return isinstance(value, float) and value < float("inf")


def test_prediction_drift_has_a_separate_release_gate():
    value = artifact()
    value["prediction"]["current_counts"] = [190, 5, 5]
    report = audit_attribution_drift(value, evaluated_at=NOW)

    assert report.accepted is False
    assert "PREDICTION_DISTRIBUTION_DRIFT" in codes(report)
    assert report.metrics["drifting_feature_count"] == 0


def test_policy_and_configuration_digests_bind_inputs():
    baseline = audit_attribution_drift(artifact(), evaluated_at=NOW)
    changed_policy = audit_attribution_drift(
        artifact(), evaluated_at=NOW, policy=DriftPolicy(max_prediction_js=0.16)
    )
    changed_config = artifact()
    changed_config["explainer_sha256"] = HASH_C

    assert baseline.policy_sha256 != changed_policy.policy_sha256
    assert (
        baseline.configuration_sha256
        != audit_attribution_drift(
            changed_config, evaluated_at=NOW
        ).configuration_sha256
    )


def test_canonical_digest_ignores_feature_order_and_offset_spelling():
    value = artifact()
    reordered = deepcopy(value)
    reordered["features"].reverse()
    reordered["created_at"] = "2026-09-26T15:50:00+03:00"

    assert (
        audit_attribution_drift(value, evaluated_at=NOW).artifact_sha256
        == audit_attribution_drift(reordered, evaluated_at=NOW).artifact_sha256
    )


def test_stale_and_future_artifacts_fail_closed():
    stale = artifact()
    stale["created_at"] = "2026-09-26T12:00:00Z"
    stale["current_window"]["ended_at"] = "2026-09-26T11:59:00Z"
    assert "STALE_ARTIFACT" in codes(
        audit_attribution_drift(
            stale,
            evaluated_at=NOW,
            policy=DriftPolicy(max_artifact_age_seconds=60),
        )
    )

    future = artifact()
    future["created_at"] = "2026-09-26T13:06:00Z"
    assert "FUTURE_ARTIFACT" in codes(audit_attribution_drift(future, evaluated_at=NOW))


@pytest.mark.parametrize(
    "payload",
    [b"", b"[]", b'{"a":1,"a":2}', b'{"a":NaN}', b"\xff", b"{"],
)
def test_strict_loader_rejects_malformed_or_ambiguous_json(payload):
    with pytest.raises(ValueError):
        load_artifact(payload)


def test_strict_loader_enforces_byte_budget():
    payload = json.dumps(artifact()).encode()
    with pytest.raises(ValueError):
        load_artifact(payload, max_bytes=len(payload) - 1)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_version="v2"),
        lambda value: value.update(extra=True),
        lambda value: value.update(model_sha256="ABC"),
        lambda value: value.update(monitor_id="bad id"),
        lambda value: value.update(created_at="2026-09-26"),
        lambda value: value.update(features=[]),
        lambda value: value["features"][0].update(extra=True),
        lambda value: value["features"][0].update(risk="unknown"),
        lambda value: value["features"][0].update(reference_mean_abs=-1.0),
        lambda value: value["features"][0].update(current_topk_rate=True),
        lambda value: value["prediction"].update(extra=True),
    ],
)
def test_rejects_malformed_contract(mutate):
    value = artifact()
    mutate(value)
    with pytest.raises(ValueError):
        audit_attribution_drift(value, evaluated_at=NOW)


def test_rejects_duplicate_features_and_histogram_mismatches():
    duplicate = artifact()
    duplicate["features"].append(deepcopy(duplicate["features"][0]))
    with pytest.raises(ValueError):
        audit_attribution_drift(duplicate, evaluated_at=NOW)

    wrong_sum = artifact()
    wrong_sum["features"][0]["current_counts"] = [1, 1, 1]
    with pytest.raises(ValueError):
        audit_attribution_drift(wrong_sum, evaluated_at=NOW)

    wrong_bins = artifact()
    wrong_bins["prediction"]["current_counts"] = [100, 100]
    with pytest.raises(ValueError):
        audit_attribution_drift(wrong_bins, evaluated_at=NOW)


def test_window_chronology_is_fail_closed():
    overlap = artifact()
    overlap["reference_window"]["ended_at"] = "2026-09-20T00:00:00Z"
    with pytest.raises(ValueError):
        audit_attribution_drift(overlap, evaluated_at=NOW)

    premature = artifact()
    premature["created_at"] = "2026-09-26T11:00:00Z"
    with pytest.raises(ValueError):
        audit_attribution_drift(premature, evaluated_at=NOW)


def test_resource_budgets_fail_closed():
    with pytest.raises(ValueError):
        audit_attribution_drift(
            artifact(), evaluated_at=NOW, policy=DriftPolicy(max_features=4)
        )
    with pytest.raises(ValueError):
        audit_attribution_drift(
            artifact(), evaluated_at=NOW, policy=DriftPolicy(min_samples_per_window=201)
        )
    with pytest.raises(ValueError):
        audit_attribution_drift(
            artifact(), evaluated_at=NOW, policy=DriftPolicy(max_bins_per_histogram=2)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_js_drift_threshold": -0.1},
        {"max_prediction_js": float("nan")},
        {"mean_abs_floor": 0.0},
        {"max_features": True},
        {"require_no_critical_drift": "yes"},
    ],
)
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        audit_attribution_drift(
            artifact(), evaluated_at=NOW, policy=DriftPolicy(**kwargs)
        )


def test_evaluated_at_must_be_timezone_aware():
    with pytest.raises(ValueError):
        audit_attribution_drift(artifact(), evaluated_at=NOW.replace(tzinfo=None))


def test_findings_are_bounded_for_large_failed_artifacts():
    value = artifact()
    value["features"] = [
        feature(f"feature-{index}", "critical", [200, 0, 0], [0, 0, 200])
        for index in range(300)
    ]
    report = audit_attribution_drift(value, evaluated_at=NOW)
    assert len(report.findings) == 256
    assert report.findings[-1].code == "FINDINGS_TRUNCATED"
    assert report.accepted is False


def test_cli_exposes_accept_reject_and_malformed_codes(tmp_path, capsys):
    source = tmp_path / "artifact.json"
    output = tmp_path / "report.json"
    current = artifact()
    current["created_at"] = datetime.now(UTC).isoformat()
    current["current_window"]["ended_at"] = current["created_at"]
    source.write_text(json.dumps(current), encoding="utf-8")
    assert main([str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["accepted"] is True

    current["prediction"]["current_counts"] = [190, 5, 5]
    source.write_text(json.dumps(current), encoding="utf-8")
    assert main([str(source)]) == 2
    assert json.loads(capsys.readouterr().out)["accepted"] is False

    source.write_text('{"duplicate":1,"duplicate":2}', encoding="utf-8")
    assert main([str(source)]) == 3
    malformed = json.loads(capsys.readouterr().out)
    assert malformed["error_code"] == "MALFORMED_ARTIFACT"
    assert "duplicate" not in json.dumps(malformed)
