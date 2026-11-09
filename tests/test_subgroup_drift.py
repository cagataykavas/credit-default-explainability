from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from creditxai.subgroup_drift import (
    SCHEMA_VERSION,
    SubgroupDriftPolicy,
    audit_subgroup_drift,
    main,
)

NOW = datetime(2026, 9, 26, 13, 0, tzinfo=UTC)
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def feature(feature_id: str, risk: str, sample_count: int) -> dict[str, object]:
    quarter = sample_count // 4
    middle = sample_count - (2 * quarter)
    return {
        "feature_id": feature_id,
        "risk": risk,
        "bin_edges_sha256": HASH_D,
        "reference_counts": [quarter, middle, quarter],
        "current_counts": [quarter, middle, quarter],
        "reference_mean_abs": 0.20,
        "current_mean_abs": 0.21,
        "reference_topk_rate": 0.40,
        "current_topk_rate": 0.42,
    }


def drift_artifact(sample_count: int, monitor_id: str) -> dict[str, object]:
    return {
        "schema_version": "credit-attribution-drift/v1",
        "monitor_id": monitor_id,
        "model_sha256": HASH_A,
        "explainer_sha256": HASH_B,
        "created_at": "2026-09-26T12:50:00Z",
        "reference_window": {
            "manifest_sha256": HASH_B,
            "sample_count": sample_count,
            "started_at": "2026-09-01T00:00:00Z",
            "ended_at": "2026-09-08T00:00:00Z",
        },
        "current_window": {
            "manifest_sha256": HASH_C,
            "sample_count": sample_count,
            "started_at": "2026-09-15T00:00:00Z",
            "ended_at": "2026-09-26T12:00:00Z",
        },
        "features": [
            feature("income-secret", "standard", sample_count),
            feature("debt-secret", "critical", sample_count),
        ],
        "prediction": {
            "bin_edges_sha256": HASH_A,
            "reference_counts": [sample_count // 2, sample_count // 2],
            "current_counts": [sample_count // 2, sample_count // 2],
        },
    }


def partition() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "partition_id": "region-secret",
        "aggregate": drift_artifact(200, "aggregate-secret"),
        "slices": [
            {
                "slice_id": "north-secret",
                "artifact": drift_artifact(100, "north-monitor-secret"),
            },
            {
                "slice_id": "south-secret",
                "artifact": drift_artifact(100, "south-monitor-secret"),
            },
        ],
    }


def policy(**kwargs) -> SubgroupDriftPolicy:
    return SubgroupDriftPolicy(
        governed_slice_ids=("north-secret", "south-secret"), **kwargs
    )


def codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def test_complete_stable_partition_is_accepted_without_raw_identifiers():
    report = audit_subgroup_drift(partition(), evaluated_at=NOW, policy=policy())

    assert report.accepted is True
    assert report.metrics == {
        "governed_slice_count": 2,
        "observed_slice_count": 2,
        "eligible_slice_count": 2,
        "drifting_slice_count": 0,
        "reference_coverage": 1.0,
        "current_coverage": 1.0,
    }
    encoded = json.dumps(report.as_dict())
    assert "secret" not in encoded
    assert len(report.partition_sha256) == 64
    assert len(report.policy_sha256) == 64


def test_subgroup_drift_blocks_when_aggregate_remains_stable():
    value = partition()
    value["slices"][0]["artifact"]["features"][1]["current_topk_rate"] = 0.90

    report = audit_subgroup_drift(value, evaluated_at=NOW, policy=policy())

    assert report.aggregate_report.accepted is True
    assert report.accepted is False
    assert report.metrics["drifting_slice_count"] == 1
    assert "SUBGROUP_DRIFT" in codes(report)


def test_missing_and_undeclared_slices_fail_closed_with_hashed_evidence():
    value = partition()
    value["slices"].pop()
    value["slices"].append(
        {
            "slice_id": "other-secret",
            "artifact": drift_artifact(100, "other-monitor-secret"),
        }
    )

    report = audit_subgroup_drift(value, evaluated_at=NOW, policy=policy())

    assert {"MISSING_GOVERNED_SLICE", "UNDECLARED_SLICE"} <= codes(report)
    assert "secret" not in json.dumps(
        [finding.as_dict() for finding in report.findings]
    )


def test_minimum_support_and_complete_coverage_are_independent_gates():
    value = partition()
    value["slices"][0]["artifact"] = drift_artifact(40, "north-monitor-secret")
    value["slices"][1]["artifact"] = drift_artifact(100, "south-monitor-secret")

    report = audit_subgroup_drift(value, evaluated_at=NOW, policy=policy())

    assert {"INSUFFICIENT_SLICE_SUPPORT", "INCOMPLETE_PARTITION"} <= codes(report)
    assert report.metrics["eligible_slice_count"] == 1
    assert report.metrics["reference_coverage"] == 0.7


def test_partial_partitions_can_be_explicitly_allowed_but_still_require_support():
    value = partition()
    value["slices"][0]["artifact"] = drift_artifact(80, "north-monitor-secret")
    value["slices"][1]["artifact"] = drift_artifact(80, "south-monitor-secret")

    report = audit_subgroup_drift(
        value,
        evaluated_at=NOW,
        policy=policy(require_complete_partition=False),
    )

    assert report.accepted is True
    assert report.metrics["reference_coverage"] == 0.8


def test_overlapping_partition_is_rejected_even_when_partial_is_allowed():
    value = partition()
    value["slices"][0]["artifact"] = drift_artifact(120, "north-monitor-secret")
    value["slices"][1]["artifact"] = drift_artifact(120, "south-monitor-secret")

    report = audit_subgroup_drift(
        value,
        evaluated_at=NOW,
        policy=policy(require_complete_partition=False),
    )

    assert report.accepted is False
    assert "OVERLAPPING_OR_INVALID_PARTITION" in codes(report)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(schema_version="v2"),
        lambda value: value["slices"].append(deepcopy(value["slices"][0])),
        lambda value: value["slices"][0]["artifact"].update(model_sha256=HASH_D),
        lambda value: value["slices"][0]["artifact"]["features"][0].update(
            bin_edges_sha256=HASH_C
        ),
        lambda value: value["slices"][0]["artifact"]["reference_window"].update(
            started_at="2026-08-01T00:00:00Z"
        ),
    ],
)
def test_incompatible_or_ambiguous_partition_is_malformed(mutate):
    value = partition()
    mutate(value)
    with pytest.raises(ValueError):
        audit_subgroup_drift(value, evaluated_at=NOW, policy=policy())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"governed_slice_ids": ()},
        {"governed_slice_ids": ("north", "north")},
        {"governed_slice_ids": ("bad id",)},
        {"governed_slice_ids": ("north",), "min_samples_per_slice": 0},
        {"governed_slice_ids": ("north",), "max_slices": True},
        {"governed_slice_ids": ("north",), "require_complete_partition": "yes"},
    ],
)
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        audit_subgroup_drift(
            partition(),
            evaluated_at=NOW,
            policy=SubgroupDriftPolicy(**kwargs),
        )


def test_cli_exposes_accept_reject_and_malformed_codes(tmp_path, capsys):
    source = tmp_path / "partition.json"
    value = partition()
    current_time = datetime.now(UTC).isoformat()
    for artifact in [
        value["aggregate"],
        *(item["artifact"] for item in value["slices"]),
    ]:
        artifact["created_at"] = current_time
        artifact["current_window"]["ended_at"] = current_time
    source.write_text(json.dumps(value), encoding="utf-8")
    arguments = [
        str(source),
        "--governed-slice",
        "north-secret",
        "--governed-slice",
        "south-secret",
    ]
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["accepted"] is True

    value["slices"].pop()
    source.write_text(json.dumps(value), encoding="utf-8")
    assert main(arguments) == 2
    assert json.loads(capsys.readouterr().out)["accepted"] is False

    source.write_text('{"a":1,"a":2}', encoding="utf-8")
    assert main(arguments) == 3
    malformed = json.loads(capsys.readouterr().out)
    assert malformed["error_code"] == "MALFORMED_PARTITION_ARTIFACT"
