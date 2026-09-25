from __future__ import annotations

import json
from copy import deepcopy

import pytest

from creditxai.baseline_sensitivity import (
    AuditPolicy,
    EvidenceError,
    audit_baseline_sensitivity,
    load_evidence,
    main,
)


def _attrs(a: float, b: float, c: float) -> list[dict[str, float | str]]:
    return [
        {"feature_id": "income", "value": a},
        {"feature_id": "debt_ratio", "value": b},
        {"feature_id": "utilization", "value": c},
    ]


def evidence() -> dict:
    return {
        "schema_version": "1.0",
        "method_id": "exact-shapley-v1",
        "model_id": "credit-risk-2026-09",
        "cases": [
            {
                "case_id": "case-001",
                "observed_probability": 0.8,
                "baselines": [
                    {
                        "baseline_id": "median-cohort",
                        "baseline_probability": 0.2,
                        "attributions": _attrs(0.3, 0.2, 0.1),
                    },
                    {
                        "baseline_id": "low-risk-cohort",
                        "baseline_probability": 0.3,
                        "attributions": _attrs(0.25, 0.17, 0.08),
                    },
                    {
                        "baseline_id": "recent-cohort",
                        "baseline_probability": 0.4,
                        "attributions": _attrs(0.2, 0.14, 0.06),
                    },
                ],
            }
        ],
    }


def test_accepts_consistent_complete_explanations():
    report = audit_baseline_sensitivity(evidence(), AuditPolicy())
    assert report.accepted
    assert report.reason_codes == ()
    assert report.pair_comparison_count == 3
    assert report.cases[0].accepted
    assert report.cases[0].min_pairwise_cosine > 0.99
    assert len(report.evidence_sha256) == 64
    assert len(report.report_sha256) == 64


def test_rejects_direction_and_salience_instability():
    artifact = evidence()
    artifact["cases"][0]["baselines"][2]["attributions"] = _attrs(-0.4, 0.1, 0.7)
    report = audit_baseline_sensitivity(artifact, AuditPolicy())
    assert not report.accepted
    assert "ATTRIBUTION_DIRECTION_UNSTABLE" in report.reason_codes
    assert "SALIENT_FEATURE_SIGNS_UNSTABLE" in report.reason_codes


def test_rejects_top_k_churn():
    artifact = evidence()
    artifact["cases"][0]["baselines"][2]["attributions"] = [
        {"feature_id": "income", "value": 0.01},
        {"feature_id": "debt_ratio", "value": 0.01},
        {"feature_id": "utilization", "value": 0.01},
        {"feature_id": "late_payments", "value": 0.18},
        {"feature_id": "age", "value": 0.19},
    ]
    for baseline in artifact["cases"][0]["baselines"][:2]:
        baseline["attributions"].extend(
            [
                {"feature_id": "late_payments", "value": 0.0},
                {"feature_id": "age", "value": 0.0},
            ]
        )
    report = audit_baseline_sensitivity(artifact, AuditPolicy())
    assert "TOP_K_FEATURES_UNSTABLE" in report.reason_codes


def test_rejects_completeness_mismatch():
    artifact = evidence()
    artifact["cases"][0]["baselines"][0]["attributions"][0]["value"] = 0.31
    report = audit_baseline_sensitivity(artifact, AuditPolicy())
    assert "COMPLETENESS_MISMATCH" in report.reason_codes


def test_rejects_weak_contrast():
    artifact = evidence()
    artifact["cases"][0]["observed_probability"] = 0.405
    artifact["cases"][0]["baselines"][2]["attributions"] = _attrs(0.002, 0.002, 0.001)
    report = audit_baseline_sensitivity(
        artifact, AuditPolicy(max_completeness_error=1.0, min_pairwise_cosine=-1.0)
    )
    assert "WEAK_BASELINE_CONTRAST" in report.reason_codes


def test_rejects_excessive_baseline_output_span():
    artifact = evidence()
    artifact["cases"][0]["baselines"][0]["baseline_probability"] = 0.01
    artifact["cases"][0]["baselines"][0]["attributions"] = _attrs(0.4, 0.25, 0.14)
    report = audit_baseline_sensitivity(
        artifact, AuditPolicy(max_baseline_probability_span=0.2)
    )
    assert "BASELINE_OUTPUT_SPAN_EXCEEDED" in report.reason_codes


def test_allows_governed_failed_case_fraction():
    artifact = evidence()
    second = deepcopy(artifact["cases"][0])
    second["case_id"] = "case-002"
    second["baselines"][0]["attributions"][0]["value"] = 0.31
    artifact["cases"].append(second)
    report = audit_baseline_sensitivity(
        artifact, AuditPolicy(min_cases=2, max_failed_case_fraction=0.5)
    )
    assert report.accepted
    assert report.failed_case_count == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x.update(schema_version="2.0"),
        lambda x: x.update(extra=True),
        lambda x: x["cases"].clear(),
        lambda x: x["cases"].append(deepcopy(x["cases"][0])),
        lambda x: x["cases"][0]["baselines"].pop(),
        lambda x: x["cases"][0]["baselines"][0].update(baseline_probability=1.1),
        lambda x: x["cases"][0]["baselines"][0]["attributions"].append(
            {"feature_id": "income", "value": 0.0}
        ),
        lambda x: x["cases"][0]["baselines"][0]["attributions"].pop(),
        lambda x: x["cases"][0]["baselines"][0]["attributions"][0].update(
            value=float("nan")
        ),
        lambda x: x["cases"][0]["baselines"][1]["attributions"][0].update(
            feature_id="other"
        ),
    ],
)
def test_fails_closed_on_malformed_evidence(mutate):
    artifact = evidence()
    mutate(artifact)
    with pytest.raises(EvidenceError):
        audit_baseline_sensitivity(artifact, AuditPolicy())


@pytest.mark.parametrize(
    "policy",
    [
        AuditPolicy(top_k=0),
        AuditPolicy(min_pairwise_cosine=2.0),
        AuditPolicy(max_failed_case_fraction=-0.1),
        AuditPolicy(min_cases=2, max_cases=1),
        AuditPolicy(min_baselines_per_case=21, max_baselines_per_case=20),
    ],
)
def test_rejects_invalid_policy(policy):
    with pytest.raises(EvidenceError):
        audit_baseline_sensitivity(evidence(), policy)


def test_pair_budget_is_fail_closed():
    with pytest.raises(EvidenceError, match="pair-comparison budget"):
        audit_baseline_sensitivity(evidence(), AuditPolicy(max_pair_comparisons=2))


def test_report_is_deterministic_and_redacts_case_and_baseline_ids():
    first = audit_baseline_sensitivity(evidence(), AuditPolicy()).as_dict()
    second = audit_baseline_sensitivity(evidence(), AuditPolicy()).as_dict()
    rendered = json.dumps(first, sort_keys=True)
    assert first == second
    assert "case-001" not in rendered
    assert "median-cohort" not in rendered


def test_strict_loader_rejects_duplicate_keys_and_non_finite(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"1.0","schema_version":"1.0"}')
    with pytest.raises(EvidenceError, match="duplicate"):
        load_evidence(duplicate)
    non_finite = tmp_path / "nan.json"
    non_finite.write_text('{"value":NaN}')
    with pytest.raises(EvidenceError, match="non-finite"):
        load_evidence(non_finite)


def test_loader_enforces_byte_budget(tmp_path):
    path = tmp_path / "large.json"
    path.write_text(json.dumps(evidence()))
    with pytest.raises(EvidenceError, match="byte budget"):
        load_evidence(path, max_bytes=10)


def test_cli_writes_atomic_accepted_report(tmp_path, capsys):
    source = tmp_path / "evidence.json"
    target = tmp_path / "report.json"
    source.write_text(json.dumps(evidence()))
    assert main([str(source), "--output", str(target)]) == 0
    assert json.loads(target.read_text())["accepted"] is True
    assert json.loads(capsys.readouterr().out)["accepted"] is True


def test_cli_distinguishes_policy_rejection_and_malformed_input(tmp_path, capsys):
    rejected = evidence()
    rejected["cases"][0]["baselines"][0]["attributions"][0]["value"] = 0.31
    rejected_path = tmp_path / "rejected.json"
    rejected_path.write_text(json.dumps(rejected))
    assert main([str(rejected_path)]) == 2
    assert json.loads(capsys.readouterr().out)["accepted"] is False

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{")
    assert main([str(malformed_path)]) == 3
    assert json.loads(capsys.readouterr().out)["error_code"] == "MALFORMED_EVIDENCE"
