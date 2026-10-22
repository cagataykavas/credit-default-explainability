from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.explain import FeatureContribution, local_perturbation_attribution
from creditxai.faithfulness import (
    EXIT_ACCEPTED,
    EXIT_MALFORMED,
    FaithfulnessAuditError,
    FaithfulnessPolicy,
    audit_sequential_ablation,
    cohort_content_sha256,
    main,
    write_report_atomic,
)
from creditxai.model import CreditRiskModel

MODEL_DIGEST = "a" * 64
COHORT_DIGEST = "b" * 64
EXPLAINER_DIGEST = "c" * 64


def _baseline() -> dict[str, Any]:
    return {
        "income": 50_000.0,
        "debt_ratio": 0.30,
        "late_payments": 1,
        "utilization": 0.40,
        "account_age_months": 60,
        "age": 40,
        "employment": "salaried",
        "housing": "mortgage",
    }


def _rows(count: int = 4) -> pd.DataFrame:
    values = []
    for index in range(count):
        values.append(
            {
                "income": 35_000.0 + index * 1_000,
                "debt_ratio": 0.72 - index * 0.01,
                "late_payments": 4,
                "utilization": 0.86 - index * 0.01,
                "account_age_months": 24,
                "age": 36,
                "employment": "unemployed",
                "housing": "rent",
            }
        )
    return pd.DataFrame(
        values, columns=FEATURES, index=[f"customer-{i}" for i in range(count)]
    )


class AdditiveModel:
    def __init__(self) -> None:
        self.baseline = _baseline()

    def baseline_row(self) -> dict[str, Any]:
        return dict(self.baseline)

    def predict_probability(self, row: pd.DataFrame) -> float:
        value = row.iloc[0]
        score = (
            -1.2
            - 0.00002 * (float(value["income"]) - 50_000.0)
            + 2.4 * (float(value["debt_ratio"]) - 0.30)
            + 0.35 * (float(value["late_payments"]) - 1.0)
            + 1.8 * (float(value["utilization"]) - 0.40)
            - 0.004 * (float(value["account_age_months"]) - 60.0)
            + 0.002 * (float(value["age"]) - 40.0)
            + (0.55 if value["employment"] == "unemployed" else 0.0)
            + (0.20 if value["housing"] == "rent" else 0.0)
        )
        return 1.0 / (1.0 + math.exp(-score))


def _audit(
    *,
    model: Any | None = None,
    cohort: pd.DataFrame | None = None,
    policy: FaithfulnessPolicy | None = None,
    attribution_fn=local_perturbation_attribution,
):
    active_cohort = _rows() if cohort is None else cohort
    return audit_sequential_ablation(
        model or AdditiveModel(),
        active_cohort,
        model_sha256=MODEL_DIGEST,
        cohort_sha256=cohort_content_sha256(active_cohort),
        explainer_sha256=EXPLAINER_DIGEST,
        policy=policy,
        attribution_fn=attribution_fn,
    )


def test_accepts_faithful_attribution_order() -> None:
    report = _audit()

    assert report.outcome == "accepted"
    assert report.eligible_cases == 4
    assert report.failed_cases == 0
    assert report.prediction_calls == 4 * 29
    assert all(metric.ordering_advantage > 0 for metric in report.metrics)
    assert all(metric.top_k_recovery > 0 for metric in report.metrics)
    assert len(report.evidence_sha256) == 64


def test_is_deterministic_and_input_order_independent() -> None:
    first = _audit(cohort=_rows(3))
    reordered = _rows(3).iloc[::-1]
    second = _audit(cohort=reordered)

    assert first.evidence_sha256 == second.evidence_sha256
    assert first.metrics == second.metrics


def test_cohort_identity_is_order_independent_and_content_sensitive() -> None:
    cohort = _rows(3)
    reordered = cohort.iloc[::-1]
    changed_value = cohort.copy(deep=True)
    changed_value.loc["customer-0", "income"] += 1.0
    changed_identity = cohort.copy(deep=True)
    changed_identity.index = ["replacement", "customer-1", "customer-2"]

    assert cohort_content_sha256(cohort) == cohort_content_sha256(reordered)
    assert cohort_content_sha256(cohort) != cohort_content_sha256(changed_value)
    assert cohort_content_sha256(cohort) != cohort_content_sha256(changed_identity)


def test_string_and_integer_case_identities_do_not_collide() -> None:
    cohort = _rows(2)
    cohort.index = [1, "1"]

    report = _audit(cohort=cohort)

    assert len({metric.case_ref for metric in report.metrics}) == 2


def test_rejects_declared_digest_that_does_not_match_evaluated_cohort() -> None:
    with pytest.raises(FaithfulnessAuditError, match="does not match"):
        audit_sequential_ablation(
            AdditiveModel(),
            _rows(),
            model_sha256=MODEL_DIGEST,
            cohort_sha256=COHORT_DIGEST,
            explainer_sha256=EXPLAINER_DIGEST,
        )


def test_rejects_cohort_mutation_during_audit() -> None:
    cohort = _rows()
    mutated = False

    def mutating_attribution(model, row):
        nonlocal mutated
        if not mutated:
            cohort.loc["customer-0", "income"] += 1.0
            mutated = True
        return local_perturbation_attribution(model, row)

    with pytest.raises(FaithfulnessAuditError, match="changed during the audit"):
        _audit(cohort=cohort, attribution_fn=mutating_attribution)


def test_rejects_reverse_ranked_attributions() -> None:
    def reversed_attributions(model, row):
        return tuple(reversed(local_perturbation_attribution(model, row)))

    report = _audit(attribution_fn=reversed_attributions)

    assert report.outcome == "rejected"
    assert report.failed_cases == report.eligible_cases
    codes = {finding.code for finding in report.findings}
    assert "ordering_advantage_below_minimum" in codes
    assert "failed_case_fraction_above_maximum" in codes


def test_allows_a_governed_failed_case_fraction() -> None:
    def partly_reversed(model, row):
        values = local_perturbation_attribution(model, row)
        return tuple(reversed(values)) if row.index[0] == "customer-0" else values

    report = _audit(
        cohort=_rows(4),
        policy=FaithfulnessPolicy(max_failed_case_fraction=0.25),
        attribution_fn=partly_reversed,
    )

    assert report.outcome == "accepted"
    assert report.failed_cases == 1
    assert any(not metric.passed for metric in report.metrics)


def test_rejects_when_too_few_cases_have_output_contrast() -> None:
    cohort = pd.DataFrame(
        [_baseline(), _baseline(), _rows(1).iloc[0]], columns=FEATURES
    )

    report = _audit(cohort=cohort)

    assert report.outcome == "rejected"
    assert report.excluded_low_contrast_cases == 2
    assert "eligible_case_fraction_below_minimum" in {
        finding.code for finding in report.findings
    }


def test_policy_can_require_stronger_top_k_recovery() -> None:
    report = _audit(policy=FaithfulnessPolicy(min_top_k_recovery=1.0))

    assert report.outcome == "rejected"
    assert "top_k_recovery_below_minimum" in {
        finding.code for finding in report.findings
    }


def test_detects_prediction_drift() -> None:
    class DriftingModel(AdditiveModel):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def predict_probability(self, row: pd.DataFrame) -> float:
            self.calls += 1
            return min(1.0, super().predict_probability(row) + self.calls * 1e-5)

    report = _audit(
        model=DriftingModel(),
        policy=FaithfulnessPolicy(max_prediction_drift=1e-8),
    )

    assert report.outcome == "rejected"
    assert "prediction_drift_above_maximum" in {
        finding.code for finding in report.findings
    }


@pytest.mark.parametrize("digest_name", ["model", "cohort", "explainer"])
def test_rejects_noncanonical_digests(digest_name: str) -> None:
    kwargs = {
        "model_sha256": MODEL_DIGEST,
        "cohort_sha256": COHORT_DIGEST,
        "explainer_sha256": EXPLAINER_DIGEST,
    }
    kwargs[f"{digest_name}_sha256"] = "A" * 64

    with pytest.raises(FaithfulnessAuditError, match="lowercase SHA-256"):
        audit_sequential_ablation(AdditiveModel(), _rows(), **kwargs)


def test_rejects_wrong_columns_and_duplicate_indices() -> None:
    with pytest.raises(FaithfulnessAuditError, match="FEATURES order"):
        _audit(cohort=_rows()[list(reversed(FEATURES))])

    cohort = _rows(2)
    cohort.index = ["duplicate", "duplicate"]
    with pytest.raises(FaithfulnessAuditError, match="indices must be unique"):
        _audit(cohort=cohort)


@pytest.mark.parametrize(
    ("feature", "value", "match"),
    [
        ("income", 0.0, "positive"),
        ("income", float("nan"), "finite"),
        ("debt_ratio", 1.1, "in \\[0, 1\\]"),
        ("utilization", -0.1, "in \\[0, 1\\]"),
        ("late_payments", 1.5, "non-negative integer"),
        ("account_age_months", -1, "non-negative integer"),
        ("age", True, "numeric"),
        ("employment", "", "bounded non-empty"),
        ("housing", "x" * 129, "bounded non-empty"),
    ],
)
def test_rejects_invalid_cohort_values(feature: str, value: object, match: str) -> None:
    cohort = _rows(1)
    cohort[feature] = cohort[feature].astype(object)
    cohort.loc[cohort.index[0], feature] = value

    with pytest.raises(FaithfulnessAuditError, match=match):
        _audit(cohort=cohort)


def test_rejects_invalid_baseline_contract() -> None:
    model = AdditiveModel()
    model.baseline.pop("income")

    with pytest.raises(FaithfulnessAuditError, match="feature set"):
        _audit(model=model)


@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.1, "0.5"])
def test_rejects_invalid_model_probabilities(value: object) -> None:
    class InvalidModel(AdditiveModel):
        def predict_probability(self, row: pd.DataFrame):
            return value

    with pytest.raises(FaithfulnessAuditError, match="probability"):
        _audit(model=InvalidModel())


def test_enforces_prediction_call_budget() -> None:
    with pytest.raises(FaithfulnessAuditError, match="budget exhausted"):
        _audit(policy=FaithfulnessPolicy(max_prediction_calls=10))


def test_rejects_invalid_attribution_shape_and_values() -> None:
    def missing(model, row):
        return local_perturbation_attribution(model, row)[:-1]

    with pytest.raises(FaithfulnessAuditError, match="one attribution"):
        _audit(attribution_fn=missing)

    def duplicate(model, row):
        values = local_perturbation_attribution(model, row)
        return [*values[:-1], values[0]]

    with pytest.raises(FaithfulnessAuditError, match="incomplete or duplicated"):
        _audit(attribution_fn=duplicate)

    def nonfinite(model, row):
        values = local_perturbation_attribution(model, row)
        return [
            FeatureContribution(
                feature=item.feature,
                observed=item.observed,
                baseline=item.baseline,
                probability_delta=float("nan")
                if index == 0
                else item.probability_delta,
                direction=item.direction,
            )
            for index, item in enumerate(values)
        ]

    with pytest.raises(FaithfulnessAuditError, match="must be finite"):
        _audit(attribution_fn=nonfinite)

    def invalid_object(model, row):
        return [object()] * len(FEATURES)

    with pytest.raises(FaithfulnessAuditError, match="invalid attribution object"):
        _audit(attribution_fn=invalid_object)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": True},
        {"top_k": 33, "max_features": 32},
        {"max_cases": 0},
        {"max_features": 0},
        {"max_prediction_calls": 0},
        {"max_findings": 0},
        {"min_output_gap": float("nan")},
        {"min_eligible_case_fraction": 1.1},
        {"min_top_k_recovery": True},
        {"max_prediction_drift": -0.1},
        {"max_failed_case_fraction": 1.1},
        {"min_ordering_advantage": 2.1},
    ],
)
def test_rejects_invalid_policies(kwargs: dict[str, object]) -> None:
    with pytest.raises(FaithfulnessAuditError):
        FaithfulnessPolicy(**kwargs)


def test_bounds_and_sorts_findings() -> None:
    def reversed_attributions(model, row):
        return tuple(reversed(local_perturbation_attribution(model, row)))

    report = _audit(
        policy=FaithfulnessPolicy(max_findings=2),
        attribution_fn=reversed_attributions,
    )

    assert len(report.findings) == 2
    assert report.findings_truncated is True
    assert list(report.findings) == sorted(
        report.findings, key=lambda item: (item.code, item.case_ref or "")
    )


def test_report_is_privacy_reduced() -> None:
    report = _audit(cohort=_rows(1))
    encoded = json.dumps(report.to_dict())

    assert "customer-0" not in encoded
    assert "unemployed" not in encoded
    assert "35000" not in encoded


def test_atomic_report_writer(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.json"
    report = _audit(cohort=_rows(1))

    write_report_atomic(path, report)

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["evidence_sha256"] == report.evidence_sha256
    assert not list(path.parent.glob(f".{path.name}.*"))


def test_real_credit_model_integration() -> None:
    frame = synthetic_credit_data(rows=1_000, seed=19)
    model, _ = CreditRiskModel.fit(frame)
    cohort = frame.iloc[800:816].drop(columns="default")[FEATURES]
    policy = FaithfulnessPolicy(max_failed_case_fraction=0.25)

    first = _audit(model=model, cohort=cohort, policy=policy)
    second = _audit(model=model, cohort=cohort, policy=policy)

    assert first == second
    assert first.outcome == "accepted"
    assert first.eligible_cases >= 8


def test_cli_writes_accepted_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "faithfulness.json"

    code = main(
        [
            "--rows",
            "800",
            "--audit-rows",
            "12",
            "--seed",
            "19",
            "--output",
            str(output),
        ]
    )

    assert code == EXIT_ACCEPTED
    assert json.loads(output.read_text(encoding="utf-8"))["outcome"] == "accepted"
    assert json.loads(capsys.readouterr().out)["outcome"] == "accepted"


def test_cli_returns_malformed_for_invalid_resource_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--rows", "100", "--output", str(tmp_path / "report.json")])

    assert code == EXIT_MALFORMED
    assert json.loads(capsys.readouterr().out)["outcome"] == "malformed"
