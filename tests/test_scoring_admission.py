from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import Applicant, create_app, to_frame
from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.explain import local_perturbation_attribution
from creditxai.explanation_contract import (
    bind_explanation,
    build_credit_feature_contract,
    fingerprint_credit_model,
    sign_promotion_receipt,
)
from creditxai.model import CreditRiskModel
from creditxai.scoring_admission import (
    ScoringAdmissionError,
    ScoringAdmissionGuard,
    load_environment_admission,
)

KEY_ID = "credit-prod-2026-11"
KEY = b"production-key-material-for-api-tests"
AUDIENCE = "credit-xai-production"
APPLICANT = Applicant(
    income=42_000,
    debt_ratio=0.63,
    late_payments=3,
    utilization=0.82,
    account_age_months=24,
    age=34,
    employment="salaried",
    housing="rent",
)


@pytest.fixture(scope="module")
def fitted_model() -> CreditRiskModel:
    model, _ = CreditRiskModel.fit(synthetic_credit_data(rows=2200, seed=117))
    return model


def promotion_artifact(
    model: CreditRiskModel,
    *,
    now: datetime,
    expires_at: datetime,
) -> dict:
    contract = build_credit_feature_contract(
        model,
        model_digest=fingerprint_credit_model(model),
    )
    receipt = sign_promotion_receipt(
        contract,
        key_id=KEY_ID,
        secret=KEY,
        audience=AUDIENCE,
        issued_at=now - timedelta(minutes=1),
        expires_at=expires_at,
    )
    row = to_frame(APPLICANT)
    return bind_explanation(
        contract,
        local_perturbation_attribution(model, row),
        generated_at=now,
        promotion=receipt,
    ).to_dict()


def guard_for(model: CreditRiskModel, artifact: object) -> ScoringAdmissionGuard:
    return ScoringAdmissionGuard.from_promotion_artifact(
        model,
        artifact,
        audience=AUDIENCE,
        trusted_promotion_keys={KEY_ID: KEY},
    )


def test_admitted_api_binds_response_to_running_model(
    fitted_model: CreditRiskModel,
) -> None:
    now = datetime.now(UTC)
    guard = guard_for(
        fitted_model,
        promotion_artifact(
            fitted_model,
            now=now,
            expires_at=now + timedelta(hours=1),
        ),
    )
    client = TestClient(create_app(admission=guard))

    readiness = client.get("/ready")
    response = client.post("/score", json=APPLICANT.model_dump())

    assert readiness.status_code == 200
    assert response.status_code == 200
    payload = response.json()
    assert payload["admission"]["accepted"]
    assert payload["admission"]["promotion_verified"]
    assert payload["admission"]["model_digest"] == fingerprint_credit_model(
        fitted_model
    )
    assert payload["admission"]["attribution_count"] == len(FEATURES)
    assert "explanation" in payload
    assert "counterfactual" in payload
    rendered = json.dumps(payload)
    assert KEY_ID not in rendered
    assert KEY.decode() not in rendered


def test_tampered_promotion_fails_closed_at_score_boundary(
    fitted_model: CreditRiskModel,
) -> None:
    now = datetime.now(UTC)
    artifact = promotion_artifact(
        fitted_model,
        now=now,
        expires_at=now + timedelta(hours=1),
    )
    artifact["promotion"]["signature"] = "0" * 64
    client = TestClient(create_app(admission=guard_for(fitted_model, artifact)))

    readiness = client.get("/ready")
    response = client.post("/score", json=APPLICANT.model_dump())

    assert readiness.status_code == 503
    assert response.status_code == 503
    assert response.json() == {"detail": "explanation admission rejected"}


def test_expired_promotion_fails_closed_at_score_boundary(
    fitted_model: CreditRiskModel,
) -> None:
    now = datetime.now(UTC)
    artifact = promotion_artifact(
        fitted_model,
        now=now - timedelta(hours=2),
        expires_at=now - timedelta(hours=1),
    )
    client = TestClient(create_app(admission=guard_for(fitted_model, artifact)))

    readiness = client.get("/ready")
    response = client.post("/score", json=APPLICANT.model_dump())

    assert readiness.status_code == 503
    assert response.status_code == 503


def test_promotion_for_different_fitted_model_is_rejected_before_serving(
    fitted_model: CreditRiskModel,
) -> None:
    now = datetime.now(UTC)
    other_model, _ = CreditRiskModel.fit(synthetic_credit_data(rows=2200, seed=118))
    artifact = promotion_artifact(
        fitted_model,
        now=now,
        expires_at=now + timedelta(hours=1),
    )

    with pytest.raises(ScoringAdmissionError, match="running model"):
        guard_for(other_model, artifact)


def test_fingerprint_changes_with_fitted_classifier_state(
    fitted_model: CreditRiskModel,
) -> None:
    altered_model = copy.deepcopy(fitted_model)
    estimator = (
        altered_model.pipeline.named_steps["classifier"]
        .calibrated_classifiers_[0]
        .estimator
    )
    estimator.coef_[0, 0] += 1e-6

    assert fingerprint_credit_model(altered_model) != fingerprint_credit_model(
        fitted_model
    )


def test_environment_loader_uses_indirect_secret_reference(
    fitted_model: CreditRiskModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    artifact_path = tmp_path / "promotion.json"
    artifact_path.write_text(
        json.dumps(
            promotion_artifact(
                fitted_model,
                now=now,
                expires_at=now + timedelta(hours=1),
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CREDIT_XAI_PROMOTION_ARTIFACT", str(artifact_path))
    monkeypatch.setenv("CREDIT_XAI_PROMOTION_AUDIENCE", AUDIENCE)
    monkeypatch.setenv("CREDIT_XAI_PROMOTION_KEY_ID", KEY_ID)
    monkeypatch.setenv("CREDIT_XAI_PROMOTION_KEY_ENV", "TEST_PROMOTION_SECRET")
    monkeypatch.setenv("TEST_PROMOTION_SECRET", KEY.decode())

    guard = load_environment_admission(fitted_model)

    assert guard is not None
    assert guard.check_ready(now=now).accepted


def test_partial_environment_configuration_is_rejected(
    fitted_model: CreditRiskModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CREDIT_XAI_PROMOTION_ARTIFACT", "/missing/promotion.json")

    with pytest.raises(ScoringAdmissionError, match="incomplete"):
        load_environment_admission(fitted_model)
