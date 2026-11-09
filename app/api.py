from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from creditxai.counterfactual import find_actionable_counterfactual
from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.model import CreditRiskModel
from creditxai.scoring_admission import (
    ScoringAdmissionError,
    ScoringAdmissionGuard,
    load_environment_admission,
)


class Applicant(BaseModel):
    income: float = Field(gt=0)
    debt_ratio: float = Field(ge=0, le=1)
    late_payments: int = Field(ge=0)
    utilization: float = Field(ge=0, le=1)
    account_age_months: int = Field(ge=0)
    age: int = Field(ge=18, le=100)
    employment: str
    housing: str


@lru_cache(maxsize=1)
def demo_model() -> CreditRiskModel:
    model, _ = CreditRiskModel.fit(synthetic_credit_data(rows=5000, seed=42))
    return model


def to_frame(payload: Applicant) -> pd.DataFrame:
    return pd.DataFrame([{name: payload.model_dump()[name] for name in FEATURES}])


@lru_cache(maxsize=1)
def environment_admission() -> ScoringAdmissionGuard | None:
    return load_environment_admission(demo_model())


def create_app(
    admission: ScoringAdmissionGuard | None = None,
    *,
    use_environment: bool = False,
) -> FastAPI:
    api = FastAPI(title="Credit Default Explainability", version="1.1.0")
    provider: Callable[[], ScoringAdmissionGuard | None]
    if admission is not None:
        provider = lambda: admission
    elif use_environment:
        provider = environment_admission
    else:
        provider = lambda: None

    def require_admission() -> ScoringAdmissionGuard:
        try:
            configured = provider()
        except ScoringAdmissionError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="explanation admission configuration is invalid",
            ) from error
        if configured is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="explanation admission is not configured",
            )
        return configured

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/ready")
    def ready() -> dict[str, object]:
        guard = require_admission()
        try:
            audit = guard.check_ready(now=datetime.now(UTC))
        except ScoringAdmissionError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="explanation admission rejected",
            ) from error
        return {
            "status": "ready",
            "explanation_admission": "configured",
            "model_digest": audit.model_digest,
            "pipeline_digest": audit.pipeline_digest,
            "contract_digest": audit.contract_digest,
        }

    @api.post("/score")
    def score(payload: Applicant) -> dict[str, object]:
        guard = require_admission()
        row = to_frame(payload)
        try:
            admitted = guard.score(row, now=datetime.now(UTC))
        except ScoringAdmissionError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="explanation admission rejected",
            ) from error
        return {
            "explanation": admitted.explanation,
            "counterfactual": find_actionable_counterfactual(
                guard.model, row
            ).as_dict(),
            "admission": admitted.admission_summary(),
        }

    @api.get("/demo")
    def demo() -> dict[str, object]:
        payload = Applicant(
            income=42_000,
            debt_ratio=0.63,
            late_payments=3,
            utilization=0.82,
            account_age_months=24,
            age=34,
            employment="salaried",
            housing="rent",
        )
        return score(payload)

    return api


app = create_app(use_environment=True)
