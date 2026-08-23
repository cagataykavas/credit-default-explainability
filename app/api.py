from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from creditxai.counterfactual import find_actionable_counterfactual
from creditxai.data import FEATURES, synthetic_credit_data
from creditxai.explain import explain_decision
from creditxai.model import CreditRiskModel

app = FastAPI(title="Credit Default Explainability", version="1.0.0")


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score")
def score(payload: Applicant) -> dict[str, object]:
    row = to_frame(payload)
    model = demo_model()
    return {
        "explanation": explain_decision(model, row),
        "counterfactual": find_actionable_counterfactual(model, row).as_dict(),
    }


@app.get("/demo")
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
