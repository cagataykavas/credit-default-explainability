from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "income",
    "debt_ratio",
    "late_payments",
    "utilization",
    "account_age_months",
    "age",
    "employment",
    "housing",
]
TARGET = "default"


def synthetic_credit_data(rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.6, 0.55, rows)
    debt_ratio = np.clip(rng.beta(2.0, 4.5, rows), 0, 1)
    late = rng.poisson(0.75, rows)
    utilization = np.clip(rng.beta(2.2, 2.8, rows), 0, 1)
    account_age = rng.integers(1, 300, rows)
    age = rng.integers(21, 75, rows)
    employment = rng.choice(["salaried", "self_employed", "student", "unemployed"], rows, p=[0.62, 0.20, 0.07, 0.11])
    housing = rng.choice(["rent", "mortgage", "owned"], rows, p=[0.42, 0.38, 0.20])

    logit = (
        -4.0
        + 2.3 * debt_ratio
        + 1.9 * utilization
        + 0.42 * late
        - 0.000012 * income
        - 0.0025 * account_age
        + 0.65 * (employment == "unemployed")
        + 0.22 * (housing == "rent")
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    target = rng.binomial(1, np.clip(probability, 0.01, 0.90))
    return pd.DataFrame(
        {
            "income": income,
            "debt_ratio": debt_ratio,
            "late_payments": late,
            "utilization": utilization,
            "account_age_months": account_age,
            "age": age,
            "employment": employment,
            "housing": housing,
            TARGET: target,
        }
    )
