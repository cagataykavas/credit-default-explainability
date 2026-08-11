from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_credit_model(df: pd.DataFrame, target: str = "default"):
    numeric = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric + [target]]
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(df.drop(columns=target), df[target])
    return model


def explain_global(model, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    importance = permutation_importance(model, x, y, scoring="roc_auc", n_repeats=8, random_state=42)
    return pd.DataFrame({
        "feature": x.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)


def scorecard(model, row: pd.DataFrame) -> dict[str, float]:
    probability = float(model.predict_proba(row)[0, 1])
    odds = probability / max(1e-9, 1 - probability)
    score = 600 - 50 / np.log(2) * np.log(max(odds, 1e-9))
    return {"default_probability": probability, "illustrative_credit_score": float(score)}


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 2500
    df = pd.DataFrame({
        "income": rng.lognormal(10.5, 0.5, n),
        "debt_ratio": rng.beta(2, 5, n),
        "late_payments": rng.poisson(0.8, n),
        "age": rng.integers(21, 70, n),
        "segment": rng.choice(["retail", "salary", "self_employed"], n),
    })
    logit = -3.0 + 4.0 * df["debt_ratio"] + 0.45 * df["late_payments"] - 0.00001 * df["income"]
    p = 1 / (1 + np.exp(-logit))
    df["default"] = rng.binomial(1, p)
    split = int(n * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    model = train_credit_model(train)
    pred = model.predict_proba(test.drop(columns="default"))[:, 1]
    print("ROC-AUC:", round(roc_auc_score(test["default"], pred), 4))
    print(explain_global(model, test.drop(columns="default"), test["default"]).head())
    print(scorecard(model, test.drop(columns="default").iloc[[0]]))
