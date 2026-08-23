from __future__ import annotations

import argparse
import json
from pathlib import Path

from .counterfactual import find_actionable_counterfactual
from .data import FEATURES, synthetic_credit_data
from .explain import explain_decision
from .global_importance import permutation_feature_importance
from .model import CreditRiskModel
from .report import render_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explainable credit-risk reference project")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)

    frame = synthetic_credit_data(args.rows, args.seed)
    model, metrics = CreditRiskModel.fit(frame)
    test = frame.iloc[int(len(frame) * 0.8) :]
    importance = permutation_feature_importance(model, test, repeats=5, seed=args.seed)
    example = test.sort_values("debt_ratio", ascending=False).drop(columns="default").iloc[[0]][FEATURES]
    explanation = explain_decision(model, example)
    counterfactual = find_actionable_counterfactual(model, example).as_dict()
    result = {
        "metrics": metrics,
        "global_importance": importance,
        "example": {key: (value.item() if hasattr(value, "item") else value) for key, value in example.iloc[0].to_dict().items()},
        "explanation": explanation,
        "counterfactual": counterfactual,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    render_report(result, args.output / "report.html")
    model.save(str(args.output / "credit_risk.joblib"))
    print(json.dumps({"metrics": metrics, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
