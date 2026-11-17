from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .counterfactual import find_actionable_counterfactual
from .data import FEATURES, synthetic_credit_data
from .explain import explain_decision
from .global_importance import permutation_feature_importance
from .interactions import InteractionPolicy, audit_pairwise_interactions
from .model import CreditRiskModel
from .report import render_report


def _unit_interval(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be finite and in [0, 1]")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explainable credit-risk reference project"
    )
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-pair-interaction", type=_unit_interval, default=0.05)
    parser.add_argument("--fail-on-pair-interaction", action="store_true")
    args = parser.parse_args(argv)

    frame = synthetic_credit_data(args.rows, args.seed)
    model, metrics = CreditRiskModel.fit(frame)
    test = frame.iloc[int(len(frame) * 0.8) :]
    importance = permutation_feature_importance(model, test, repeats=5, seed=args.seed)
    example = (
        test.sort_values("debt_ratio", ascending=False)
        .drop(columns="default")
        .iloc[[0]][FEATURES]
    )
    explanation = explain_decision(model, example)
    interaction_audit = audit_pairwise_interactions(
        model,
        example,
        model_revision=f"synthetic-credit-seed-{args.seed}",
        features=tuple(FEATURES),
        policy=InteractionPolicy(
            max_abs_interaction=args.max_pair_interaction,
            neutral_epsilon=min(1e-8, args.max_pair_interaction),
        ),
    ).to_dict()
    counterfactual = find_actionable_counterfactual(model, example).as_dict()
    result = {
        "metrics": metrics,
        "global_importance": importance,
        "example": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in example.iloc[0].to_dict().items()
        },
        "explanation": explanation,
        "pairwise_interaction_audit": interaction_audit,
        "counterfactual": counterfactual,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    render_report(result, args.output / "report.html")
    model.save(str(args.output / "credit_risk.joblib"))
    print(
        json.dumps(
            {
                "metrics": metrics,
                "pairwise_interaction_accepted": interaction_audit["accepted"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return (
        2 if args.fail_on_pair_interaction and not interaction_audit["accepted"] else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
