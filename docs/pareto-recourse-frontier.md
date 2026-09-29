# Pareto recourse frontier

The original demo returns one greedy counterfactual. That is useful for illustrating constraints,
but one path can hide materially different trade-offs: a sparse but larger change, several smaller
changes, or a more costly candidate with a lower model score.

`creditxai.recourse_frontier` performs deterministic best-first search over an explicit action
grid. It returns only feasible candidates that are nondominated across three objectives:

1. normalized action cost;
2. number of changed features; and
3. resulting model probability.

This is a candidate generator for model review, not financial advice or a guarantee that an action
will change a real-world outcome.

## Guardrails

- Only `income`, `debt_ratio`, `late_payments`, and `utilization` are actionable.
- Income may only increase; the other three fields may only decrease.
- Age, employment, housing, account age, unknown fields, no-ops, duplicate values, invalid counts,
  and out-of-domain ratios fail closed before scoring.
- Search uses a model-call budget and a maximum changed-feature count. Low-cost states are explored
  first; an incomplete search is explicitly marked `search_truncated`. If the budget ends before
  finding a candidate, the result says `SEARCH_INCOMPLETE_NO_FEASIBLE_OPTION` rather than claiming
  the supplied grid has no feasible option.
- The returned frontier has a separate result budget and `result_truncated` flag.
- Non-finite/out-of-range predictions and backend failures become stable reason codes without
  exposing exception messages.
- Each option has a deterministic SHA-256 identity derived from its ordered changes.

## Example

```python
from creditxai.recourse_frontier import RecourseSearchPolicy, generate_recourse_frontier

frontier = generate_recourse_frontier(
    model,
    applicant,
    policy=RecourseSearchPolicy(
        target_probability=0.25,
        max_model_calls=300,
        max_changed_features=3,
        max_results=10,
    ),
)

for option in frontier.options:
    print(option.option_id, option.probability, option.normalized_cost, option.changes)
```

The default grid evaluates up to three bounded values per actionable field. Custom `ActionGrid`
instances should come from a versioned domain policy, with reviewed units, direction, feasibility,
and cost scales.

## Interpretation and limitations

Pareto optimality is only relative to the supplied discrete grid, cost scales, model, and search
budget. If `search_truncated` is true, an unexplored state may dominate a returned candidate.
Correlated changes can create implausible records even when each field is individually valid.
Probability reduction is a model response, not a causal promise, approval decision, adverse-action
reason, or recommendation to a person.

Before presentation, production candidates need domain feasibility checks, protected-class and
proxy review, monotonicity testing, uncertainty bounds, human approval, and an immutable record of
the model and policy revisions. The next integration step is to feed every generated option into
the repository's counterfactual recourse audit and retain only inclusion-minimal, policy-admitted
candidates.
