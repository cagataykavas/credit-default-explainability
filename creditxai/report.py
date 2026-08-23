from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def render_report(result: dict[str, Any], output: str | Path) -> Path:
    metrics = result["metrics"]
    explanation = result["explanation"]
    cf = result["counterfactual"]
    importance_rows = "".join(
        f"<tr><td>{html.escape(str(item['feature']))}</td><td>{float(item['importance_mean']):.4f}</td><td>{float(item['importance_std']):.4f}</td></tr>"
        for item in result["global_importance"]
    )
    local_rows = "".join(
        f"<tr><td>{html.escape(str(item['feature']))}</td><td>{html.escape(str(item['observed']))}</td><td>{float(item['probability_delta']):+.4f}</td><td>{html.escape(str(item['direction']))}</td></tr>"
        for item in explanation["contributions"]
    )
    changes = "<br>".join(f"{html.escape(k)} → {html.escape(str(v))}" for k, v in cf["changed_features"].items()) or "No change needed"
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Credit XAI Report</title><style>
body{{background:#0a1020;color:#edf3ff;font-family:Inter,system-ui,sans-serif;margin:0;padding:32px}}main{{max-width:1100px;margin:auto}}.muted{{color:#9ba9c1}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}}.card{{background:#121b30;border:1px solid #293955;border-radius:14px;padding:17px}}.big{{font-size:26px;font-weight:800}}
table{{width:100%;border-collapse:collapse;background:#121b30;margin-bottom:28px}}th,td{{padding:10px;border-bottom:1px solid #293955;text-align:left}}th{{color:#a7bdff}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><p class="muted">Synthetic credit-risk explainability reference</p><h1>Credit Default Explainability</h1>
<div class="cards"><div class="card"><div class="big">{metrics['roc_auc']:.3f}</div><div>ROC-AUC</div></div><div class="card"><div class="big">{metrics['brier_score']:.3f}</div><div>Brier score</div></div><div class="card"><div class="big">{explanation['default_probability']:.1%}</div><div>Example probability</div></div><div class="card"><div class="big">{cf['probability_after']:.1%}</div><div>Counterfactual probability</div></div></div>
<h2>Decision explanation</h2><p><strong>{html.escape(str(explanation['decision']))}</strong> · reasons: {html.escape(', '.join(explanation['reason_codes']))}</p>
<table><thead><tr><th>Feature</th><th>Observed</th><th>Probability delta</th><th>Direction</th></tr></thead><tbody>{local_rows}</tbody></table>
<h2>Global permutation importance</h2><table><thead><tr><th>Feature</th><th>Mean importance</th><th>Std</th></tr></thead><tbody>{importance_rows}</tbody></table>
<h2>Illustrative actionable counterfactual</h2><p>{changes}</p><p class="muted">{html.escape(str(explanation['caveat']))}</p>
</main></body></html>"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
