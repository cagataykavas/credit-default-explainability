# Fail-closed scoring admission

The offline lineage audit proves that an explanation artifact is internally consistent. A serving
boundary must also prove that the process answering a request is running the exact fitted model and
preprocessing state authorized by the promotion receipt. Otherwise a valid report can sit beside an
unrelated or silently replaced model.

`creditxai.scoring_admission` closes that gap for the reference API:

1. fingerprint the fitted preprocessing state, calibrated classifier coefficients and calibrators,
   and explainer baselines;
2. rebuild the ordered raw-to-transformed feature contract from the running model;
3. require an authenticated promotion artifact for that exact model, pipeline, and contract;
4. bind each request's local attributions to the runtime contract;
5. recheck audience, signature, validity window, lineage digests, feature mappings, and attribution
   directions before releasing the explanation; and
6. return only hashes, bounded counts, and a hashed key reference as admission evidence.

The fitted-model fingerprint is semantic rather than a hash of Joblib or pickle bytes. Serialization
bytes can change across Python or dependency builds without changing model behavior. The fingerprint
instead commits to the state used by prediction and explanation.

## Deployment inputs

Create the signed promotion artifact outside the scoring service using the workflow documented in
[`explanation-schema-lineage.md`](explanation-schema-lineage.md). The artifact's contract must use
`fingerprint_credit_model(model)` as its `model_digest`. Mount the JSON read-only and provide the
verification key through a secret-backed environment variable:

```bash
export CREDIT_XAI_PROMOTION_ARTIFACT=/run/credit-xai/promotion.json
export CREDIT_XAI_PROMOTION_AUDIENCE=credit-xai-production
export CREDIT_XAI_PROMOTION_KEY_ID=credit-prod-2026-11
export CREDIT_XAI_PROMOTION_KEY_ENV=CREDIT_XAI_PROMOTION_KEY
export CREDIT_XAI_PROMOTION_KEY='replace-with-secret-manager-injected-material'

uvicorn app.api:app --host 0.0.0.0 --port 8000
```

`CREDIT_XAI_PROMOTION_KEY_ENV` names the variable containing the secret. Key material is never placed
in a command-line argument, artifact, response, or log produced by this implementation.

## Readiness and failure behavior

`GET /health` reports process liveness. `GET /ready` runs the complete admission path against the
model's synthetic baseline, without customer inputs. `POST /score` uses the same guard for the actual
request.

The readiness and scoring routes return HTTP `503` when:

- configuration is missing or partial;
- the promotion file is malformed or has no receipt;
- the contract does not match the running fitted model;
- the receipt signature, audience, or lineage digests do not verify;
- the receipt is expired, future-dated, or valid for longer than policy permits; or
- the generated attribution fails the strict feature-lineage audit.

Rejection details stay inside the admission boundary. Clients receive a stable generic failure, so a
misconfigured key ring or receipt does not become an oracle for deployment secrets.

## Trust boundary and limitations

This gate prevents an unapproved fitted model or malformed explanation from being served by a
cooperating process. It does not make HMAC publicly verifiable, protect a secret from a compromised
process, establish signer identity, or prevent a privileged operator from replacing both the model
and its configured evidence. Production promotion should use KMS/HSM-backed signing and workload
identity rather than a shared environment secret.

The repository still fits deterministic synthetic demo data in process. A real deployment should
load an immutable reviewed model artifact, verify its content identity before deserialization, and
keep the promotion file and key under separately controlled release channels. Passing admission does
not establish attribution faithfulness, fairness, causal meaning, calibration, or legal suitability
of a reason code.
