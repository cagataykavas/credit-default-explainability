from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from .data import FEATURES
from .explain import explain_decision, local_perturbation_attribution
from .explanation_contract import (
    ContractAuditReport,
    ContractPolicy,
    ExplanationContractArtifact,
    ExplanationContractError,
    ModelFeatureContract,
    PromotionReceipt,
    audit_explanation_contract,
    bind_explanation,
    build_credit_feature_contract,
    fingerprint_credit_model,
    load_artifact,
    parse_artifact,
)
from .model import CreditRiskModel


class ScoringAdmissionError(RuntimeError):
    """Raised when the scoring service cannot safely release an explanation."""

    def __init__(self, message: str, *, reasons: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.reasons = reasons


@dataclass(frozen=True)
class AdmittedScore:
    explanation: dict[str, Any]
    audit: ContractAuditReport

    def admission_summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.audit.schema_version,
            "accepted": self.audit.accepted,
            "evidence_digest": self.audit.evidence_digest,
            "model_digest": self.audit.model_digest,
            "pipeline_digest": self.audit.pipeline_digest,
            "contract_digest": self.audit.contract_digest,
            "raw_feature_count": self.audit.raw_feature_count,
            "transformed_feature_count": self.audit.transformed_feature_count,
            "attribution_count": self.audit.attribution_count,
            "promotion_verified": self.audit.promotion_verified,
            "promotion_key_reference": self.audit.promotion_key_reference,
        }


@dataclass(frozen=True)
class ScoringAdmissionGuard:
    """Bind one running model to an authenticated explanation promotion."""

    model: CreditRiskModel
    contract: ModelFeatureContract
    promotion: PromotionReceipt
    policy: ContractPolicy
    trusted_promotion_keys: Mapping[str, bytes]

    def __post_init__(self) -> None:
        runtime_contract = build_credit_feature_contract(
            self.model,
            model_digest=fingerprint_credit_model(self.model),
        )
        if self.contract != runtime_contract:
            raise ScoringAdmissionError(
                "promotion contract does not match the running model"
            )
        if not self.policy.require_signed_promotion:
            raise ScoringAdmissionError(
                "scoring admission must require signed promotion"
            )
        if (
            self.promotion.model_digest != runtime_contract.model_digest
            or self.promotion.pipeline_digest != runtime_contract.pipeline_digest
            or self.promotion.contract_digest != runtime_contract.contract_digest
        ):
            raise ScoringAdmissionError(
                "promotion receipt does not authorize the running model contract"
            )
        object.__setattr__(
            self,
            "trusted_promotion_keys",
            MappingProxyType(dict(self.trusted_promotion_keys)),
        )

    @classmethod
    def from_promotion_artifact(
        cls,
        model: CreditRiskModel,
        artifact: ExplanationContractArtifact | object,
        *,
        audience: str,
        trusted_promotion_keys: Mapping[str, bytes],
        max_future_skew_seconds: int = 60,
        max_promotion_lifetime_seconds: int = 2_592_000,
    ) -> ScoringAdmissionGuard:
        parsed = (
            artifact
            if isinstance(artifact, ExplanationContractArtifact)
            else parse_artifact(artifact)
        )
        if parsed.promotion is None:
            raise ScoringAdmissionError("promotion artifact has no signed receipt")

        return cls(
            model=model,
            contract=parsed.contract,
            promotion=parsed.promotion,
            policy=ContractPolicy(
                max_age_seconds=60,
                max_future_skew_seconds=max_future_skew_seconds,
                require_signed_promotion=True,
                required_promotion_audience=audience,
                max_promotion_lifetime_seconds=max_promotion_lifetime_seconds,
            ),
            trusted_promotion_keys=dict(trusted_promotion_keys),
        )

    def score(self, row: pd.DataFrame, *, now: datetime) -> AdmittedScore:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ScoringAdmissionError("admission timestamp must be timezone-aware")
        contributions = local_perturbation_attribution(self.model, row)
        artifact = bind_explanation(
            self.contract,
            contributions,
            generated_at=now.astimezone(UTC),
            promotion=self.promotion,
        )
        try:
            audit = audit_explanation_contract(
                artifact,
                now=now.astimezone(UTC),
                policy=self.policy,
                trusted_promotion_keys=self.trusted_promotion_keys,
            )
        except ExplanationContractError as error:
            raise ScoringAdmissionError("explanation admission is malformed") from error
        if not audit.accepted:
            raise ScoringAdmissionError(
                "explanation admission rejected",
                reasons=audit.reasons,
            )
        return AdmittedScore(
            explanation=explain_decision(
                self.model,
                row,
                contributions=contributions,
            ),
            audit=audit,
        )

    def check_ready(self, *, now: datetime) -> ContractAuditReport:
        """Exercise the full admission path with the model's non-customer baseline."""
        baseline = self.model.baseline_row()
        row = pd.DataFrame([{name: baseline[name] for name in FEATURES}])
        return self.score(row, now=now).audit


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        raise ScoringAdmissionError(f"required admission setting is missing: {name}")
    return value


def load_environment_admission(
    model: CreditRiskModel,
) -> ScoringAdmissionGuard | None:
    """Load an admission guard without placing key material in arguments or files."""
    path_value = os.environ.get("CREDIT_XAI_PROMOTION_ARTIFACT")
    setting_names = (
        "CREDIT_XAI_PROMOTION_ARTIFACT",
        "CREDIT_XAI_PROMOTION_AUDIENCE",
        "CREDIT_XAI_PROMOTION_KEY_ID",
        "CREDIT_XAI_PROMOTION_KEY_ENV",
    )
    configured = [name for name in setting_names if os.environ.get(name)]
    if not configured:
        return None
    if len(configured) != len(setting_names):
        missing = sorted(set(setting_names) - set(configured))
        raise ScoringAdmissionError(
            "scoring admission configuration is incomplete: " + ", ".join(missing)
        )

    audience = _required_environment("CREDIT_XAI_PROMOTION_AUDIENCE")
    key_id = _required_environment("CREDIT_XAI_PROMOTION_KEY_ID")
    key_environment = _required_environment("CREDIT_XAI_PROMOTION_KEY_ENV")
    secret = _required_environment(key_environment).encode("utf-8")
    try:
        artifact = load_artifact(Path(path_value or ""))
    except (ExplanationContractError, OSError) as error:
        raise ScoringAdmissionError("promotion artifact could not be loaded") from error
    return ScoringAdmissionGuard.from_promotion_artifact(
        model,
        artifact,
        audience=audience,
        trusted_promotion_keys={key_id: secret},
    )
