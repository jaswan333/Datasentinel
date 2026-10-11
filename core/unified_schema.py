"""
DataSentinel - Unified Audit Schema (Milestone 8)
Defines the standard strongly-typed data contract for multi-modal dataset audits
across both Tabular and Deep Learning Computer Vision pipelines.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field, ConfigDict


class SignalDetail(BaseModel):
    """Metadata and distribution statistics for an individual evidence signal."""
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Signal identifier")
    category: str = Field(..., description="Diagnostic domain or evidence category")
    weight: float = Field(..., ge=0.0, description="Configured policy weight")
    status: Literal["AVAILABLE", "UNAVAILABLE", "ERROR"] = Field(
        default="AVAILABLE",
        description="Whether this signal was extracted or unavailable"
    )
    mean_risk: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Mean normalized risk score in [0.0, 1.0]"
    )
    high_risk_fraction: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of samples with high suspicion (>= 0.70)"
    )
    description: str = Field(default="", description="Technical rationale for this signal")


class SuspiciousSample(BaseModel):
    """Sample-level risk attribution and evidence record."""
    model_config = ConfigDict(extra="ignore")

    sample_id: Union[int, str] = Field(..., description="Unique sample identifier or index")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Normalized risk index (0-100)")
    risk_tier: Literal["HIGH", "MEDIUM", "LOW", "INSUFFICIENT_EVIDENCE"] = Field(
        ...,
        description="Actionable suspicion tier"
    )
    contributing_signals: Dict[str, float] = Field(
        default_factory=dict,
        description="Raw or normalized risk scores per available signal"
    )
    recommended_action: str = Field(
        default="HUMAN_REVIEW",
        description="Actionable policy decision"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (assigned label, predicted label, group_id, etc.)"
    )


class QuarantineManifest(BaseModel):
    """Auditable quarantine manifest requiring human confirmation before destructive purging."""
    model_config = ConfigDict(extra="ignore")

    total_candidate_samples: int = Field(..., description="Total candidate pool evaluated (e.g. 1200)")
    quarantined_count: int = Field(..., description="Number of samples isolated in quarantine")
    quarantine_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of pool quarantined")
    operating_threshold: float = Field(..., description="Decision threshold applied")
    threshold_policy: str = Field(
        default="max recall under FPR_val <= 0.05",
        description="Policy rule used to select threshold"
    )
    requires_human_review: bool = Field(
        default=True,
        description="Guarantees non-destructive screening requiring human sign-off"
    )
    confirmation_status: Literal["PENDING_OPERATOR_REVIEW", "CONFIRMED", "REJECTED"] = Field(
        default="PENDING_OPERATOR_REVIEW",
        description="Operator review sign-off status"
    )
    operator_notes: Optional[str] = Field(
        default=None,
        description="Operator review notes"
    )
    confirmed_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp of operator confirmation"
    )
    quarantined_sample_ids: List[Union[int, str]] = Field(
        default_factory=list,
        description="List of quarantined sample IDs"
    )


class ModelUtilityComparison(BaseModel):
    """Model performance benchmarks before and after dataset screening."""
    model_config = ConfigDict(extra="ignore")

    evaluation_dataset: str = Field(..., description="Description of the pristine evaluation benchmark")
    clean_reference_accuracy: Optional[float] = Field(default=None, description="Accuracy of model trained on clean data")
    victim_clean_accuracy: Optional[float] = Field(default=None, description="Accuracy of model trained on poisoned data")
    screened_clean_accuracy: Optional[float] = Field(default=None, description="Accuracy of model retrained on screened data")
    clean_accuracy_delta_pp: Optional[float] = Field(default=None, description="Change in accuracy from victim to screened")
    victim_asr: Optional[float] = Field(default=None, description="Attack Success Rate of compromised victim model")
    screened_asr: Optional[float] = Field(default=None, description="Attack Success Rate of screened retrained model")
    asr_reduction_pp: Optional[float] = Field(default=None, description="ASR reduction achieved by screening (pp)")
    clean_accuracy_pre: Optional[float] = Field(default=None, description="Pre-quarantine clean accuracy")
    clean_accuracy_post: Optional[float] = Field(default=None, description="Post-quarantine clean accuracy")
    asr_pre: Optional[float] = Field(default=None, description="Pre-quarantine ASR")
    asr_post: Optional[float] = Field(default=None, description="Post-quarantine ASR")
    evaluation_notes: str = Field(default="", description="Fair evaluation notes (architecture, seeds, splits)")

    def model_post_init(self, __context: Any) -> None:
        if self.clean_accuracy_pre is None:
            self.clean_accuracy_pre = self.victim_clean_accuracy
        if self.clean_accuracy_post is None:
            self.clean_accuracy_post = self.screened_clean_accuracy
        if self.asr_pre is None:
            self.asr_pre = self.victim_asr
        if self.asr_post is None:
            self.asr_post = self.screened_asr


class ThreatLimitation(BaseModel):
    """Explicit disclosure of an unresolved threat or detector boundary condition."""
    model_config = ConfigDict(extra="ignore")

    threat_family: str = Field(..., description="Target threat family (e.g. clean_label_manipulation)")
    status: str = Field(
        default="UNRESOLVED_LIMITATION",
        description="Status of defense against this threat"
    )
    observed_recall: Optional[str] = Field(
        default=None,
        description="Empirical recall achieved on held-out test split (e.g. '0/12 (0.0%)')"
    )
    empirical_recall: Optional[float] = Field(
        default=0.0,
        description="Empirical recall float representation"
    )
    operating_fpr: Optional[str] = Field(
        default=None,
        description="Test false positive rate under validation constraint"
    )
    empirical_fpr: Optional[float] = Field(
        default=0.0307,
        description="Empirical FPR float representation"
    )
    statement: str = Field(..., description="Transparent limitation statement")
    limitation_statement: Optional[str] = Field(
        default=None,
        description="Alias for limitation statement"
    )
    mitigation_recommendation: str = Field(
        default="",
        description="Recommended empirical mitigation (e.g. Mixup defensive training)"
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.limitation_statement:
            self.limitation_statement = self.statement


class UnifiedAuditResult(BaseModel):
    """Master Unified Audit Schema across Tabular and Computer Vision pipelines."""
    model_config = ConfigDict(extra="ignore")

    # 1. Dataset & Audit Metadata
    audit_id: str = Field(..., description="Unique audit run identifier")
    dataset_name: str = Field(..., description="Name of the audited dataset or file")
    modality: Literal["TABULAR", "IMAGE"] = Field(..., description="Dataset modality")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of audit generation"
    )
    dataset_fingerprint: str = Field(default="", description="SHA256 hash or data fingerprint")
    n_rows: int = Field(..., ge=0, description="Total sample rows in dataset")
    n_columns: int = Field(..., ge=0, description="Total features or channels in dataset")

    # 2. Reproducibility & Versioning
    detector_version: str = Field(default="DataSentinel-v1.0", description="Detector engine version")
    policy_version: str = Field(default="1.0.0-evidence-weighted", description="Scoring policy version")

    # 3. Evidence-Based Risk Score & Trust Index (0-100 evidence index, NOT probability)
    risk_score: float = Field(..., ge=0.0, le=100.0, description="0-100 evidence-weighted risk index")
    trust_score: float = Field(..., ge=0.0, le=100.0, description="100.0 - risk_score")
    risk_level: Literal["LOW_RISK", "MODERATE_RISK", "ELEVATED_RISK", "CRITICAL_RISK"] = Field(
        ...,
        description="Categorical risk tier"
    )
    overall_status: Literal["AUDITED_SECURE", "TRUSTED", "SUSPICIOUS", "POTENTIALLY_COMPROMISED", "SEVERELY_POISONED"] = Field(
        default="AUDITED_SECURE",
        description="Overall pipeline recommendation status"
    )
    evidence_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of expected signal weight available"
    )

    # 4. Signals & Threat Detection
    primary_threat_family: str = Field(default="unknown", description="Primary detected threat family")
    available_signals: List[str] = Field(default_factory=list, description="List of successfully computed signals")
    missing_signals: List[str] = Field(default_factory=list, description="Signals marked unavailable")
    signal_breakdown: Dict[str, SignalDetail] = Field(default_factory=dict, description="Detailed signal breakdown")

    # 5. Quarantine & Screening
    quarantine_manifest: QuarantineManifest = Field(..., description="Quarantine manifest")

    # 6. Model Utility Comparison (Before vs After)
    model_utility: Optional[ModelUtilityComparison] = Field(
        default=None,
        description="Utility benchmarks on pristine evaluation data"
    )

    # 7. Explicit Threat Limitations (Mandatory for clean-label)
    unresolved_threat_limitations: List[ThreatLimitation] = Field(
        default_factory=list,
        description="Explicit disclosure of known defense boundaries"
    )

    # 8. Actionable Guidance & Inspection
    recommended_action: str = Field(default="APPROVE_FOR_PRODUCTION", description="Actionable next step")
    sample_results: List[SuspiciousSample] = Field(
        default_factory=list,
        description="Top suspicious records for human review"
    )

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Backwards-compatible export mapping to legacy UI and agent formats."""
        res = self.model_dump()
        # Aliases expected by legacy tools and UI components
        res["dataset_id"] = self.audit_id
        res["dataset_risk_score"] = self.risk_score
        res["dataset_trust_score"] = self.trust_score
        res["dataset_risk_level"] = self.risk_level
        res["high_risk_quarantine"] = self.quarantine_manifest.quarantined_count
        res["high_risk_samples"] = self.quarantine_manifest.quarantined_count
        res["status"] = self.overall_status
        res["mode"] = self.modality
        res["threat_limitations"] = res.get("unresolved_threat_limitations", [])
        return res
