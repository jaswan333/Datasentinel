/**
 * DataSentinel - Unified Audit & Machine Learning Domain Types
 * Strongly typed contracts matching the FastAPI backend schemas.
 */

export interface ColumnStat {
  name: string;
  dtype: string;
  unique_count: number;
  is_suggested_cat: boolean;
  is_suggested_target: boolean;
}

export interface InspectColumnsResponse {
  filename: string;
  columns: string[];
  suggested_target: string | null;
  suggested_categoricals: string[];
  column_stats: ColumnStat[];
}

export type SignalStatus = "AVAILABLE" | "UNAVAILABLE" | "ERROR";

export interface SignalDetail {
  name: string;
  category: string;
  weight: number;
  status: SignalStatus;
  mean_risk: number | null;
  high_risk_fraction: number | null;
  description: string;
}

export type RiskTier = "HIGH" | "MEDIUM" | "LOW" | "INSUFFICIENT_EVIDENCE";

export interface SuspiciousSample {
  sample_id: string | number;
  risk_score: number;
  risk_tier: RiskTier;
  contributing_signals?: Record<string, number>;
  recommended_action?: string;
  metadata?: Record<string, any>;
}

export type QuarantineConfirmationStatus = "PENDING_OPERATOR_REVIEW" | "CONFIRMED" | "REJECTED";

export interface QuarantineManifest {
  total_candidate_samples: number;
  quarantined_count: number;
  quarantine_rate: number;
  operating_threshold: number;
  threshold_policy: string;
  requires_human_review: boolean;
  confirmation_status: QuarantineConfirmationStatus;
  operator_notes?: string | null;
  confirmed_at?: string | null;
  quarantined_sample_ids: (string | number)[];
}

export interface ModelUtilityComparison {
  evaluation_dataset: string;
  clean_reference_accuracy?: number | null;
  victim_clean_accuracy?: number | null;
  screened_clean_accuracy?: number | null;
  clean_accuracy_delta_pp?: number | null;
  victim_asr?: number | null;
  screened_asr?: number | null;
  asr_reduction_pp?: number | null;
  clean_accuracy_pre?: number | null;
  clean_accuracy_post?: number | null;
  asr_pre?: number | null;
  asr_post?: number | null;
  evaluation_notes?: string;
}

export interface ThreatLimitation {
  threat_family: string;
  status: string;
  observed_recall?: string | null;
  empirical_recall?: number | null;
  operating_fpr?: string | null;
  empirical_fpr?: number | null;
  statement: string;
  limitation_statement?: string | null;
  mitigation_recommendation?: string;
}

export type RiskLevel = "LOW_RISK" | "MODERATE_RISK" | "ELEVATED_RISK" | "CRITICAL_RISK";
export type PipelineStatus =
  | "AUDITED_SECURE"
  | "TRUSTED"
  | "SUSPICIOUS"
  | "POTENTIALLY_COMPROMISED"
  | "SEVERELY_POISONED"
  | "QUARANTINE_CONFIRMED";

export interface UnifiedAuditResult {
  audit_id: string;
  dataset_name: string;
  modality: "TABULAR" | "IMAGE";
  created_at?: string;
  dataset_fingerprint?: string;
  n_rows: number;
  n_columns: number;
  detector_version?: string;
  policy_version?: string;
  risk_score: number;
  trust_score: number;
  risk_level: RiskLevel;
  overall_status: PipelineStatus;
  evidence_coverage: number;
  primary_threat_family?: string;
  available_signals: string[];
  missing_signals: string[];
  signal_breakdown: Record<string, SignalDetail>;
  quarantine_manifest: QuarantineManifest;
  model_utility?: ModelUtilityComparison | null;
  unresolved_threat_limitations: ThreatLimitation[];
  recommended_action: string;
  sample_results: SuspiciousSample[];

  // Backwards-compatible legacy fields
  dataset_risk_score?: number;
  dataset_trust_score?: number;
  dataset_risk_level?: RiskLevel;
  high_risk_quarantine?: number;
  high_risk_samples?: number;
  status?: PipelineStatus;
  mode?: string;
}

export interface AuditResponse {
  dataset_id: string;
  filename: string;
  modality?: string;
  audit: UnifiedAuditResult;
  is_demo?: boolean;
}

export interface QuarantineDetailsResponse {
  dataset_id: string;
  manifest: QuarantineManifest;
  confirmation_status: QuarantineConfirmationStatus;
  quarantined_count: number;
  operating_threshold: number;
  threshold_policy: string;
  operator_notes: string | null;
  quarantined_samples: SuspiciousSample[];
}

export interface QuarantineConfirmResponse {
  status: string;
  dataset_id: string;
  quarantined_samples: number;
  quarantined_count: number;
  quarantine_status: "CONFIRMED";
  operator_notes: string;
  message: string;
}

export interface AgentAskResponse {
  dataset_id: string;
  question: string;
  answer: string;
  audit_agent?: any;
  investigation_agent?: any;
}

export interface ImageSamplePreview {
  sample_id: string;
  filename: string;
  width: number;
  height: number;
  channels: number;
  label: string;
  preview_url: string;
}

export interface ImageDatasetInspectionResponse {
  dataset_id: string;
  dataset_name: string;
  modality: "IMAGE";
  total_images: number;
  rejected_count: number;
  uncompressed_size_mb: number;
  formats: Record<string, number>;
  dimensions_summary: {
    min: string;
    max: string;
    common: string;
  };
  labels_distribution: Record<string, number>;
  has_labels: boolean;
  rejected_files: Array<{ filename: string; reason: string }>;
  sample_previews: ImageSamplePreview[];
}

export interface AuditHistoryItem {
  dataset_id: string;
  dataset_name: string;
  modality: "TABULAR" | "IMAGE";
  created_at: string;
  risk_score: number;
  trust_score: number;
  risk_level: RiskLevel;
  overall_status: string;
  evidence_coverage: number;
  quarantined_count: number;
  total_candidate_samples: number;
  confirmation_status: QuarantineConfirmationStatus;
  has_unresolved_limitations: boolean;
}
