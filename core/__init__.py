"""
DataSentinel Core Modules
Evidence Fusion & Dataset Trust Scoring Pipeline (Milestone 4)
"""

from .evidence_fusion import (
    compute_oof_probabilities,
    compute_cleanlab_risk,
    compute_label_disagreement,
    compute_prediction_entropy,
    compute_isolation_forest_risk,
    compute_lof_risk,
    compute_ecod_risk,
    compute_knn_distance_risk,
    compute_exact_hash_risk,
    compute_near_duplicate_similarity,
    compute_class_centroid_distance_ratio,
    compute_knn_label_disagreement,
    compute_trigger_rarity_risk,
    build_evidence_table,
    train_fusion_model,
    find_optimal_threshold,
    fit_final_fusion_model,
    evaluate_detector_suite,
    evaluate_with_train_val_test_split,
    evaluate_cross_attack_generalization,
    EvidenceFusionPipeline,
    CORE_FEATURE_COLUMNS,
    EXPANDED_FEATURE_COLUMNS,
    EVIDENCE_FEATURE_COLUMNS
)

from .trust_score import (
    classify_sample_risk,
    calculate_dataset_trust_score,
    generate_quarantine_manifest,
    generate_screened_dataset,
    evaluate_screening_recovery,
    generate_dataset_trust_report,
    format_sample_risk_audit
)

from .tabular_schema import (
    inspect_tabular_schema
)

from .tabular_preprocessing import (
    build_tabular_preprocessor,
    fit_transform_tabular
)

from .adaptive_risk_score import (
    normalize_weights,
    calculate_sample_risk,
    calculate_dataset_summary,
    format_dataset_audit_card,
    DEFAULT_SUPERVISED_WEIGHTS,
    DEFAULT_UNSUPERVISED_WEIGHTS
)

from .tabular_auditor import (
    audit_tabular_dataset,
    TabularAuditResult
)

__all__ = [
    "compute_oof_probabilities",
    "compute_cleanlab_risk",
    "compute_label_disagreement",
    "compute_prediction_entropy",
    "compute_isolation_forest_risk",
    "compute_lof_risk",
    "compute_ecod_risk",
    "compute_knn_distance_risk",
    "compute_exact_hash_risk",
    "compute_near_duplicate_similarity",
    "compute_class_centroid_distance_ratio",
    "compute_knn_label_disagreement",
    "compute_trigger_rarity_risk",
    "build_evidence_table",
    "train_fusion_model",
    "find_optimal_threshold",
    "fit_final_fusion_model",
    "evaluate_detector_suite",
    "evaluate_with_train_val_test_split",
    "evaluate_cross_attack_generalization",
    "EvidenceFusionPipeline",
    "CORE_FEATURE_COLUMNS",
    "EXPANDED_FEATURE_COLUMNS",
    "EVIDENCE_FEATURE_COLUMNS",
    "classify_sample_risk",
    "calculate_dataset_trust_score",
    "generate_quarantine_manifest",
    "generate_screened_dataset",
    "evaluate_screening_recovery",
    "generate_dataset_trust_report",
    "format_sample_risk_audit",
    "inspect_tabular_schema",
    "build_tabular_preprocessor",
    "fit_transform_tabular",
    "normalize_weights",
    "calculate_sample_risk",
    "calculate_dataset_summary",
    "format_dataset_audit_card",
    "DEFAULT_SUPERVISED_WEIGHTS",
    "DEFAULT_UNSUPERVISED_WEIGHTS",
    "audit_tabular_dataset",
    "TabularAuditResult"
]

