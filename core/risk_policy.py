"""
DataSentinel - Evidence-Based Risk Policy Engine (Milestone 8)
Calculates evidence-weighted risk indices, tracks signal coverage,
and attaches explicit unresolved threat limitations without inventing risk signals.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import pandas as pd

from core.unified_schema import (
    SignalDetail,
    SuspiciousSample,
    QuarantineManifest,
    ThreatLimitation
)


POLICY_VERSION = "1.0.0-evidence-weighted"

# Standard versioned weights for each diagnostic domain
TABULAR_SUPERVISED_WEIGHTS: Dict[str, Tuple[str, float, str]] = {
    # signal_name: (category, weight, description)
    "cleanlab_risk": ("label_consistency", 0.20, "Confident learning out-of-fold label inconsistency"),
    "label_disagreement": ("label_consistency", 0.15, "Supervised classifier prediction disagreement"),
    "prediction_entropy": ("uncertainty", 0.15, "Model prediction entropy / probability margin deficit"),
    "isolation_forest_risk": ("feature_anomaly", 0.15, "Tree-based recursive isolation path length anomaly"),
    "lof_risk": ("density_anomaly", 0.10, "Local Outlier Factor neighborhood density deficit"),
    "exact_hash_risk": ("image_integrity", 0.05, "Exact MD5 feature hash collision"),
    "near_duplicate_similarity": ("image_integrity", 0.05, "Cosine / Jaccard near-duplicate similarity"),
    "class_centroid_distance_ratio": ("representation", 0.05, "Distance ratio between own and competing class centroids"),
    "knn_label_disagreement": ("representation", 0.05, "K-nearest neighbors voting label conflict"),
    "trigger_rarity_risk": ("rarity", 0.05, "Rare category or extreme outlier trigger signature")
}

TABULAR_UNSUPERVISED_WEIGHTS: Dict[str, Tuple[str, float, str]] = {
    "isolation_forest_risk": ("feature_anomaly", 0.40, "Tree-based recursive isolation path length anomaly"),
    "lof_risk": ("density_anomaly", 0.30, "Local Outlier Factor neighborhood density deficit"),
    "exact_hash_risk": ("image_integrity", 0.15, "Exact MD5 feature hash collision"),
    "near_duplicate_similarity": ("image_integrity", 0.10, "Near-duplicate feature similarity"),
    "trigger_rarity_risk": ("rarity", 0.05, "Rare category combination or extreme feature rarity")
}

IMAGE_DL_WEIGHTS: Dict[str, Tuple[str, float, str]] = {
    # Group A: Label & Consistency
    "training_loss_risk": ("label_consistency", 0.08, "Penultimate cross-entropy training loss percentile"),
    "prediction_entropy_risk": ("label_consistency", 0.07, "Softmax prediction entropy uncertainty percentile"),
    "prediction_margin_risk": ("label_consistency", 0.08, "Margin deficit between top-1 and competing class"),
    "assigned_confidence_risk": ("label_consistency", 0.07, "Confidence deficit on assigned class label"),
    "oof_label_disagreement": ("label_consistency", 0.07, "Out-of-fold probe prediction disagreement"),
    # Group B: Deep Representations
    "knn_embedding_risk": ("deep_representation", 0.08, "K-nearest neighbor distance in 512D ResNet18 latent space"),
    "lof_embedding_risk": ("deep_representation", 0.08, "Local Outlier Factor in 512D ResNet18 latent space"),
    "centroid_distance_risk": ("deep_representation", 0.07, "Euclidean ratio to assigned class feature centroid"),
    "activation_cluster_risk": ("deep_representation", 0.08, "K-Means cluster silhouette deficit in 10D PCA space"),
    "spectral_signature_risk": ("deep_representation", 0.08, "Top-singular vector projection outlier score"),
    # Group C: Image Integrity
    "exact_duplicate_risk": ("image_integrity", 0.06, "MD5 raw pixel byte collision"),
    "perceptual_duplicate_risk": ("image_integrity", 0.06, "64-bit dHash perceptual near-duplicate collision"),
    "image_statistics_risk": ("image_integrity", 0.06, "Isolation Forest on RGB color moments and Laplacian variance"),
    # Group D: Model Behavior
    "local_patch_sensitivity_risk": ("model_behavior", 0.07, "Spatial prediction sensitivity across 5 occlusion masks")
}


def get_default_unresolved_limitations(modality: str = "IMAGE") -> List[ThreatLimitation]:
    """Returns mandatory transparent disclosures of unresolved detector limitations."""
    limitations = []
    if modality.upper() in {"IMAGE", "MULTI_MODAL"}:
        limitations.append(
            ThreatLimitation(
                threat_family="clean_label_manipulation",
                status="UNRESOLVED_LIMITATION",
                observed_recall="0 / 12 (0.0% on held-out test split under FPR <= 0.05)",
                operating_fpr="3.07% test FPR (calibrated on Val <= 5%)",
                statement=(
                    "Clean-label centroid manipulation remains an unresolved detector limitation "
                    "under operational false-positive ceilings (FPR <= 5%). While 2D FFT spectral analysis "
                    "identifies statistically significant high-frequency spatial attenuation (p < 0.001, "
                    "Cohen's d = 0.6169), natural image frequency variance in distribution tails prevents "
                    "reliable post-hoc individual-sample detection."
                ),
                mitigation_recommendation=(
                    "Apply defensive training (Mixup alpha=0.20) as an empirical training-time mitigation, "
                    "and enforce validation FPR ceilings to prevent downstream utility collapse."
                )
            )
        )
    return limitations


def calculate_evidence_weighted_risk(
    evidence_df: pd.DataFrame,
    modality: str = "TABULAR",
    is_supervised: bool = True,
    high_risk_threshold: float = 0.70
) -> Tuple[float, float, str, str, Dict[str, SignalDetail], List[SuspiciousSample]]:
    """
    Computes evidence-weighted risk index R and evidence coverage C according to:
        R = 100 * (sum_{i in A} w_i * r_i) / (sum_{i in A} w_i)
        C = (sum_{i in A} w_i) / (sum_{i in All} w_i)
    
    Parameters:
        evidence_df: DataFrame with evidence signals (normalized in [0, 1]).
        modality: "TABULAR" or "IMAGE".
        is_supervised: Whether labels were available during audit.
        high_risk_threshold: Continuous probability cutoff for HIGH tier.

    Returns:
        (risk_score, evidence_coverage, risk_level, overall_status, signal_breakdown, top_suspicious_samples)
    """
    if modality.upper() == "IMAGE":
        expected_weights = IMAGE_DL_WEIGHTS
    elif is_supervised:
        expected_weights = TABULAR_SUPERVISED_WEIGHTS
    else:
        expected_weights = TABULAR_UNSUPERVISED_WEIGHTS

    total_expected_weight = sum(w for _, w, _ in expected_weights.values())
    n_samples = len(evidence_df)

    if n_samples == 0:
        return 0.0, 0.0, "LOW_RISK", "AUDITED_SECURE", {}, []

    # Map available signals
    available_signals: Set[str] = set()
    signal_breakdown: Dict[str, SignalDetail] = {}

    for sig_name, (cat, w, desc) in expected_weights.items():
        if sig_name in evidence_df.columns:
            series = pd.to_numeric(evidence_df[sig_name], errors="coerce")
            valid_vals = series.dropna()
            if len(valid_vals) > 0:
                available_signals.add(sig_name)
                mean_r = float(np.clip(valid_vals.mean(), 0.0, 1.0))
                high_frac = float((valid_vals >= high_risk_threshold).mean())
                signal_breakdown[sig_name] = SignalDetail(
                    name=sig_name,
                    category=cat,
                    weight=w,
                    status="AVAILABLE",
                    mean_risk=round(mean_r, 4),
                    high_risk_fraction=round(high_frac, 4),
                    description=desc
                )
                continue

        signal_breakdown[sig_name] = SignalDetail(
            name=sig_name,
            category=cat,
            weight=w,
            status="UNAVAILABLE",
            mean_risk=None,
            high_risk_fraction=None,
            description=desc
        )

    # Calculate evidence coverage
    available_weight_sum = sum(expected_weights[s][1] for s in available_signals)
    coverage = float(available_weight_sum / total_expected_weight) if total_expected_weight > 0 else 0.0

    # Calculate per-sample evidence-weighted risk index
    if available_weight_sum > 0:
        sample_risks = np.zeros(n_samples, dtype=float)
        for sig_name in available_signals:
            w = expected_weights[sig_name][1]
            s_vals = pd.to_numeric(evidence_df[sig_name], errors="coerce").fillna(0.0).to_numpy()
            sample_risks += np.clip(s_vals, 0.0, 1.0) * w
        sample_risks = (sample_risks / available_weight_sum) * 100.0
    else:
        sample_risks = np.zeros(n_samples, dtype=float)

    # Overall dataset risk index: mean of sample risks + high risk rate penalty
    mean_risk = float(np.mean(sample_risks))
    high_rate = float(np.mean(sample_risks >= (high_risk_threshold * 100.0)))
    dataset_risk_score = round(float(mean_risk * 0.6 + high_rate * 100.0 * 0.4), 2)
    dataset_risk_score = max(0.0, min(100.0, dataset_risk_score))

    # Determine risk level & status
    if dataset_risk_score < 30.0:
        risk_level = "LOW_RISK"
        overall_status = "AUDITED_SECURE"
    elif dataset_risk_score < 50.0:
        risk_level = "MODERATE_RISK"
        overall_status = "SUSPICIOUS"
    elif dataset_risk_score < 70.0:
        risk_level = "ELEVATED_RISK"
        overall_status = "POTENTIALLY_COMPROMISED"
    else:
        risk_level = "CRITICAL_RISK"
        overall_status = "SEVERELY_POISONED"

    # Extract top suspicious samples
    sample_ids = evidence_df["sample_id"].values if "sample_id" in evidence_df.columns else np.arange(n_samples)
    top_indices = np.argsort(sample_risks)[::-1][:min(50, n_samples)]

    suspicious_samples: List[SuspiciousSample] = []
    for idx in top_indices:
        r_score = float(sample_risks[idx])
        tier = "HIGH" if r_score >= 70.0 else ("MEDIUM" if r_score >= 40.0 else "LOW")
        action = "QUARANTINE_REVIEW" if tier == "HIGH" else ("FLAG_SECONDARY" if tier == "MEDIUM" else "CLEAR")

        contributing = {}
        for s in available_signals:
            val = evidence_df.loc[idx, s] if idx in evidence_df.index else None
            if pd.notna(val):
                contributing[s] = round(float(val), 4)

        metadata = {}
        for meta_col in ["label", "original_label", "assigned_label", "group_id", "partition"]:
            if meta_col in evidence_df.columns:
                metadata[meta_col] = str(evidence_df.loc[idx, meta_col])

        suspicious_samples.append(
            SuspiciousSample(
                sample_id=str(sample_ids[idx]),
                risk_score=round(r_score, 2),
                risk_tier=tier,
                contributing_signals=contributing,
                recommended_action=action,
                metadata=metadata
            )
        )

    return dataset_risk_score, round(coverage, 4), risk_level, overall_status, signal_breakdown, suspicious_samples
