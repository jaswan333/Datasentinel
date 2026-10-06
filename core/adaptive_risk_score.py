"""
DataSentinel - Adaptive Evidence & Risk Scoring Engine (Milestone 4.5)
Dynamically adapts to available evidence groups (supervised vs unsupervised),
renormalizes weights over available signals, computes evidence coverage,
and prevents missing tests from artificially distorting risk scores.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd


DEFAULT_SUPERVISED_WEIGHTS: Dict[str, float] = {
    "label_risk": 0.35,
    "uncertainty_risk": 0.15,
    "anomaly_risk": 0.25,
    "duplicate_risk": 0.15,
    "consistency_risk": 0.10,
}

DEFAULT_UNSUPERVISED_WEIGHTS: Dict[str, float] = {
    "anomaly_risk": 0.50,
    "duplicate_risk": 0.25,
    "rarity_risk": 0.25,
}


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize dictionary weights so their values sum to 1.0."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one weight must be greater than zero.")
    return {k: v / total for k, v in weights.items()}


def calculate_sample_risk(
    evidence_df: pd.DataFrame,
    mode: str,
    weights: Optional[Dict[str, float]] = None,
    minimum_coverage: float = 0.75
) -> pd.DataFrame:
    """
    Calculate adaptive sample-level risk and evidence coverage.
    
    Every evidence signal is expected to be normalized in [0, 1] (higher = more suspicious).
    Missing evidence is excluded and weights are renormalized over available signals.

    Parameters:
        evidence_df: DataFrame containing evidence columns (and optional metadata like sample_id).
        mode: 'supervised' or 'unsupervised'.
        weights: Optional dictionary of signal weights.
        minimum_coverage: Minimum required coverage fraction (default 0.75 / 75%) to assign a risk tier.
                          If coverage is below this threshold, tier is 'INSUFFICIENT_EVIDENCE'.

    Returns:
        Augmented DataFrame with 'risk_score' (0-100), 'evidence_coverage' (0-100), and 'risk_level'.
    """
    if mode not in {"supervised", "unsupervised"}:
        raise ValueError(f"mode must be 'supervised' or 'unsupervised', got '{mode}'")

    if weights is None:
        weights = (
            DEFAULT_SUPERVISED_WEIGHTS
            if mode == "supervised"
            else DEFAULT_UNSUPERVISED_WEIGHTS
        )

    norm_weights = normalize_weights(weights)
    total_expected_weight = 1.0

    sample_scores: List[float] = []
    coverage_scores: List[float] = []

    for _, row in evidence_df.iterrows():
        weighted_sum = 0.0
        available_weight = 0.0

        for signal, weight in norm_weights.items():
            if signal not in evidence_df.columns:
                continue

            val = row[signal]
            if pd.isna(val):
                continue

            val_clipped = float(np.clip(val, 0.0, 1.0))
            weighted_sum += val_clipped * weight
            available_weight += weight

        if available_weight == 0.0:
            sample_scores.append(np.nan)
            coverage_scores.append(0.0)
            continue

        # Renormalize over available evidence
        risk = weighted_sum / available_weight
        coverage = available_weight / total_expected_weight

        sample_scores.append(risk)
        coverage_scores.append(coverage)

    result = evidence_df.copy()
    result["risk_score"] = np.array(sample_scores) * 100.0
    result["evidence_coverage"] = np.array(coverage_scores) * 100.0

    def _classify(row: pd.Series) -> str:
        score = row["risk_score"]
        cov = row["evidence_coverage"]

        if pd.isna(score) or cov < (minimum_coverage * 100.0):
            return "INSUFFICIENT_EVIDENCE"

        if score >= 55.0:
            return "HIGH"
        if score >= 35.0:
            return "MEDIUM"
        return "LOW"

    result["risk_level"] = result.apply(_classify, axis=1)
    return result


def calculate_dataset_summary(
    risk_df: pd.DataFrame,
    mode: str = "supervised",
    minimum_coverage: float = 75.0
) -> Dict[str, Any]:
    """
    Calculate aggregate dataset-level risk index, trust score, and evidence coverage.

    Parameters:
        risk_df: DataFrame output from calculate_sample_risk.
        mode: 'supervised' or 'unsupervised'.
        minimum_coverage: Minimum coverage percentage (e.g. 75.0) required to include sample.

    Returns:
        Dictionary summarizing dataset risk, trust score (if supervised), evidence coverage, and status.
    """
    valid = risk_df[risk_df["evidence_coverage"] >= minimum_coverage]
    total_samples = len(risk_df)
    valid_count = len(valid)

    if valid_count == 0:
        return {
            "mode": mode.upper(),
            "total_samples": total_samples,
            "valid_samples": 0,
            "mean_evidence_coverage": round(float(risk_df["evidence_coverage"].mean()), 2) if "evidence_coverage" in risk_df.columns else 0.0,
            "dataset_risk_score": None,
            "dataset_trust_score": None,
            "status": "INSUFFICIENT_EVIDENCE",
            "risk_category": "UNKNOWN",
            "message": "Insufficient evidence coverage across all samples."
        }

    mean_risk = float(valid["risk_score"].mean()) / 100.0
    high_risk_rate = float((valid["risk_level"] == "HIGH").mean())
    medium_risk_rate = float((valid["risk_level"] == "MEDIUM").mean())
    low_risk_rate = float((valid["risk_level"] == "LOW").mean())
    insufficient_count = int((risk_df["risk_level"] == "INSUFFICIENT_EVIDENCE").sum())

    # Dataset Risk Index (40% mean risk + 40% high risk proportion + 20% medium risk proportion)
    dataset_risk = float(0.40 * mean_risk + 0.40 * high_risk_rate + 0.20 * medium_risk_rate)
    dataset_risk_score = round(dataset_risk * 100.0, 2)

    if dataset_risk >= 0.40:
        status = "SUSPICIOUS"
        risk_category = "CRITICAL / ELEVATED"
    elif dataset_risk >= 0.20:
        status = "REVIEW"
        risk_category = "MODERATE"
    else:
        status = "LOW_RISK"
        risk_category = "LOW"

    mean_cov = round(float(risk_df["evidence_coverage"].mean()), 2)

    # Detailed signal breakdown rates (percentage of samples with signal >= 0.50)
    signal_breakdowns: Dict[str, float] = {}
    for col in ["label_risk", "anomaly_risk", "duplicate_risk", "uncertainty_risk", "consistency_risk", "rarity_risk"]:
        if col in risk_df.columns:
            s_series = risk_df[col].dropna()
            if len(s_series) > 0:
                signal_breakdowns[f"{col}_pct"] = round(float((s_series >= 0.50).mean() * 100.0), 2)
            else:
                signal_breakdowns[f"{col}_pct"] = 0.0

    summary: Dict[str, Any] = {
        "mode": f"{mode.upper()} TABULAR",
        "audit_mode": mode.lower(),
        "total_samples": total_samples,
        "valid_samples": valid_count,
        "insufficient_samples": insufficient_count,
        "mean_evidence_coverage_pct": mean_cov,
        "dataset_risk_score": dataset_risk_score,
        "risk_category": risk_category,
        "status": status,
        "sample_breakdown": {
            "high_risk_pct": round(high_risk_rate * 100.0, 2),
            "medium_risk_pct": round(medium_risk_rate * 100.0, 2),
            "low_risk_pct": round(low_risk_rate * 100.0, 2),
            "high_risk_count": int((valid["risk_level"] == "HIGH").sum()),
            "medium_risk_count": int((valid["risk_level"] == "MEDIUM").sum()),
            "low_risk_count": int((valid["risk_level"] == "LOW").sum()),
            "insufficient_count": insufficient_count
        },
        "signal_breakdown": signal_breakdowns
    }

    if mode == "supervised":
        dataset_trust_score = round(max(0.0, min(100.0, 100.0 * (1.0 - dataset_risk))), 2)
        summary["dataset_trust_score"] = dataset_trust_score
        summary["label_analysis"] = "APPLIED"
    else:
        summary["dataset_trust_score"] = None
        summary["label_analysis"] = "NOT APPLICABLE"

    return summary


def format_dataset_audit_card(summary: Dict[str, Any]) -> str:
    """
    Format dataset audit summary into a clean terminal ASCII UI card.
    """
    mode_str = summary.get("mode", "TABULAR")
    status = summary.get("status", "UNKNOWN")
    risk_cat = summary.get("risk_category", "UNKNOWN")
    cov = summary.get("mean_evidence_coverage_pct", 100.0)
    risk_score = summary.get("dataset_risk_score", 0.0)
    trust_score = summary.get("dataset_trust_score")
    sb = summary.get("signal_breakdown", {})

    lines = [
        "┌────────────────────────────────────────────────────────┐",
        "│                  DATASENTINEL AUDIT                    │",
        "├────────────────────────────────────────────────────────┤",
        f"│ Mode:              {mode_str:<35} │"
    ]

    if trust_score is not None:
        lines.append(f"│ Dataset Trust:     {trust_score:.1f} / 100                             │")
        lines.append(f"│ Risk Level:        {risk_cat:<35} │")
    else:
        risk_val_str = f"{risk_score:.1f} / 100" if risk_score is not None else "N/A"
        lines.append(f"│ Anomaly Risk:      {risk_val_str:<35} │")
        lines.append(f"│ Risk Level:        {risk_cat:<35} │")

    lines.append(f"│ Evidence Coverage: {cov:.1f}%                                   │")
    lines.append(f"│ Audit Status:      {status:<35} │")
    lines.append("├────────────────────────────────────────────────────────┤")
    lines.append("│ Evidence Signal Breakdown (Alert Rate >= 0.50):        │")

    if summary.get("audit_mode") == "supervised":
        label_pct = sb.get("label_risk_pct", 0.0)
        anom_pct = sb.get("anomaly_risk_pct", 0.0)
        dup_pct = sb.get("duplicate_risk_pct", 0.0)
        cons_pct = sb.get("consistency_risk_pct", 0.0)
        unc_pct = sb.get("uncertainty_risk_pct", 0.0)
        lines.append(f"│   • Label Risk:            {label_pct:>5.1f}%                          │")
        lines.append(f"│   • Model Uncertainty:     {unc_pct:>5.1f}%                          │")
        lines.append(f"│   • Feature Anomalies:     {anom_pct:>5.1f}%                          │")
        lines.append(f"│   • Duplicates:            {dup_pct:>5.1f}%                          │")
        lines.append(f"│   • Consistency Drift:     {cons_pct:>5.1f}%                          │")
    else:
        anom_pct = sb.get("anomaly_risk_pct", 0.0)
        dup_pct = sb.get("duplicate_risk_pct", 0.0)
        rarity_pct = sb.get("rarity_risk_pct", 0.0)
        lines.append(f"│   • Feature Anomalies:     {anom_pct:>5.1f}%                          │")
        lines.append(f"│   • Duplicates:            {dup_pct:>5.1f}%                          │")
        lines.append(f"│   • Rare Joint Patterns:   {rarity_pct:>5.1f}%                          │")
        lines.append("│   • Label Analysis:        NOT APPLICABLE              │")

    lines.append("└────────────────────────────────────────────────────────┘")
    return "\n".join(lines)
