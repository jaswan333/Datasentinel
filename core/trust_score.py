import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def classify_sample_risk(
    risk_probabilities: Union[pd.Series, np.ndarray],
    high_threshold: float = 0.70,
    medium_threshold: float = 0.40
) -> pd.Series:
    """
    Classify continuous sample risk probabilities into actionable discrete tiers:
    - HIGH: >= high_threshold (e.g. 0.70) -> QUARANTINE / HUMAN REVIEW
    - MEDIUM: >= medium_threshold and < high_threshold -> AUDIT / FLAG
    - LOW: < medium_threshold -> APPROVE / TRUST
    """
    probs = pd.Series(risk_probabilities)
    
    def _tier(p: float) -> str:
        if p >= high_threshold:
            return "HIGH"
        elif p >= medium_threshold:
            return "MEDIUM"
        else:
            return "LOW"
            
    return probs.apply(_tier)


def calculate_dataset_trust_score(
    risk_probabilities: Union[pd.Series, np.ndarray],
    high_threshold: float = 0.70,
    weight_mean: float = 0.50,
    weight_high: float = 0.50
) -> Dict[str, Union[float, str]]:
    """
    Compute aggregate Dataset Trust Score on a 0-100 scale:
    dataset_risk = weight_mean * mean_risk + weight_high * high_risk_rate
    trust_score = 100 * (1 - dataset_risk)
    """
    probs = np.asarray(risk_probabilities, dtype=float)
    
    mean_risk = float(np.mean(probs))
    high_risk_rate = float(np.mean(probs >= high_threshold))
    
    dataset_risk = float(weight_mean * mean_risk + weight_high * high_risk_rate)
    trust_score = float(max(0.0, min(100.0, 100.0 * (1.0 - dataset_risk))))
    
    # Grade interpretation
    if trust_score >= 85.0:
        risk_level = "LOW_RISK"
        status = "TRUSTED"
        action = "APPROVE_FOR_PRODUCTION"
    elif trust_score >= 70.0:
        risk_level = "MODERATE_RISK"
        status = "SUSPICIOUS"
        action = "QUARANTINE_FLAGGED_SAMPLES_AND_REVIEW"
    elif trust_score >= 50.0:
        risk_level = "ELEVATED_RISK"
        status = "POTENTIALLY_COMPROMISED"
        action = "FULL_AUDIT_AND_PURGE_HIGH_RISK"
    else:
        risk_level = "CRITICAL_RISK"
        status = "SEVERELY_POISONED"
        action = "REJECT_DATASET_RETRAIN_FROM_BACKUP"

    return {
        "dataset_trust_score": round(trust_score, 2),
        "dataset_risk": round(dataset_risk, 4),
        "mean_sample_risk": round(mean_risk, 4),
        "high_risk_rate": round(high_risk_rate, 4),
        "risk_level": risk_level,
        "status": status,
        "recommended_action": action
    }


def generate_quarantine_manifest(
    evidence_df: pd.DataFrame,
    risk_level_col: str = "risk_level",
    sort_col: str = "risk_probability",
    target_level: str = "HIGH",
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter and rank high-suspicion samples for quarantine isolation.
    """
    quarantine = evidence_df[evidence_df[risk_level_col] == target_level].copy()
    if sort_col in quarantine.columns:
        quarantine = quarantine.sort_values(by=sort_col, ascending=False)
        
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        quarantine.to_csv(output_path, index=False)
        
    return quarantine


def generate_dataset_trust_report(
    evidence_df: pd.DataFrame,
    poison_rate: float = 0.05,
    risk_prob_col: str = "risk_probability",
    risk_level_col: str = "risk_level",
    high_threshold: float = 0.70,
    output_path: Optional[str] = None
) -> Dict[str, Union[int, float, str]]:
    """
    Generate comprehensive dataset integrity and audit summary report.
    """
    probs = evidence_df[risk_prob_col].values
    trust_metrics = calculate_dataset_trust_score(
        probs,
        high_threshold=high_threshold
    )
    
    total_samples = int(len(evidence_df))
    tier_counts = evidence_df[risk_level_col].value_counts().to_dict()
    
    high_count = int(tier_counts.get("HIGH", 0))
    medium_count = int(tier_counts.get("MEDIUM", 0))
    low_count = int(tier_counts.get("LOW", 0))
    
    report = {
        "poison_rate_simulated": float(poison_rate),
        "total_samples": total_samples,
        "sample_risk_breakdown": {
            "high_risk_quarantine": high_count,
            "medium_risk_review": medium_count,
            "low_risk_cleared": low_count
        },
        "high_risk_percentage": round((high_count / total_samples) * 100.0, 2),
        "mean_sample_risk": trust_metrics["mean_sample_risk"],
        "dataset_risk_index": trust_metrics["dataset_risk"],
        "dataset_trust_score": trust_metrics["dataset_trust_score"],
        "dataset_risk_level": trust_metrics["risk_level"],
        "overall_status": trust_metrics["status"],
        "recommended_action": trust_metrics["recommended_action"]
    }
    
    # Ground-truth audit validation if ground truth column present
    if "poisoned" in evidence_df.columns:
        gt_poisoned = evidence_df["poisoned"].values
        high_mask = (evidence_df[risk_level_col] == "HIGH").values
        tp = int(np.logical_and(high_mask, gt_poisoned == 1).sum())
        fp = int(np.logical_and(high_mask, gt_poisoned == 0).sum())
        total_poison = int(gt_poisoned.sum())
        
        report["quarantine_audit_accuracy"] = {
            "ground_truth_poisoned": total_poison,
            "quarantined_true_positives": tp,
            "quarantined_false_positives": fp,
            "quarantine_recall": round(tp / total_poison, 4) if total_poison > 0 else 0.0,
            "quarantine_precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        }
        
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)
            
    return report


def format_sample_risk_audit(sample_row: Union[pd.Series, dict]) -> str:
    """
    Format a sample row into the human-readable DataSentinel audit format:
    Sample #8421
    Suspicion Probability: 0.91
    Risk Level: HIGH
    Evidence:
    ✓ Label inconsistency
    ✓ High model uncertainty
    ✓ Moderate anomaly score
    ✓ Local neighborhood inconsistency
    Decision:
    QUARANTINE / HUMAN REVIEW
    """
    sample_id = sample_row.get("sample_id", "N/A")
    prob = float(sample_row.get("risk_probability", 0.0))
    risk_level = sample_row.get("risk_level", "UNKNOWN")
    
    cl_risk = float(sample_row.get("cleanlab_risk", 0.0))
    disag = float(sample_row.get("label_disagreement", 0.0))
    entropy = float(sample_row.get("prediction_entropy", 0.0))
    iso = float(sample_row.get("isolation_forest_risk", 0.0))
    lof = float(sample_row.get("lof_risk", 0.0))
    
    evidence_lines = []
    if cl_risk >= 0.50 or disag >= 0.50:
        evidence_lines.append("[!] Strong label inconsistency & disagreement")
    elif cl_risk >= 0.30:
        evidence_lines.append("[*] Mild label inconsistency")
        
    if entropy >= 0.60:
        evidence_lines.append("[!] High model prediction entropy / uncertainty")
    elif entropy >= 0.40:
        evidence_lines.append("[*] Moderate model uncertainty")
        
    if iso >= 0.60:
        evidence_lines.append("[!] Elevated tree-isolation anomaly score")
    elif iso >= 0.40:
        evidence_lines.append("[*] Moderate tree-isolation anomaly score")
        
    if lof >= 0.60:
        evidence_lines.append("[!] High local neighborhood density inconsistency")
    elif lof >= 0.40:
        evidence_lines.append("[*] Moderate local neighborhood inconsistency")
        
    if not evidence_lines:
        evidence_lines.append("[+] Minimal anomalies detected across all signals")
        
    if risk_level == "HIGH":
        decision = "QUARANTINE / HUMAN REVIEW"
    elif risk_level == "MEDIUM":
        decision = "FLAG FOR SECONDARY AUDIT"
    else:
        decision = "APPROVE / CLEAN RECORD"
        
    card = (
        f"Sample #{sample_id}\n"
        f"Suspicion Probability: {prob:.2f}\n"
        f"Risk Level: {risk_level}\n\n"
        f"Evidence:\n" + "\n".join(evidence_lines) + "\n\n"
        f"Decision:\n{decision}"
    )
    return card


def generate_screened_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    evidence_df: pd.DataFrame,
    risk_level_col: str = "risk_level",
    target_quarantine_level: str = "HIGH",
    output_screened_path: Optional[str] = None,
    output_quarantine_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Purge high-risk poisoned samples to produce a screened, purified dataset for downstream retraining.
    """
    quarantine_mask = (evidence_df[risk_level_col] == target_quarantine_level)
    quarantine_manifest = evidence_df[quarantine_mask].copy()
    if "risk_probability" in quarantine_manifest.columns:
        quarantine_manifest = quarantine_manifest.sort_values(by="risk_probability", ascending=False)

    quarantine_ids = set(quarantine_manifest["sample_id"].values)
    keep_mask = [idx not in quarantine_ids for idx in X.index]

    X_screened = X.loc[keep_mask].copy()
    y_screened = y.loc[keep_mask].copy()

    if output_screened_path is not None:
        os.makedirs(os.path.dirname(output_screened_path), exist_ok=True)
        df_out = X_screened.copy()
        df_out["default_payment_next_month"] = y_screened.values
        df_out.to_csv(output_screened_path, index=True)

    if output_quarantine_path is not None:
        os.makedirs(os.path.dirname(output_quarantine_path), exist_ok=True)
        quarantine_manifest.to_csv(output_quarantine_path, index=False)

    return X_screened, y_screened, quarantine_manifest


def evaluate_screening_recovery(
    clean_metrics: Dict[str, float],
    poisoned_metrics: Dict[str, float],
    screened_metrics: Dict[str, float],
    metric: str = "f1"
) -> Dict[str, Union[str, float]]:
    """
    Calculate performance recovery after DataSentinel screening:
    recovery_pct = 100 * (screened - poisoned) / (clean - poisoned)
    """
    c_val = clean_metrics.get(metric, 0.0)
    p_val = poisoned_metrics.get(metric, 0.0)
    s_val = screened_metrics.get(metric, 0.0)

    delta_harm = round(float(p_val - c_val), 4)
    delta_recovery = round(float(s_val - p_val), 4)

    if abs(c_val - p_val) > 1e-5:
        recovery_pct = round(float((s_val - p_val) / (c_val - p_val) * 100.0), 2)
    else:
        recovery_pct = 100.0 if s_val >= p_val else 0.0

    return {
        "metric": metric,
        "clean_value": round(float(c_val), 4),
        "poisoned_value": round(float(p_val), 4),
        "screened_value": round(float(s_val), 4),
        "degradation_due_to_poison": delta_harm,
        "gain_from_screening": delta_recovery,
        "recovery_percentage": recovery_pct
    }

