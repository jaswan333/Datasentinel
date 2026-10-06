"""
Script to execute Milestone 3 pipeline and generate all required artifacts:
- models/evidence_fusion_model.pkl
- experiments/evidence_fusion_results.csv
- experiments/sample_risk_scores_5pct.csv
- experiments/quarantine_5pct.csv
- experiments/dataset_trust_report.json
- experiments/fusion_roc_curve.png
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    classification_report
)

from core.evidence_fusion import (
    build_evidence_table,
    train_fusion_model,
    fit_final_fusion_model,
    find_optimal_threshold,
    evaluate_detector_suite,
    EVIDENCE_FEATURE_COLUMNS
)
from core.trust_score import (
    classify_sample_risk,
    calculate_dataset_trust_score,
    generate_quarantine_manifest,
    generate_dataset_trust_report,
    format_sample_risk_audit
)


def main():
    print("=" * 70)
    print("DataSentinel — Milestone 3 Evidence Fusion Execution")
    print("=" * 70)

    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    # 1. Load 5% Poisoned Dataset & Ground Truth
    data_path = "data/poisoned/credit_card_train_poisoned_5pct.csv"
    gt_path = "experiments/poison_ground_truth_5pct.csv"

    print(f"\n[Step 1] Loading datasets from '{data_path}' and '{gt_path}'...")
    df_pois = pd.read_csv(data_path)
    gt = pd.read_csv(gt_path)

    feature_cols = [c for c in df_pois.columns if c not in ["default_payment_next_month", "is_poisoned"]]
    X = df_pois[feature_cols]
    y = df_pois["default_payment_next_month"]
    y_true = gt["poisoned"].values

    n_samples = len(y)
    n_poisoned = int(y_true.sum())
    print(f"Total training samples: {n_samples}")
    print(f"Known poisoned samples (Ground Truth): {n_poisoned} ({n_poisoned / n_samples * 100:.1f}%)")

    # 2. Extract Evidence Signals & Build Evidence Table
    sample_scores_path = "experiments/sample_risk_scores_5pct.csv"
    if os.path.exists(sample_scores_path):
        print(f"\n[Step 2] Found existing sample risk scores at '{sample_scores_path}'. Loading...")
        evidence = pd.read_csv(sample_scores_path)
    else:
        print("\n[Step 2] Extracting 5 independent evidence signals...")
        print("  Signal 1: Cleanlab Label Quality Risk Score")
        print("  Signal 2: 5-Fold OOF Label Disagreement")
        print("  Signal 3: Normalized Prediction Entropy (Uncertainty)")
        print("  Signal 4: Isolation Forest Anomaly Risk")
        print("  Signal 5: Local Outlier Factor (LOF) Density Deviation")

        evidence = build_evidence_table(
            X=X,
            y=y,
            sample_ids=gt["original_index"],
            ground_truth=y_true,
            random_state=42,
            n_jobs=-1
        )

    print("\nEvidence Table summary (first 5 samples):")
    print(evidence.head().to_string(index=False))

    # 3. Train & Evaluate Fusion Model on 70/30 Split
    print("\n[Step 3] Training Evidence Fusion Model (Logistic Regression + StandardScaler on 70/30 Split)...")
    fusion_split_pipe, test_metrics, best_th, split_data = train_fusion_model(
        X_evidence=evidence,
        y_ground_truth=y_true,
        test_size=0.30,
        random_state=42
    )

    print(f"Held-out Test Evaluation (30% split = {len(split_data['y_test'])} samples):")
    print(f"  Test ROC-AUC:              {test_metrics['roc_auc']:.4f}")
    print(f"  Default Threshold (0.50)   Precision: {test_metrics['precision_default_0.5']:.4f} | Recall: {test_metrics['recall_default_0.5']:.4f} | F1: {test_metrics['f1_default_0.5']:.4f}")
    print(f"  Optimal Threshold ({best_th:.2f})   Precision: {test_metrics['precision_optimal']:.4f} | Recall: {test_metrics['recall_optimal']:.4f} | F1: {test_metrics['f1_optimal']:.4f}")

    # Inspect Learned Weights
    log_reg = fusion_split_pipe.named_steps["model"]
    print("\nLearned Signal Coefficients (Importance Weights):")
    for feat, w in zip(EVIDENCE_FEATURE_COLUMNS, log_reg.coef_[0]):
        print(f"  {feat:<25}: {w:+.4f}")
    print(f"  {'Intercept':<25}: {log_reg.intercept_[0]:+.4f}")

    # 4. Fit Final Production Fusion Model on Entire Dataset
    print("\n[Step 4] Fitting Final Fusion Model on entire dataset & saving artifact...")
    final_fusion_model = fit_final_fusion_model(
        X_evidence=evidence,
        y_ground_truth=y_true,
        random_state=42
    )
    model_path = "models/evidence_fusion_model.pkl"
    joblib.dump(final_fusion_model, model_path)
    print(f"Saved final fusion model to: '{model_path}'")

    # Generate sample risk probabilities
    full_risk_prob = final_fusion_model.predict_proba(evidence[EVIDENCE_FEATURE_COLUMNS])[:, 1]
    evidence["risk_probability"] = full_risk_prob

    # Classify sample risk level (prototype thresholds: >= 0.70 HIGH, >= 0.40 MEDIUM, else LOW)
    evidence["risk_level"] = classify_sample_risk(
        full_risk_prob,
        high_threshold=0.70,
        medium_threshold=0.40
    )

    # Save sample risk scores
    sample_scores_path = "experiments/sample_risk_scores_5pct.csv"
    evidence.to_csv(sample_scores_path, index=False)
    print(f"Saved complete sample risk scores to: '{sample_scores_path}'")

    # 5. Quarantine List
    print("\n[Step 5] Creating Quarantine List (HIGH risk samples)...")
    quarantine_path = "experiments/quarantine_5pct.csv"
    quarantine = generate_quarantine_manifest(
        evidence_df=evidence,
        risk_level_col="risk_level",
        sort_col="risk_probability",
        target_level="HIGH",
        output_path=quarantine_path
    )
    print(f"Quarantine list created with {len(quarantine)} samples saved to: '{quarantine_path}'")
    tp_quar = int((quarantine["poisoned"] == 1).sum())
    fp_quar = int((quarantine["poisoned"] == 0).sum())
    print(f"  True Positives in Quarantine : {tp_quar} / {n_poisoned} ({tp_quar / n_poisoned * 100:.1f}% Recall)")
    print(f"  False Positives in Quarantine: {fp_quar} ({fp_quar / len(quarantine) * 100:.1f}%)")
    print(f"  Quarantine Precision         : {tp_quar / len(quarantine):.4f}")

    # 6. Detector Suite Comparison Table
    print("\n[Step 6] Benchmarking Isolation Forest vs. Cleanlab vs. DataSentinel Fusion...")
    detector_comparison = evaluate_detector_suite(
        evidence_df=evidence,
        ground_truth=y_true,
        fusion_probabilities=full_risk_prob,
        fusion_threshold=best_th
    )

    # Also add default 0.50 row for complete transparency
    fus_pred_05 = (full_risk_prob >= 0.50).astype(int)
    row_05 = {
        "Detector": "DataSentinel Evidence Fusion (Default th=0.50)",
        "Flagged": int(fus_pred_05.sum()),
        "True Positives": int(np.logical_and(fus_pred_05 == 1, y_true == 1).sum()),
        "False Positives": int(np.logical_and(fus_pred_05 == 1, y_true == 0).sum()),
        "Precision": round(float(precision_score(y_true, fus_pred_05, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, fus_pred_05, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, fus_pred_05, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_true, full_risk_prob)), 4)
    }
    detector_comparison = pd.concat([detector_comparison, pd.DataFrame([row_05])], ignore_index=True)
    comparison_path = "experiments/evidence_fusion_results.csv"
    detector_comparison.to_csv(comparison_path, index=False)
    print(detector_comparison.to_string(index=False))
    print(f"\nSaved detector comparison table to: '{comparison_path}'")

    # 7. Dataset Trust Score & Report
    print("\n[Step 7] Calculating holistic Dataset Trust Score & generating JSON audit report...")
    trust_report_path = "experiments/dataset_trust_report.json"
    trust_report = generate_dataset_trust_report(
        evidence_df=evidence,
        poison_rate=0.05,
        risk_prob_col="risk_probability",
        risk_level_col="risk_level",
        high_threshold=0.70,
        output_path=trust_report_path
    )
    print(json.dumps(trust_report, indent=4))
    print(f"Saved dataset trust report to: '{trust_report_path}'")

    # 8. Human-Readable Sample Audit Demo
    print("\n[Step 8] Generating Sample Risk Cards Demo:")
    top_sample = quarantine.iloc[0]
    clean_sample = evidence[evidence["risk_level"] == "LOW"].iloc[0]
    print("\n--- HIGH RISK SUSPICIOUS SAMPLE ---")
    print(format_sample_risk_audit(top_sample))
    print("\n--- LOW RISK TRUSTED SAMPLE ---")
    print(format_sample_risk_audit(clean_sample))

    # 9. Build Multi-Panel Visualizations
    print("\n[Step 9] Generating Publication-Quality Visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: ROC Curves
    fpr_iso, tpr_iso, _ = roc_curve(y_true, evidence["isolation_forest_risk"])
    fpr_cl, tpr_cl, _ = roc_curve(y_true, evidence["cleanlab_risk"])
    fpr_fus, tpr_fus, _ = roc_curve(y_true, full_risk_prob)

    auc_iso = roc_auc_score(y_true, evidence["isolation_forest_risk"])
    auc_cl = roc_auc_score(y_true, evidence["cleanlab_risk"])
    auc_fus = roc_auc_score(y_true, full_risk_prob)

    axes[0].plot(fpr_iso, tpr_iso, label=f"Isolation Forest (ROC-AUC = {auc_iso:.4f})", color="steelblue", linestyle="--", linewidth=1.8)
    axes[0].plot(fpr_cl, tpr_cl, label=f"Cleanlab Confident Learning (ROC-AUC = {auc_cl:.4f})", color="darkorange", linestyle="-.", linewidth=2.0)
    axes[0].plot(fpr_fus, tpr_fus, label=f"DataSentinel Evidence Fusion (ROC-AUC = {auc_fus:.4f})", color="crimson", linewidth=2.8)
    axes[0].plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random Chance (0.50)")

    axes[0].set_title("Detector Discrimination Performance (ROC Curves)", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=11)
    axes[0].legend(fontsize=10.5, loc="lower right", frameon=True, framealpha=0.9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Risk Probability Distribution (Clean vs Poisoned)
    clean_probs = full_risk_prob[y_true == 0]
    poison_probs = full_risk_prob[y_true == 1]

    sns.histplot(clean_probs, bins=40, color="forestgreen", alpha=0.5, label=f"Clean Samples (n={len(clean_probs)})", ax=axes[1], stat="density", kde=True)
    sns.histplot(poison_probs, bins=40, color="crimson", alpha=0.6, label=f"Poisoned Samples (n={len(poison_probs)})", ax=axes[1], stat="density", kde=True)

    axes[1].axvline(0.70, color="firebrick", linestyle="--", linewidth=2, label="High Risk / Quarantine Threshold (0.70)")
    axes[1].axvline(0.40, color="goldenrod", linestyle=":", linewidth=2, label="Medium Risk Threshold (0.40)")

    axes[1].set_title("Sample Suspicion Probability Distribution", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("DataSentinel Suspicion Probability", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].legend(fontsize=10, loc="upper right", frameon=True, framealpha=0.9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    viz_path = "experiments/fusion_roc_curve.png"
    plt.savefig(viz_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve and risk distribution plot to: '{viz_path}'")

    print("\n" + "=" * 70)
    print("Milestone 3 Execution Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
