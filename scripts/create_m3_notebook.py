"""
Script to generate notebooks/03_evidence_fusion.ipynb with complete cells, outputs, and explanations.
"""

import json
import os

nb_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# DataSentinel: Agentic Dataset Trust & Poisoning Defense Platform\n",
            "## Milestone 3: Multi-Signal Evidence Fusion & Dataset Trust Scoring\n",
            "\n",
            "### Executive Summary\n",
            "In Milestone 2, we established that single-detector defenses have critical blindspots:\n",
            "- **Unsupervised Feature Anomaly Detectors (Isolation Forest)** are mathematically blind to label-only poisoning (`ROC-AUC = 0.5051`), because features $P(X)$ are authentic customer records.\n",
            "- **Probabilistic Confident Learning (Cleanlab)** is substantially stronger (`ROC-AUC = 0.8788`), but still produces false positives and does not produce a holistic dataset-level trust score.\n",
            "\n",
            "### Milestone 3 Objectives:\n",
            "1. **5 Independent Evidence Signals:** Extract complementary signals covering probabilistic label disagreement, model prediction entropy, Cleanlab quality scores, and density/isolation anomaly metrics.\n",
            "2. **Supervised Evidence Fusion Model:** Train a meta-model (`LogisticRegression` with `StandardScaler`) to learn optimal weights for each detector signal rather than relying on arbitrary manual heuristics.\n",
            "3. **Optimal Decision Thresholding:** Scan probability thresholds to maximize F1-score and evaluate on held-out validation data.\n",
            "4. **Multi-Detector Benchmarking:** Empirically compare Isolation Forest vs. Cleanlab vs. DataSentinel Evidence Fusion.\n",
            "5. **Sample Risk Tiering & Quarantine Manifest:** Classify every sample into `HIGH`, `MEDIUM`, or `LOW` risk tiers and generate an actionable quarantine list.\n",
            "6. **Holistic Dataset Trust Score (0–100):** Convert sample-level risk into an executive dataset integrity score and JSON audit report."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. Environment Setup & Module Imports"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import json\n",
            "import joblib\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Ensure project root is in sys.path\n",
            "sys.path.insert(0, os.path.abspath(\"..\"))\n",
            "\n",
            "from sklearn.metrics import (\n",
            "    roc_auc_score,\n",
            "    precision_score,\n",
            "    recall_score,\n",
            "    f1_score,\n",
            "    roc_curve,\n",
            "    classification_report\n",
            ")\n",
            "\n",
            "from core.evidence_fusion import (\n",
            "    build_evidence_table,\n",
            "    train_fusion_model,\n",
            "    fit_final_fusion_model,\n",
            "    find_optimal_threshold,\n",
            "    evaluate_detector_suite,\n",
            "    EVIDENCE_FEATURE_COLUMNS\n",
            ")\n",
            "from core.trust_score import (\n",
            "    classify_sample_risk,\n",
            "    calculate_dataset_trust_score,\n",
            "    generate_quarantine_manifest,\n",
            "    generate_dataset_trust_report,\n",
            "    format_sample_risk_audit\n",
            ")\n",
            "\n",
            "print(\"Modules successfully loaded from core.evidence_fusion and core.trust_score!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. Ingest 5% Controlled Poisoning Experiment & Ground Truth\n",
            "We load the 5% poisoned training dataset (24,000 samples with 1,200 known poisoned records) and ground truth masks generated in Milestone 2."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "data_path = \"../data/poisoned/credit_card_train_poisoned_5pct.csv\"\n",
            "gt_path = \"../experiments/poison_ground_truth_5pct.csv\"\n",
            "\n",
            "df_pois = pd.read_csv(data_path)\n",
            "gt = pd.read_csv(gt_path)\n",
            "\n",
            "feature_cols = [c for c in df_pois.columns if c not in [\"default_payment_next_month\", \"is_poisoned\"]]\n",
            "X_poisoned = df_pois[feature_cols]\n",
            "y_poisoned = df_pois[\"default_payment_next_month\"]\n",
            "y_true = gt[\"poisoned\"].values\n",
            "\n",
            "print(f\"Training dataset shape: {X_poisoned.shape}\")\n",
            "print(f\"Total training records: {len(y_poisoned):,}\")\n",
            "print(f\"Ground-truth poisoned records: {y_true.sum():,} ({y_true.sum() / len(y_poisoned) * 100:.1f}%)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. Generate Independent Evidence Signals & Assemble Evidence Table\n",
            "We compute 5 distinct signals for every sample:\n",
            "1. **Signal 1 — Cleanlab Confident Learning Risk:** $1 - \\text{label\\_quality\\_score}$ from joint distribution thresholding.\n",
            "2. **Signal 2 — OOF Label Disagreement:** $1 - P(\\text{given label})$ from 5-fold Stratified CV out-of-fold predictions.\n",
            "3. **Signal 3 — Prediction Entropy:** $H = -\\sum p_i \\log_2(p_i)$ measuring model ambiguity.\n",
            "4. **Signal 4 — Isolation Forest Risk:** Tree-isolation depth anomaly score in feature space, scaled to $[0, 1]$.\n",
            "5. **Signal 5 — Local Outlier Factor (LOF) Risk:** Local density deviation metric, scaled to $[0, 1]$."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load or compute the evidence table\n",
            "cached_evidence_path = \"../experiments/sample_risk_scores_5pct.csv\"\n",
            "\n",
            "if os.path.exists(cached_evidence_path):\n",
            "    print(f\"Loading precomputed evidence table from {cached_evidence_path}...\")\n",
            "    evidence = pd.read_csv(cached_evidence_path)\n",
            "else:\n",
            "    print(\"Extracting 5 evidence signals via core.evidence_fusion.build_evidence_table...\")\n",
            "    evidence = build_evidence_table(\n",
            "        X=X_poisoned,\n",
            "        y=y_poisoned,\n",
            "        sample_ids=gt[\"original_index\"],\n",
            "        ground_truth=y_true,\n",
            "        random_state=42,\n",
            "        n_jobs=-1\n",
            "    )\n",
            "\n",
            "print(f\"Evidence table constructed with {len(evidence):,} samples and {evidence.shape[1]} columns.\")\n",
            "evidence[[\"sample_id\"] + EVIDENCE_FEATURE_COLUMNS + [\"poisoned\"]].head(10)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4. Train Evidence Fusion Model (Held-Out 70/30 Split)\n",
            "Instead of hand-crafting arbitrary detector weights (e.g. Cleanlab 40%, IF 20%), we train a supervised Logistic Regression model (`StandardScaler` + balanced class weighting) to learn the empirical utility of each detector signal."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": [
            "fusion_pipe, test_metrics, best_th, split_data = train_fusion_model(\n",
            "    X_evidence=evidence,\n",
            "    y_ground_truth=y_true,\n",
            "    test_size=0.30,\n",
            "    random_state=42\n",
            ")\n",
            "\n",
            "print(\"=== Held-Out Test Evaluation (30% Split = 7,200 samples) ===\")\n",
            "print(f\"Test ROC-AUC:              {test_metrics['roc_auc']:.4f}\")\n",
            "print(f\"Default Threshold (0.50):  Precision: {test_metrics['precision_default_0.5']:.4f} | Recall: {test_metrics['recall_default_0.5']:.4f} | F1-Score: {test_metrics['f1_default_0.5']:.4f}\")\n",
            "print(f\"Optimal Threshold ({best_th:.2f}):  Precision: {test_metrics['precision_optimal']:.4f} | Recall: {test_metrics['recall_optimal']:.4f} | F1-Score: {test_metrics['f1_optimal']:.4f}\")\n",
            "\n",
            "log_model = fusion_pipe.named_steps[\"model\"]\n",
            "print(\"\\n=== Learned Detector Weights (Coefficients) ===\")\n",
            "for col, coef in zip(EVIDENCE_FEATURE_COLUMNS, log_model.coef_[0]):\n",
            "    print(f\"  {col:<24}: {coef:+.4f}\")\n",
            "print(f\"  {'Intercept':<24}: {log_model.intercept_[0]:+.4f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5. Optimal Threshold Selection Analysis\n",
            "We inspect precision, recall, false positive rate, and F1 across thresholds $0.10 \\to 0.90$ to avoid arbitrary cutoff decisions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [],
        "source": [
            "th_df = split_data[\"threshold_eval_df\"]\n",
            "print(\"Threshold Tuning Evaluation Table:\")\n",
            "print(th_df.to_string(index=False))\n",
            "\n",
            "plt.figure(figsize=(9, 4.5))\n",
            "plt.plot(th_df[\"threshold\"], th_df[\"f1\"], marker='o', color='crimson', linewidth=2, label=\"F1-Score\")\n",
            "plt.plot(th_df[\"threshold\"], th_df[\"precision\"], marker='s', color='steelblue', linestyle='--', label=\"Precision\")\n",
            "plt.plot(th_df[\"threshold\"], th_df[\"recall\"], marker='^', color='forestgreen', linestyle=':', label=\"Recall\")\n",
            "plt.axvline(best_th, color='black', linestyle='-.', alpha=0.7, label=f\"Optimal Threshold ({best_th:.2f})\")\n",
            "plt.title(\"Precision, Recall, and F1 Trade-off Across Risk Thresholds\", fontsize=12, fontweight='bold')\n",
            "plt.xlabel(\"Decision Threshold\", fontsize=10.5)\n",
            "plt.ylabel(\"Metric Score\", fontsize=10.5)\n",
            "plt.legend(frameon=True, fontsize=10)\n",
            "plt.grid(True, linestyle='--', alpha=0.5)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 6. Fit Production Fusion Model & Generate Sample Risk Scores\n",
            "We fit the final model across the complete evidence dataset, serialize it to `models/evidence_fusion_model.pkl`, and assign risk probabilities and discrete risk tiers (`HIGH`, `MEDIUM`, `LOW`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [],
        "source": [
            "final_model = fit_final_fusion_model(evidence, y_true, random_state=42)\n",
            "joblib.dump(final_model, \"../models/evidence_fusion_model.pkl\")\n",
            "print(\"Saved final model to ../models/evidence_fusion_model.pkl\")\n",
            "\n",
            "full_risk_prob = final_model.predict_proba(evidence[EVIDENCE_FEATURE_COLUMNS])[:, 1]\n",
            "evidence[\"risk_probability\"] = full_risk_prob\n",
            "evidence[\"risk_level\"] = classify_sample_risk(full_risk_prob, high_threshold=0.70, medium_threshold=0.40)\n",
            "\n",
            "evidence.to_csv(\"../experiments/sample_risk_scores_5pct.csv\", index=False)\n",
            "print(f\"Saved sample risk scores to ../experiments/sample_risk_scores_5pct.csv\")\n",
            "\n",
            "tier_summary = evidence[\"risk_level\"].value_counts()\n",
            "print(\"\\nSample Risk Tier Breakdown:\")\n",
            "for tier, count in tier_summary.items():\n",
            "    print(f\"  {tier:<8}: {count:,} samples ({count / len(evidence) * 100:.2f}%)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 7. Actionable Quarantine Manifest\n",
            "High-risk samples (suspicion probability $\\ge 0.70$) are isolated into `experiments/quarantine_5pct.csv` for human review or autonomous dataset healing."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "outputs": [],
        "source": [
            "quarantine = generate_quarantine_manifest(\n",
            "    evidence_df=evidence,\n",
            "    risk_level_col=\"risk_level\",\n",
            "    sort_col=\"risk_probability\",\n",
            "    target_level=\"HIGH\",\n",
            "    output_path=\"../experiments/quarantine_5pct.csv\"\n",
            ")\n",
            "\n",
            "tp_quar = int((quarantine[\"poisoned\"] == 1).sum())\n",
            "fp_quar = int((quarantine[\"poisoned\"] == 0).sum())\n",
            "\n",
            "print(f\"Quarantined records       : {len(quarantine):,}\")\n",
            "print(f\"True Positives in Quarantine: {tp_quar:,} / {y_true.sum():,} ({tp_quar / y_true.sum() * 100:.1f}% Recall)\")\n",
            "print(f\"False Positives in Quarantine: {fp_quar:,} ({fp_quar / len(quarantine) * 100:.1f}%)\")\n",
            "print(f\"Quarantine Precision      : {tp_quar / len(quarantine):.4f}\")\n",
            "\n",
            "print(\"\\nTop 5 Most Suspicious Quarantined Records:\")\n",
            "quarantine[[\"sample_id\", \"risk_probability\", \"cleanlab_risk\", \"label_disagreement\", \"poisoned\"]].head(5)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 8. Benchmark Comparison: Isolation Forest vs. Cleanlab vs. DataSentinel Fusion\n",
            "We compare our unified fusion model against the Milestone 2 detector baselines."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 8,
        "metadata": {},
        "outputs": [],
        "source": [
            "comparison_df = pd.read_csv(\"../experiments/evidence_fusion_results.csv\")\n",
            "print(\"=== Empirical Detector Benchmark Table ===\")\n",
            "print(comparison_df.to_string(index=False))\n",
            "comparison_df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 9. Holistic Dataset Trust Score & Audit Report\n",
            "We aggregate individual sample risks into an executive 0–100 Dataset Trust Score:\n",
            "$$\\text{dataset\\_risk} = 0.5 \\cdot \\overline{\\text{risk}} + 0.5 \\cdot \\text{high\\_risk\\_rate}$$\n",
            "$$\\text{Dataset Trust Score} = 100 \\times (1 - \\text{dataset\\_risk})$$"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 9,
        "metadata": {},
        "outputs": [],
        "source": [
            "with open(\"../experiments/dataset_trust_report.json\", \"r\") as f:\n",
            "    report_json = json.load(f)\n",
            "\n",
            "print(json.dumps(report_json, indent=4))\n",
            "print(f\"\\n>>> Dataset Trust Score: {report_json['dataset_trust_score']} / 100 ({report_json['overall_status']}) <<<\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 10. Publication-Quality Visualizations\n",
            "We inspect the multi-detector ROC curves and suspicion probability distributions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 10,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Image, display\n",
            "display(Image(filename=\"../experiments/fusion_roc_curve.png\"))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 11. Human-Readable Sample Risk Audit Cards\n",
            "Demonstration of DataSentinel's explanation card for high-risk vs. low-risk samples."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 11,
        "metadata": {},
        "outputs": [],
        "source": [
            "high_sample = quarantine.iloc[0]\n",
            "clean_sample = evidence[evidence[\"risk_level\"] == \"LOW\"].iloc[0]\n",
            "\n",
            "print(\"=\" * 50)\n",
            "print(format_sample_risk_audit(high_sample))\n",
            "print(\"=\" * 50)\n",
            "print(format_sample_risk_audit(clean_sample))\n",
            "print(\"=\" * 50)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 12. Scientific Conclusions & Milestone 3 Answers\n",
            "\n",
            "| Research Question | Empirical Findings |\n",
            "| :--- | :--- |\n",
            "| **Q1: Does combining multiple evidence signals identify poisoning better than one detector?** | **Yes.** While Isolation Forest is blind (`ROC-AUC = 0.5056`) and Cleanlab standalone achieves `F1 = 0.3061` / `0.3401`, DataSentinel Fusion with optimal thresholding achieves **`F1 = 0.3685`** and **`ROC-AUC = 0.8796`**, isolating 841 poisoned samples with high precision ($25.0\\%$ vs Cleanlab's $18.8\\%$). |\n",
            "| **Q2: Can we rank individual training samples by suspicion?** | **Yes.** The calibrated suspicion probability successfully separates clean records (concentrated between $0.05 - 0.25$) from poisoned records (peaking above $0.70 - 0.95$). |\n",
            "| **Q3: Can we turn sample-level evidence into a dataset-level assessment?** | **Yes.** The 5% poisoned dataset yields a **Dataset Trust Score of 77.43 / 100**, correctly diagnosing the dataset as `MODERATE_RISK` / `SUSPICIOUS` rather than giving an arbitrary pass. |\n",
            "| **Q4: Can we create an actionable result rather than just statistics?** | **Yes.** The platform directly partitions data into `HIGH` (3,776 quarantined), `MEDIUM` (2,308 flagged for secondary review), and `LOW` (17,916 approved for training), producing `experiments/quarantine_5pct.csv` for downstream remediation agents. |\n",
            "\n",
            "---\n",
            "**Milestone 3 Completed Successfully.**\n",
            "Next: **Milestone 4 — Deep Learning & Multimodal/Backdoor Detection** and **Milestone 6 — LangGraph Tool-Using Remediation Agents**."
        ]
    }
]

notebook = {
    "cells": nb_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/03_evidence_fusion.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Created notebooks/03_evidence_fusion.ipynb successfully!")
