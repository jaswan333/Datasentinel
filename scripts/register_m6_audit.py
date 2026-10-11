"""
DataSentinel - Milestone 6: Image Audit Store Registration Bridge
Registers M6 multi-attack benchmark findings and 14-signal evidence matrices
into runtime/results/m6_cifar10.json, enabling the LangGraph agentic triad
(Audit Agent -> Investigation Agent -> Reporting Agent) to autonomously
inspect and explain deep learning training-data security risks.
"""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from services.audit_store import register_dataset

SUMMARY_PATH = "experiments/image/m6_strong_benchmark_summary.json"
BACKDOOR_EVIDENCE_PATH = "experiments/image/backdoor_patch/sample_evidence.csv"
FALLBACK_EVIDENCE_PATH = "experiments/image/m6_image_sample_evidence.csv"


def register_m6_image_audit():
    print("[*] Registering upgraded M6 benchmark into DataSentinel Audit Store...")

    # Load benchmark summary if available
    benchmark_summary = []
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            benchmark_summary = json.load(f)

    # Load sample evidence table
    if os.path.exists(BACKDOOR_EVIDENCE_PATH):
        evidence_df = pd.read_csv(BACKDOOR_EVIDENCE_PATH)
        primary_threat = "backdoor_patch"
    elif os.path.exists(FALLBACK_EVIDENCE_PATH):
        evidence_df = pd.read_csv(FALLBACK_EVIDENCE_PATH)
        primary_threat = "legacy_backdoor"
    else:
        print("[ERROR] No image evidence table found. Please run the benchmark first.")
        sys.exit(1)

    # Sort samples by risk_probability descending
    sorted_samples = evidence_df.sort_values(by="risk_probability", ascending=False)
    sample_records = sorted_samples.head(500).to_dict(orient="records")

    quarantine_count = int(evidence_df["quarantine"].sum())
    mean_risk = float(evidence_df["risk_probability"].mean())
    high_risk_rate = float(np_rate := (evidence_df["risk_probability"] >= 0.55).mean())
    risk_score = round(float((mean_risk * 0.5 + high_risk_rate * 0.5) * 100), 2)
    trust_score = round(100.0 - risk_score, 2)

    risk_level = "HIGH_RISK" if trust_score < 60 else ("MODERATE_RISK" if trust_score < 80 else "LOW_RISK")

    detector_summary = {
        "group_a_label_consistency": "Training loss risk, prediction entropy, margin deficit, assigned-label confidence, OOF probe disagreement",
        "group_b_deep_representations": "KNN 512D distance, Local Outlier Factor, class centroid ratio, activation clustering (PCA+KMeans), SVD spectral signatures",
        "group_c_image_integrity": "MD5 exact byte collision, 64-bit dHash perceptual near-duplicate, RGB & Laplacian gradient anomaly via Isolation Forest",
        "group_d_model_behaviour": "Local patch spatial sensitivity across 5 regional masks (TL, TR, BL, BR, center)",
        "evidence_fusion": "Calibrated Logistic Regression meta-model (60/20/20 group-preserved split)"
    }

    audit_payload = {
        "dataset_id": "m6_cifar10",
        "filename": "cifar10_backdoor_trojan.png",
        "mode": "IMAGE_DEEP_LEARNING",
        "primary_threat": primary_threat,
        "n_rows": int(len(evidence_df)),
        "n_columns": 3,
        "dataset_risk_score": risk_score,
        "risk_score": risk_score,
        "dataset_trust_score": trust_score,
        "trust_score": trust_score,
        "dataset_risk_level": risk_level,
        "risk_level": risk_level,
        "status": "BACKDOOR_SUSPICION" if risk_score > 30 else "AUDITED_SECURE",
        "evidence_coverage": "100.0%",
        "recommended_action": f"Quarantine {quarantine_count} high-suspicion records and retrain screened CIFAR-adapted ResNet18 model.",
        "high_risk_quarantine": quarantine_count,
        "high_risk_samples": quarantine_count,
        "detector_summary": detector_summary,
        "benchmark_summary": benchmark_summary,
        "sample_results": sample_records
    }

    register_dataset(
        dataset_id="m6_cifar10",
        file_path="data/image/raw/cifar-10-batches-py",
        audit_result=audit_payload
    )

    print(f"[SUCCESS] Successfully registered m6_cifar10 with {len(evidence_df)} samples!")
    print(f"  - Dataset Trust Score: {trust_score} / 100 ({risk_level})")
    print(f"  - Quarantined Samples: {quarantine_count}")
    print(f"  - Primary Threat Evaluated: {primary_threat}")


if __name__ == "__main__":
    register_m6_image_audit()
