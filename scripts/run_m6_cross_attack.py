"""
DataSentinel - Milestone 6: Cross-Attack Zero-Shot Generalization Runner
Evaluates whether DataSentinel learns general multi-signal dataset risk representations
or merely memorizes specific attack signatures.

Source Training Mixture (4 Attacks):
  - Random Label Flip
  - Feature Perturbation
  - Duplicate Injection
  - Patch Backdoor

Zero-Shot Unseen Target Attacks:
  - Targeted Label Poison
  - Clean-Label Manipulation

Outputs:
  experiments/image/m6_cross_attack_zero_shot.csv
  experiments/image/m6_cross_attack_zero_shot.json
  experiments/image/m6_cross_attack_zero_shot.png
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import joblib

sys.path.insert(0, os.path.abspath("."))
from detectors.image.signals import SIGNAL_COLUMNS


def run_cross_attack_evaluation(
    source_attacks: List[str] = None,
    target_attacks: List[str] = None,
    seed: int = 42
):
    print("=" * 80)
    print("DataSentinel M6: Cross-Attack Zero-Shot Transfer Evaluation")
    print("=" * 80)

    if source_attacks is None:
        source_attacks = [
            "random_label_flip",
            "feature_perturbation",
            "duplicate_injection",
            "backdoor_patch"
        ]

    if target_attacks is None:
        target_attacks = [
            "targeted_label_poison",
            "clean_label_manipulation"
        ]

    print(f"[*] Source Training Mixture: {source_attacks}")
    print(f"[*] Unseen Target Evaluation: {target_attacks}")

    # Check if sample evidence files exist
    all_attacks = source_attacks + target_attacks
    missing = [a for a in all_attacks if not os.path.exists(f"experiments/image/{a}/sample_evidence.csv")]
    if missing:
        print(f"\n[ERROR] Missing evidence tables for attacks: {missing}")
        print("Please run `python scripts/run_m6_strong_benchmark.py` first to generate evidence matrices.")
        sys.exit(1)

    # 1. Pool source training and validation partitions
    print("\n[Step 1] Pooling training & validation data from source mixture...")
    train_dfs = []
    val_dfs = []

    for sa in source_attacks:
        df_a = pd.read_csv(f"experiments/image/{sa}/sample_evidence.csv")
        train_dfs.append(df_a[df_a["partition"] == "train"])
        val_dfs.append(df_a[df_a["partition"] == "val"])

    pooled_train = pd.concat(train_dfs, ignore_index=True)
    pooled_val = pd.concat(val_dfs, ignore_index=True)

    print(f"  --> Pooled Train: {len(pooled_train)} samples ({int(pooled_train['is_poisoned'].sum())} poisoned)")
    print(f"  --> Pooled Val:   {len(pooled_val)} samples ({int(pooled_val['is_poisoned'].sum())} poisoned)")

    # 2. Train generalist fusion meta-model
    print("\n[Step 2] Training Generalist Evidence Fusion Meta-Classifier...")
    X_train = pooled_train[SIGNAL_COLUMNS].values
    y_train = pooled_train["is_poisoned"].values

    meta_model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    meta_model.fit(X_train, y_train)

    print("  Learned Signal Weights:")
    for sig, coef in zip(SIGNAL_COLUMNS, meta_model.coef_[0]):
        print(f"    {sig:<30}: {coef:+.4f}")
    print(f"    {'Intercept':<30}: {meta_model.intercept_[0]:+.4f}")

    # 3. Calibrate decision threshold on pooled validation split
    print("\n[Step 3] Calibrating Operating Threshold on Pooled Validation Set...")
    X_val = pooled_val[SIGNAL_COLUMNS].values
    y_val = pooled_val["is_poisoned"].values
    val_probs = meta_model.predict_proba(X_val)[:, 1]

    best_th = 0.50
    best_f1 = 0.0
    for th in np.arange(0.10, 0.91, 0.02):
        th = round(float(th), 2)
        preds = (val_probs >= th).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = th

    print(f"  --> Calibrated Operating Threshold: {best_th:.2f} (Validation F1: {best_f1:.4f})")

    # 4. Evaluate Zero-Shot Transfer onto Unseen Target Attacks
    print("\n[Step 4] Evaluating Zero-Shot Generalization on Completely Unseen Attacks...")
    results: List[Dict[str, Any]] = []

    for ta in target_attacks:
        df_target = pd.read_csv(f"experiments/image/{ta}/sample_evidence.csv")
        test_df = df_target[df_target["partition"] == "test"]

        X_test = test_df[SIGNAL_COLUMNS].values
        y_test = test_df["is_poisoned"].values
        n_poisoned_test = int(y_test.sum())

        test_probs = meta_model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= best_th).astype(int)

        prec = float(precision_score(y_test, test_preds, zero_division=0))
        rec = float(recall_score(y_test, test_preds, zero_division=0))
        f1 = float(f1_score(y_test, test_preds, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.5
        pr_auc = float(average_precision_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.0

        cm = confusion_matrix(y_test, test_preds)
        tn = cm[0, 0] if cm.shape == (2, 2) else 0
        fp = cm[0, 1] if cm.shape == (2, 2) else 0
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

        # Full dataset quarantine recall on unseen attack
        all_X = df_target[SIGNAL_COLUMNS].values
        all_y = df_target["is_poisoned"].values
        all_probs = meta_model.predict_proba(all_X)[:, 1]
        q_mask = (all_probs >= best_th)
        q_rec = float((q_mask & (all_y == 1)).sum() / all_y.sum()) if all_y.sum() > 0 else 0.0
        q_prec = float((q_mask & (all_y == 1)).sum() / q_mask.sum()) if q_mask.sum() > 0 else 0.0

        print(f"\n  [TARGET: {ta.upper()}] (Test Samples: {len(test_df)}, Poisoned: {n_poisoned_test})")
        print(f"    Zero-Shot Test ROC-AUC:  {roc_auc:.4f}")
        print(f"    Zero-Shot Test PR-AUC:   {pr_auc:.4f}")
        print(f"    Zero-Shot Test Recall:   {rec*100:.2f}% | Precision: {prec*100:.2f}% | F1: {f1:.4f}")
        print(f"    Full Quarantine Recall:  {q_rec*100:.2f}% | Precision: {q_prec*100:.2f}%")

        results.append({
            "target_attack": ta,
            "source_mixture": "+".join(source_attacks),
            "calibrated_threshold": float(best_th),
            "zero_shot_roc_auc": round(roc_auc, 4),
            "zero_shot_pr_auc": round(pr_auc, 4),
            "zero_shot_recall_pct": round(rec * 100, 2),
            "zero_shot_precision_pct": round(prec * 100, 2),
            "zero_shot_f1": round(f1, 4),
            "zero_shot_fpr_pct": round(fpr * 100, 2),
            "quarantine_recall_pct": round(q_rec * 100, 2),
            "quarantine_precision_pct": round(q_prec * 100, 2)
        })

    # Save artifacts
    csv_path = "experiments/image/m6_cross_attack_zero_shot.csv"
    json_path = "experiments/image/m6_cross_attack_zero_shot.json"
    df_res = pd.DataFrame(results)
    df_res.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    joblib.dump(meta_model, "models/image/m6_generalist_fusion_model.pkl")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    t_names = [r["target_attack"].replace("_", "\n") for r in results]
    x = np.arange(len(t_names))
    width = 0.35

    rocs = [r["zero_shot_roc_auc"] for r in results]
    recs = [r["zero_shot_recall_pct"] / 100.0 for r in results]

    ax.bar(x - width/2, rocs, width, label="Zero-Shot ROC-AUC", color="#2563eb")
    ax.bar(x + width/2, recs, width, label="Zero-Shot Recall", color="#8b5cf6")
    ax.axhline(0.5, color="red", linestyle="--", label="Chance (0.50)")
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_title("DataSentinel: Zero-Shot Cross-Attack Generalization", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(t_names)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("experiments/image/m6_cross_attack_zero_shot.png", dpi=200)
    plt.close()

    print("\n" + "=" * 80)
    print("ZERO-SHOT CROSS-ATTACK GENERALIZATION SUMMARY")
    print("=" * 80)
    print(f"{'Unseen Target Attack':<28} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Recall':<8} | {'Precision':<10} | {'F1':<6}")
    print("-" * 80)
    for r in results:
        print(f"{r['target_attack']:<28} | {r['zero_shot_roc_auc']:>8.4f} | {r['zero_shot_pr_auc']:>8.4f} | {r['zero_shot_recall_pct']:>7.2f}% | {r['zero_shot_precision_pct']:>9.2f}% | {r['zero_shot_f1']:>6.4f}")
    print("=" * 80)
    print(f"[SUCCESS] Zero-shot evaluation complete! Artifacts saved to:\n  - {csv_path}\n  - {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataSentinel M6 Cross-Attack Zero-Shot Evaluation")
    parser.add_argument("--source-attacks", nargs="+", default=None)
    parser.add_argument("--target-attacks", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_cross_attack_evaluation(
        source_attacks=args.source_attacks,
        target_attacks=args.target_attacks,
        seed=args.seed
    )
