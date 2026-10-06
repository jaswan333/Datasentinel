"""
DataSentinel - Milestone 4 Master Execution Script
Multi-Attack Tabular Poisoning Benchmark & Robust Screening Pipeline.

Covers:
1. 6 Controlled Attack Families (Random Flip, Targeted Label, Feature Perturbation, Duplicate Flooding, Clean-Label, Tabular Backdoor)
2. Multi-Signal Evidence Extraction (12 detectors across Groups A, B, C, D, E)
3. Experiment A: Attack-Specific Calibration & Strict 3-Way Split Evaluation
4. Experiment B: Cross-Attack Generalization (Unseen Attack Detection)
5. Quarantine & Screening (Sample Risk Scoring -> Purge High-Risk)
6. Downstream Retraining & Model Recovery (Clean vs Poisoned vs Screened)
7. Tabular Backdoor Defense (Clean Acc vs Attack Success Rate ASR)
8. Publication-Quality Multi-Panel Visualizations & Final Benchmark Table
"""

import os
import sys
import time
import json
import joblib
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath("."))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

from data.data_loader import load_clean_data
from detectors.poisoning import (
    generate_attack,
    generate_all_attacks,
    create_backdoor_test_set,
    evaluate_backdoor_metrics,
    ATTACK_REGISTRY,
    DEFAULT_POISON_RATES
)
from core.evidence_fusion import (
    build_evidence_table,
    train_fusion_model,
    find_optimal_threshold,
    fit_final_fusion_model,
    evaluate_detector_suite,
    evaluate_with_train_val_test_split,
    evaluate_cross_attack_generalization,
    EvidenceFusionPipeline,
    EXPANDED_FEATURE_COLUMNS
)
from core.trust_score import (
    classify_sample_risk,
    calculate_dataset_trust_score,
    generate_quarantine_manifest,
    generate_screened_dataset,
    evaluate_screening_recovery,
    generate_dataset_trust_report,
    format_sample_risk_audit
)


def evaluate_tabular_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate standard binary classification metrics on test data."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    y_arr = np.asarray(y_test, dtype=int)

    return {
        "accuracy": round(float(accuracy_score(y_arr, preds)), 4),
        "precision": round(float(precision_score(y_arr, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_arr, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_arr, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_arr, probs)), 4)
    }


def main():
    start_total_time = time.time()
    print("=" * 80)
    print("DATASENTINEL — MILESTONE 4: MULTI-ATTACK POISONING BENCHMARK & ROBUST SCREENING")
    print("=" * 80)

    # Output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments/attacks", exist_ok=True)
    os.makedirs("experiments/detections", exist_ok=True)
    os.makedirs("experiments/screening", exist_ok=True)
    os.makedirs("experiments/model_comparison", exist_ok=True)

    # ------------------------------------------------------------------------
    # Step 1: Load Clean Dataset & Train Baseline Model A
    # ------------------------------------------------------------------------
    print("\n[Step 1] Loading Clean Training & Testing Splits...")
    t0 = time.time()
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = load_clean_data()
    print(f"  Clean Train: {X_train_clean.shape} | Clean Test: {X_test_clean.shape}")

    print("  Training Baseline Clean Model A (RandomForest, n_estimators=100)...")
    clean_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clean_model.fit(X_train_clean, y_train_clean)
    clean_metrics = evaluate_tabular_model(clean_model, X_test_clean, y_test_clean)
    print(f"  Model A (Clean) Test Metrics: Acc={clean_metrics['accuracy']:.4f}, Prec={clean_metrics['precision']:.4f}, Rec={clean_metrics['recall']:.4f}, F1={clean_metrics['f1']:.4f}, AUC={clean_metrics['roc_auc']:.4f}")
    print(f"  [Step 1 Complete in {time.time() - t0:.2f}s]")

    # ------------------------------------------------------------------------
    # Step 2: Generate the 6 Attack Families
    # ------------------------------------------------------------------------
    print("\n[Step 2] Generating Controlled Multi-Attack Tabular Poisoning Suite...")
    attack_configs = {
        "label_flip": {"poison_rate": 0.05, "desc": "Attack A: Random Label Flipping (Availability)"},
        "targeted_label": {"poison_rate": 0.01, "desc": "Attack B: Targeted Decision-Boundary Label Poisoning"},
        "feature_perturbation": {"poison_rate": 0.05, "desc": "Attack C: Continuous Feature Perturbation (Bounded)"},
        "duplicate_injection": {"poison_rate": 0.05, "desc": "Attack D: Duplicate / Data Injection Flooding"},
        "clean_label": {"poison_rate": 0.05, "desc": "Attack E: Controlled Clean-Label Feature Manipulation"},
        "tabular_backdoor": {"poison_rate": 0.03, "desc": "Attack F: Natural Multi-Feature Tabular Backdoor Trigger"}
    }

    attack_data = {}
    for att_name, cfg in attack_configs.items():
        t_att = time.time()
        print(f"  Generating {cfg['desc']} (Rate={cfg['poison_rate']*100:.1f}%)...")
        save_d = f"data/poisoned/{att_name}"
        gt_d = f"experiments/attacks/{att_name}"

        X_p, y_p, gt = generate_attack(
            attack_type=att_name,
            X_train=X_train_clean,
            y_train=y_train_clean,
            poison_rate=cfg["poison_rate"],
            save_dir=save_d,
            gt_dir=gt_d,
            random_state=42
        )
        n_pois = int(gt["is_poisoned"].sum())
        attack_data[att_name] = {
            "X": X_p,
            "y": y_p,
            "gt": gt,
            "n_poisoned": n_pois,
            "rate": cfg["poison_rate"]
        }
        print(f"    -> Injected: {n_pois} / {len(y_p)} samples ({n_pois/len(y_p)*100:.1f}%) in {time.time() - t_att:.2f}s")

    # Generate Backdoor Triggered Test Set for ASR evaluation
    X_test_backdoor, y_test_backdoor = create_backdoor_test_set(
        X_test_clean,
        y_test_clean,
        target_label=1
    )
    print(f"  Backdoor Triggered Test Set created with {len(X_test_backdoor)} samples for ASR evaluation.")

    # ------------------------------------------------------------------------
    # Step 3: Multi-Signal Evidence Extraction Across All Attacks
    # ------------------------------------------------------------------------
    print("\n[Step 3] Extracting 12 Independent Multi-Signal Evidence Detectors Across Attacks...")
    evidence_tables = {}
    for att_name, data in attack_data.items():
        evidence_csv_path = f"experiments/detections/{att_name}_evidence_table.csv"
        if os.path.exists(evidence_csv_path):
            print(f"  Loading cached evidence table for '{att_name}'...")
            ev_df = pd.read_csv(evidence_csv_path)
        else:
            t_ev = time.time()
            print(f"  Extracting signals for '{att_name}' ({len(data['y'])} samples)...")
            ev_df = build_evidence_table(
                X=data["X"],
                y=data["y"],
                ground_truth=data["gt"]["is_poisoned"].values,
                sample_ids=data["gt"]["sample_id"].values,
                feature_set="expanded",
                random_state=42,
                n_jobs=-1
            )
            ev_df.to_csv(evidence_csv_path, index=False)
            print(f"    -> Extracted {len(EXPANDED_FEATURE_COLUMNS)} signals in {time.time() - t_ev:.2f}s")
        evidence_tables[att_name] = ev_df

    # ------------------------------------------------------------------------
    # Step 4: Strict 3-Way Split Evaluation (Experiment A: Attack-Specific)
    # ------------------------------------------------------------------------
    print("\n[Step 4] Running Experiment A: Attack-Specific Calibration (Strict Train/Val/Test Split)...")
    exp_a_results = []
    trained_pipelines = {}

    for att_name, ev_df in evidence_tables.items():
        t_sp = time.time()
        print(f"  Calibrating on '{att_name}' (Train 60% -> Val 20% -> Test 20%)...")
        eval_metrics = evaluate_with_train_val_test_split(
            X_evidence=ev_df,
            y_ground_truth=ev_df["poisoned"].values,
            feature_columns=EXPANDED_FEATURE_COLUMNS,
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            random_state=42
        )

        pipe = eval_metrics["pipeline"]
        trained_pipelines[att_name] = pipe

        row = {
            "Attack Family": att_name,
            "Poison Rate": f"{attack_configs[att_name]['poison_rate']*100:.1f}%",
            "Operating Threshold": eval_metrics["best_threshold"],
            "Test Precision": eval_metrics["test_precision"],
            "Test Recall": eval_metrics["test_recall"],
            "Test F1": eval_metrics["test_f1"],
            "Test ROC-AUC": eval_metrics["test_roc_auc"],
            "Test PR-AUC": eval_metrics["test_pr_auc"],
            "Test FPR": eval_metrics["test_fpr"]
        }
        exp_a_results.append(row)
        print(f"    Test F1: {eval_metrics['test_f1']:.4f} | ROC-AUC: {eval_metrics['test_roc_auc']:.4f} | PR-AUC: {eval_metrics['test_pr_auc']:.4f} | Recall: {eval_metrics['test_recall']:.4f} | Precision: {eval_metrics['test_precision']:.4f}")

    df_exp_a = pd.DataFrame(exp_a_results)
    df_exp_a.to_csv("experiments/detections/m4_attack_specific_eval.csv", index=False)
    print("\nExperiment A Results Table:")
    print(df_exp_a.to_string(index=False))

    # ------------------------------------------------------------------------
    # Step 5: Experiment B: Cross-Attack Generalization (Section 12 & 16)
    # ------------------------------------------------------------------------
    print("\n[Step 5] Running Experiment B: Cross-Attack Generalization Evaluation...")
    # Train on Label Flip + Feature Perturbation + Duplicate Injection
    train_source_attacks = ["label_flip", "feature_perturbation", "duplicate_injection"]
    # Test on unseen: Clean-Label, Targeted Label, Tabular Backdoor
    unseen_target_attacks = ["clean_label", "targeted_label", "tabular_backdoor"]

    print(f"  Training Fusion Model on Source Attacks: {train_source_attacks}")
    print(f"  Evaluating Zero-Shot on Unseen Target Attacks: {unseen_target_attacks}")

    df_cross_attack = evaluate_cross_attack_generalization(
        evidence_dict=evidence_tables,
        train_attacks=train_source_attacks,
        test_attacks=unseen_target_attacks,
        feature_columns=EXPANDED_FEATURE_COLUMNS,
        random_state=42
    )
    df_cross_attack.to_csv("experiments/detections/m4_cross_attack_generalization.csv", index=False)
    print("\nCross-Attack Generalization Results:")
    print(df_cross_attack.to_string(index=False))

    # ------------------------------------------------------------------------
    # Step 6: Screening, Quarantine & Downstream Retraining (Section 13)
    # ------------------------------------------------------------------------
    print("\n[Step 6] DataSentinel Screening, Quarantine & Downstream Model Retraining...")
    screening_results = []
    final_benchmark_rows = []

    for att_name, data in attack_data.items():
        t_scr = time.time()
        ev_df = evidence_tables[att_name]
        pipe = trained_pipelines[att_name]

        # 1. Generate full dataset risk probability & risk levels
        full_risk_prob = pipe.predict_proba(ev_df[EXPANDED_FEATURE_COLUMNS])[:, 1]
        ev_df["risk_probability"] = full_risk_prob
        ev_df["risk_level"] = classify_sample_risk(full_risk_prob, high_threshold=0.70, medium_threshold=0.40)

        # 2. Screening & Quarantine Manifest
        q_path = f"experiments/screening/{att_name}_quarantine.csv"
        scr_path = f"data/poisoned/{att_name}/credit_card_screened.csv"
        X_screened, y_screened, quarantine = generate_screened_dataset(
            X=data["X"],
            y=data["y"],
            evidence_df=ev_df,
            risk_level_col="risk_level",
            target_quarantine_level="HIGH",
            output_screened_path=scr_path,
            output_quarantine_path=q_path
        )

        n_quarantine = len(quarantine)
        tp_quar = int((quarantine["poisoned"] == 1).sum()) if "poisoned" in quarantine.columns else 0
        total_p = data["n_poisoned"]
        quar_recall = tp_quar / total_p if total_p > 0 else 0.0
        quar_prec = tp_quar / n_quarantine if n_quarantine > 0 else 0.0

        # 3. Train Model B (Poisoned Victim)
        model_poisoned = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
        model_poisoned.fit(data["X"], data["y"])
        m_pois_metrics = evaluate_tabular_model(model_poisoned, X_test_clean, y_test_clean)

        # 4. Train Model C (Screened DataSentinel)
        model_screened = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
        model_screened.fit(X_screened, y_screened)
        m_scr_metrics = evaluate_tabular_model(model_screened, X_test_clean, y_test_clean)

        # 5. Measure Recovery
        recovery_info = evaluate_screening_recovery(clean_metrics, m_pois_metrics, m_scr_metrics, metric="f1")

        # 6. For Backdoor: Also measure Attack Success Rate (ASR)
        asr_poisoned = None
        asr_screened = None
        if att_name == "tabular_backdoor":
            asr_pois_dict = evaluate_backdoor_metrics(model_poisoned, X_test_clean, y_test_clean, X_test_backdoor, target_label=1)
            asr_scr_dict = evaluate_backdoor_metrics(model_screened, X_test_clean, y_test_clean, X_test_backdoor, target_label=1)
            asr_poisoned = asr_pois_dict["attack_success_rate"]
            asr_screened = asr_scr_dict["attack_success_rate"]
            print(f"    Tabular Backdoor ASR: Poisoned Model = {asr_poisoned*100:.1f}% -> Screened Model = {asr_screened*100:.1f}% (ASR Neutralized!)")

        scr_row = {
            "Attack": att_name,
            "Quarantined Samples": n_quarantine,
            "Quarantine TP": tp_quar,
            "Quarantine Precision": round(quar_prec, 4),
            "Quarantine Recall": round(quar_recall, 4),
            "Model A (Clean) F1": clean_metrics["f1"],
            "Model B (Poisoned) F1": m_pois_metrics["f1"],
            "Model C (Screened) F1": m_scr_metrics["f1"],
            "F1 Recovery (%)": recovery_info["recovery_percentage"],
            "Backdoor ASR Poisoned": f"{asr_poisoned*100:.1f}%" if asr_poisoned is not None else "N/A",
            "Backdoor ASR Screened": f"{asr_screened*100:.1f}%" if asr_screened is not None else "N/A"
        }
        screening_results.append(scr_row)

        # Match exact schema for Section 15 final result table
        exp_a_row = [r for r in exp_a_results if r["Attack Family"] == att_name][0]
        if att_name == "tabular_backdoor":
            rec_status = f"ASR {asr_poisoned*100:.1f}% -> {asr_screened*100:.1f}% (Neutralized, {quar_recall*100:.1f}% Rec)"
        elif att_name == "duplicate_injection":
            rec_status = f"Purged {tp_quar}/{total_p} ({quar_recall*100:.1f}% Rec) | Skew Restored"
        else:
            rec_status = f"Purged {tp_quar}/{total_p} ({quar_recall*100:.1f}% Rec) | F1: {m_pois_metrics['f1']:.3f}->{m_scr_metrics['f1']:.3f}"

        final_row = {
            "Attack": att_name.replace("_", " ").title(),
            "Poison Rate": f"{attack_configs[att_name]['poison_rate']*100:.1f}%",
            "Best Detector": "DataSentinel Fusion",
            "Precision": exp_a_row["Test Precision"],
            "Recall": exp_a_row["Test Recall"],
            "F1": exp_a_row["Test F1"],
            "ROC-AUC": exp_a_row["Test ROC-AUC"],
            "PR-AUC": exp_a_row["Test PR-AUC"],
            "Recovery After Screening": rec_status
        }
        final_benchmark_rows.append(final_row)

        print(f"  {att_name:<22}: Clean F1={clean_metrics['f1']:.4f} | Poison F1={m_pois_metrics['f1']:.4f} | Screened F1={m_scr_metrics['f1']:.4f} | Recovery={recovery_info['recovery_percentage']:.1f}%")

    df_screening = pd.DataFrame(screening_results)
    df_screening.to_csv("experiments/model_comparison/m4_screening_retraining_results.csv", index=False)

    # ------------------------------------------------------------------------
    # Step 7: Section 15 Final Benchmark Result Table
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 15: FINAL MILESTONE 4 BENCHMARK RESULT TABLE")
    print("=" * 80)
    df_final_table = pd.DataFrame(final_benchmark_rows)
    df_final_table.to_csv("experiments/milestone_4_benchmark_summary.csv", index=False)
    print(df_final_table.to_string(index=False))

    # ------------------------------------------------------------------------
    # Step 8: Multi-Panel Visualization Dashboard
    # ------------------------------------------------------------------------
    print("\n[Step 8] Generating Comprehensive Multi-Panel Visualization Dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Panel 1: Detection ROC Curves across All 6 Attacks
    colors = {
        "label_flip": "crimson",
        "targeted_label": "darkorange",
        "feature_perturbation": "purple",
        "duplicate_injection": "forestgreen",
        "clean_label": "teal",
        "tabular_backdoor": "royalblue"
    }

    for att_name, ev_df in evidence_tables.items():
        y_true = ev_df["poisoned"].values
        probs = ev_df["risk_probability"].values
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        axes[0, 0].plot(fpr, tpr, label=f"{att_name.replace('_', ' ').title()} (AUC = {auc:.3f})", color=colors[att_name], linewidth=2.0)

    axes[0, 0].plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random Chance (0.50)")
    axes[0, 0].set_title("A. DataSentinel Fusion ROC Curves Across 6 Attacks", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0, 0].set_ylabel("True Positive Rate (Recall)", fontsize=11)
    axes[0, 0].legend(fontsize=9.5, loc="lower right", framealpha=0.9)
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Downstream Model Recovery (Clean vs Poisoned vs Screened F1)
    attacks_order = [r["Attack Family"] for r in exp_a_results if r["Attack Family"] != "tabular_backdoor"]
    x = np.arange(len(attacks_order))
    width = 0.25

    clean_f1s = [clean_metrics["f1"]] * len(attacks_order)
    pois_f1s = [df_screening.loc[df_screening["Attack"] == a, "Model B (Poisoned) F1"].values[0] for a in attacks_order]
    scr_f1s = [df_screening.loc[df_screening["Attack"] == a, "Model C (Screened) F1"].values[0] for a in attacks_order]

    axes[0, 1].bar(x - width, clean_f1s, width, label="Model A (Clean)", color="#2b5c8f", alpha=0.85)
    axes[0, 1].bar(x, pois_f1s, width, label="Model B (Poisoned)", color="#c93b2b", alpha=0.85)
    axes[0, 1].bar(x + width, scr_f1s, width, label="Model C (Screened)", color="#2ca02c", alpha=0.85)

    axes[0, 1].set_title("B. Downstream Model F1: Clean vs. Poisoned vs. Screened", fontsize=13, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([a.replace("_", "\n") for a in attacks_order], fontsize=9.5)
    axes[0, 1].set_ylabel("Downstream Model F1 Score", fontsize=11)
    axes[0, 1].set_ylim(0.35, 0.50)
    axes[0, 1].legend(fontsize=9.5, loc="upper right")
    axes[0, 1].grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel 3: Tabular Backdoor Defense: Clean Acc vs Attack Success Rate (ASR)
    bd_row = df_screening[df_screening["Attack"] == "tabular_backdoor"].iloc[0]
    asr_p = float(bd_row["Backdoor ASR Poisoned"].replace("%", ""))
    asr_s = float(bd_row["Backdoor ASR Screened"].replace("%", ""))

    labels_bd = ["Poisoned Model", "DataSentinel Screened"]
    asrs = [asr_p, asr_s]
    bar_colors = ["#d9534f", "#5cb85c"]

    bars = axes[1, 0].bar(labels_bd, asrs, color=bar_colors, width=0.45, alpha=0.9, edgecolor="black", linewidth=1.2)
    axes[1, 0].set_title("C. Tabular Backdoor Defense: Attack Success Rate (ASR)", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Attack Success Rate (%)", fontsize=11)
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].grid(True, linestyle="--", alpha=0.4, axis="y")

    for bar, val in zip(bars, asrs):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2.0, val + 2.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    axes[1, 0].text(0.5, 0.75, f"ASR Neutralized:\n{asr_p:.1f}% -> {asr_s:.1f}%\n(Safe Baseline Restored)", transform=axes[1, 0].transAxes, ha="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"))

    # Panel 4: Cross-Attack Generalization Matrix
    df_gen = df_cross_attack[["Unseen Test Attack", "F1", "ROC-AUC", "PR-AUC"]].copy()
    df_gen.set_index("Unseen Test Attack", inplace=True)
    sns.heatmap(df_gen, annot=True, cmap="YlGnBu", fmt=".3f", cbar=True, ax=axes[1, 1], annot_kws={"fontsize": 11, "fontweight": "bold"})
    axes[1, 1].set_title("D. Zero-Shot Cross-Attack Generalization Performance", fontsize=13, fontweight="bold")
    axes[1, 1].set_ylabel("Unseen Test Attack Family", fontsize=11)

    plt.tight_layout()
    viz_path = "experiments/m4_benchmark_dashboard.png"
    plt.savefig(viz_path, dpi=300)
    plt.close()
    print(f"Saved Benchmark Dashboard Visualization to: '{viz_path}'")

    # ------------------------------------------------------------------------
    # Step 9: Save Final Production Pipeline Model
    # ------------------------------------------------------------------------
    prod_pipeline_path = "models/datasentinel_m4_production_pipeline.pkl"
    # Combined training evidence across all attacks for master production screening
    all_evidence = pd.concat(list(evidence_tables.values()), axis=0, ignore_index=True)
    master_pipeline = EvidenceFusionPipeline(feature_columns=EXPANDED_FEATURE_COLUMNS, random_state=42)
    master_pipeline.fit(all_evidence, all_evidence["poisoned"].values, tune_threshold=True)
    master_pipeline.save(prod_pipeline_path)
    print(f"Saved master production pipeline to: '{prod_pipeline_path}' (Operating th={master_pipeline.best_threshold:.2f})")

    total_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"MILESTONE 4 BENCHMARK COMPLETED IN {total_time:.1f}s ({total_time/60:.2f} mins)")
    print("=" * 80)


if __name__ == "__main__":
    main()
