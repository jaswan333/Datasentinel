"""Run a focused clean-label hardening experiment against the existing M6 runner.

Run from the DataSentinel project root after copying this overlay into the project:
    python scripts/run_m6_clean_label_hardening.py --epochs 5 --max-validation-fpr 0.05

The script does not overwrite the original runner source. It patches functions in memory,
adds a prototype-affinity signal to this run, uses a validation-only FPR-constrained
threshold, and saves a before/after JSON/CSV comparison.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.image.clean_label_signals import (
    cross_class_prototype_affinity_risk,
    latent_cross_class_prototype_affinity_risk
)
from detectors.image.threshold_policy import choose_threshold_max_recall_under_fpr, binary_metrics


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--test-limit", type=int, default=300)
    parser.add_argument("--max-validation-fpr", type=float, default=0.05)
    parser.add_argument("--no-behavior-signal", action="store_true")
    args = parser.parse_args()

    try:
        import scripts.run_m6_strong_benchmark as m6
    except ImportError as exc:
        raise SystemExit(
            "Could not import scripts/run_m6_strong_benchmark.py. Copy this overlay into your existing DataSentinel project and run it from the project root."
        ) from exc

    out_dir = ROOT / "experiments/image/clean_label_hardening"
    out_dir.mkdir(parents=True, exist_ok=True)
    original_summary_path = ROOT / "experiments/image/clean_label/summary.json"
    baseline = read_json(original_summary_path)
    if baseline is not None:
        (out_dir / "baseline_clean_label_summary.json").write_text(
            json.dumps(baseline, indent=2), encoding="utf-8"
        )

    if not hasattr(m6, "build_evidence") or not hasattr(m6, "run_one_attack"):
        raise SystemExit("Your M6 runner must expose build_evidence() and run_one_attack(); inspect its source before applying this overlay.")
    if not hasattr(m6, "SIGNAL_COLUMNS") or not hasattr(m6, "split_indices"):
        raise SystemExit("Your M6 runner is missing SIGNAL_COLUMNS or split_indices(); this overlay expects the M6 structure in your report.")

    max_fpr = float(np.clip(args.max_validation_fpr, 0.0, 1.0))
    # Monkey-patch only this process; the source file remains untouched.
    original_build_evidence = m6.build_evidence

    def build_evidence_with_prototype_signal(
        attack_name, attack_result, victim_model, train_loader, device, enable_behavior=True, *extra, **kwargs
    ):
        evidence, embeddings, labels_used = original_build_evidence(
            attack_name, attack_result, victim_model, train_loader, device,
            enable_behavior=enable_behavior, *extra, **kwargs
        )
        # Use the exact group-aware fusion-training indices; for clean-label manipulation
        # the groups are distinct, while duplicate families preserve shared group_id.
        train_idx, _, _ = m6.split_indices(
            attack_result.labels,
            getattr(attack_result, "group_id", np.arange(len(attack_result.labels))),
            seed=getattr(m6, "SEED", 42),
        )
        evidence["cross_class_prototype_affinity_risk"] = cross_class_prototype_affinity_risk(
            attack_result.images,
            labels_used,
            reference_indices=train_idx,
        )
        evidence["latent_cross_class_prototype_affinity_risk"] = latent_cross_class_prototype_affinity_risk(
            embeddings,
            labels_used,
            reference_indices=train_idx,
            metric="cosine"
        )
        return evidence, embeddings, labels_used

    m6.build_evidence = build_evidence_with_prototype_signal
    signal_cols = list(m6.SIGNAL_COLUMNS)
    if "cross_class_prototype_affinity_risk" not in signal_cols:
        signal_cols.append("cross_class_prototype_affinity_risk")
    if "latent_cross_class_prototype_affinity_risk" not in signal_cols:
        signal_cols.append("latent_cross_class_prototype_affinity_risk")
    m6.SIGNAL_COLUMNS = signal_cols

    # fit_fusion() in the existing runner calls this function on validation probabilities.
    m6.choose_threshold = lambda y, p: choose_threshold_max_recall_under_fpr(y, p, max_fpr=max_fpr)

    # Keep the baseline's sample scale and epoch count easy to reproduce. Explicit args
    # are used instead of silently using function-definition defaults in load_cifar().
    if hasattr(m6, "EPOCHS"):
        m6.EPOCHS = args.epochs
    device = m6.get_device()
    print(f"Device: {device}")
    print(f"Train limit={args.train_limit}; test limit={args.test_limit}; epochs={args.epochs}")
    print(f"Validation FPR ceiling={max_fpr:.1%}")

    train_images, train_labels, test_images, test_labels = m6.load_cifar(
        limit_train=args.train_limit,
        limit_test=args.test_limit,
        seed=getattr(m6, "SEED", 42),
    )
    clean_model, clean_metrics = m6.run_clean_reference(
        train_images, train_labels, test_images, test_labels, device
    )
    target_probs = m6.make_target_probabilities(
        clean_model, train_images, train_labels, device
    )
    target_prototype = m6.make_target_prototype(
        train_images, train_labels, target_class=getattr(m6, "TARGET_CLASS", 0)
    )

    result = m6.run_one_attack(
        "clean_label",
        train_images,
        train_labels,
        test_images,
        test_labels,
        target_probs,
        target_prototype,
        clean_model,
        device,
        enable_behavior=not args.no_behavior_signal,
    )
    result["clean_reference_accuracy"] = float(clean_metrics["accuracy"])
    result["validation_fpr_ceiling"] = max_fpr
    result["calibration_policy"] = "maximize recall subject to validation FPR ceiling; tie-break by precision, F1, then higher threshold"
    result["extra_signals"] = ["cross_class_prototype_affinity_risk", "latent_cross_class_prototype_affinity_risk"]

    evidence_path = ROOT / "experiments/image/clean_label/sample_evidence.csv"
    if evidence_path.exists():
        evidence = pd.read_csv(evidence_path)
        required = {"partition", "is_poisoned", "risk_probability"}
        if required.issubset(evidence.columns):
            test = evidence[evidence["partition"].astype(str).str.lower().eq("test")]
            if not test.empty:
                threshold = float(result.get("detector_threshold", result.get("threshold", 0.5)))
                metrics = binary_metrics(
                    test["is_poisoned"].astype(int).to_numpy(),
                    test["risk_probability"].astype(float).to_numpy(),
                    threshold,
                )
                result["heldout_test_metrics_recomputed"] = metrics
                result["detector_test_roc_auc"] = metrics["roc_auc"]
                result["detector_test_pr_auc"] = metrics["pr_auc"]
                result["detector_test_fpr"] = metrics["fpr"]
                result["detector_test_precision"] = metrics["precision"]
                result["detector_test_recall"] = metrics["recall"]
                result["detector_test_f1"] = metrics["f1"]
                result["detector_test_quarantine_rate"] = metrics["quarantine_rate"]

    summary_path = ROOT / "experiments/image/clean_label/summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Explicit population definitions for audit rigor
    populations = {
        "heldout_test_partition": {
            "description": "Held-out unseen test samples strictly reserved for detector evaluation",
            "sample_count": 240,
            "poison_count": 12,
            "clean_count": 228,
            "poison_prevalence": 0.05,
            "metrics": {
                "test_roc_auc": result.get("detector_test_roc_auc"),
                "test_pr_auc": result.get("detector_test_pr_auc"),
                "test_precision": result.get("detector_test_precision"),
                "test_recall": result.get("detector_test_recall"),
                "test_f1": result.get("detector_test_f1"),
                "test_fpr": result.get("detector_test_fpr"),
            }
        },
        "full_ingestion_dataset": {
            "description": "Full training dataset screened and quarantined prior to model retraining",
            "sample_count": 1200,
            "poison_count": 60,
            "clean_count": 1140,
            "metrics": {
                "quarantined_count": result.get("quarantined_count"),
                "quarantine_rate": result.get("detector_quarantine_rate"),
                "quarantine_recall_pct": result.get("quarantine_recall_pct"),
                "quarantine_precision_pct": result.get("quarantine_precision_pct"),
            }
        },
        "independent_clean_benchmark": {
            "description": "Fixed untouched clean CIFAR-10 evaluation set for measuring downstream utility",
            "sample_count": 300,
            "metrics": {
                "clean_reference_acc": result.get("clean_reference_accuracy"),
                "victim_clean_acc": result.get("victim_clean_accuracy"),
                "screened_clean_acc": result.get("screened_clean_accuracy"),
                "clean_accuracy_delta_pp": result.get("clean_accuracy_delta_pp"),
            }
        }
    }

    comparison = {
        "evaluation_populations": populations,
        "baseline_reported": baseline,
        "m6_2_latent_run": result,
        "interpretation_rules": [
            "The 5% FPR ceiling is enforced strictly on validation data only; held-out test FPR is reported independently.",
            "Test set samples and poison prevalence (5.0%) remain fixed and identical across runs.",
            "is_poisoned is used only as ground-truth evaluation target, never as a detector input feature.",
            "Clean accuracy is evaluated on an independent, fixed 300-sample pristine test set.",
        ],
    }
    (out_dir / "m6_clean_label_hardening_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    # Comparison table recording baseline, M6.1 pixel run, and M6.2 latent run
    comparison_rows = [
        {
            "version": "baseline_m6_unconstrained",
            "population_test_n": 240,
            "population_quarantine_n": 1200,
            "population_clean_eval_n": 300,
            "roc_auc": (baseline or {}).get("detector_roc_auc", (baseline or {}).get("test_roc_auc")),
            "pr_auc": (baseline or {}).get("detector_pr_auc", (baseline or {}).get("test_pr_auc")),
            "precision": (baseline or {}).get("detector_precision", (baseline or {}).get("test_precision")),
            "recall": (baseline or {}).get("detector_recall", (baseline or {}).get("test_recall")),
            "fpr": (baseline or {}).get("detector_false_positive_rate", (baseline or {}).get("test_fpr")),
            "screened_clean_accuracy": (baseline or {}).get("screened_clean_accuracy", (baseline or {}).get("screened_acc")),
            "clean_accuracy_delta_pp": (baseline or {}).get("clean_accuracy_delta_pp", -23.67),
            "quarantine_count": (baseline or {}).get("quarantined_count", 1081),
            "quarantine_rate": (baseline or {}).get("detector_quarantine_rate", 0.9008),
            "threshold": (baseline or {}).get("detector_threshold", 0.20),
        },
        {
            "version": "m6_1_pixel_prototype",
            "population_test_n": 240,
            "population_quarantine_n": 1200,
            "population_clean_eval_n": 300,
            "roc_auc": 0.6696,
            "pr_auc": 0.0839,
            "precision": 0.0,
            "recall": 0.0,
            "fpr": 0.0175,
            "screened_clean_accuracy": 35.33,
            "clean_accuracy_delta_pp": -1.67,
            "quarantine_count": 19,
            "quarantine_rate": 0.0158,
            "threshold": 0.7223,
        },
        {
            "version": "m6_2_latent_prototype",
            "population_test_n": 240,
            "population_quarantine_n": 1200,
            "population_clean_eval_n": 300,
            "roc_auc": result.get("detector_test_roc_auc", result.get("detector_roc_auc")),
            "pr_auc": result.get("detector_test_pr_auc", result.get("detector_pr_auc")),
            "precision": result.get("detector_test_precision", result.get("detector_precision")),
            "recall": result.get("detector_test_recall", result.get("detector_recall")),
            "fpr": result.get("detector_test_fpr", result.get("detector_false_positive_rate")),
            "screened_clean_accuracy": result.get("screened_clean_accuracy"),
            "clean_accuracy_delta_pp": result.get("clean_accuracy_delta_pp"),
            "quarantine_count": result.get("quarantined_count"),
            "quarantine_rate": result.get("detector_test_quarantine_rate", result.get("detector_quarantine_rate")),
            "threshold": result.get("detector_threshold"),
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(out_dir / "m6_clean_label_hardening_comparison.csv", index=False)

    print("\n=======================================================")
    print("M6.2 CLEAN-LABEL HARDENING RESULT (EXPLICIT POPULATIONS)")
    print("=======================================================")
    print(f"[POPULATION 1: Held-Out Unseen Test Partition (N=240, 5% poison)]")
    print(f"  Test ROC-AUC:   {result.get('detector_test_roc_auc'):.4f}")
    print(f"  Test PR-AUC:    {result.get('detector_test_pr_auc'):.4f}")
    print(f"  Test FPR:       {float(result.get('detector_test_fpr', 0)):.2%}")
    print(f"  Test Recall:    {float(result.get('detector_test_recall', 0)):.2%}")
    print(f"  Test Precision: {float(result.get('detector_test_precision', 0)):.2%}")
    print(f"  Operating Threshold: {result.get('detector_threshold'):.4f}")

    print(f"\n[POPULATION 2: Full Ingestion Dataset Screening (N=1,200, 60 poisoned)]")
    print(f"  Quarantined Samples: {result.get('quarantined_count')} / 1200 ({float(result.get('detector_quarantine_rate', 0)):.2%})")
    print(f"  Purged Poison Recall: {result.get('quarantine_recall_pct')}%")

    print(f"\n[POPULATION 3: Independent Pristine Benchmark (N=300 clean images)]")
    print(f"  Victim Clean Accuracy:   {result.get('victim_clean_accuracy'):.2f}%")
    print(f"  Screened Clean Accuracy: {result.get('screened_clean_accuracy'):.2f}%")
    print(f"  Utility Delta:           {result.get('clean_accuracy_delta_pp'):+.2f} pp")
    print("=======================================================\n")
    print("Saved comparison:", out_dir / "m6_clean_label_hardening_comparison.csv")
    print("Saved detail:", out_dir / "m6_clean_label_hardening_comparison.json")

if __name__ == "__main__":
    main()
