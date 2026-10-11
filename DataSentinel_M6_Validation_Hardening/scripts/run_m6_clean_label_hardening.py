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

from detectors.image.clean_label_signals import cross_class_prototype_affinity_risk
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
        return evidence, embeddings, labels_used

    m6.build_evidence = build_evidence_with_prototype_signal
    signal_cols = list(m6.SIGNAL_COLUMNS)
    if "cross_class_prototype_affinity_risk" not in signal_cols:
        signal_cols.append("cross_class_prototype_affinity_risk")
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
    result["extra_signal"] = "cross_class_prototype_affinity_risk"

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
                # Add explicit keys so downstream reports can consume them easily.
                result["detector_test_roc_auc"] = metrics["roc_auc"]
                result["detector_test_pr_auc"] = metrics["pr_auc"]
                result["detector_test_fpr"] = metrics["fpr"]
                result["detector_test_precision"] = metrics["precision"]
                result["detector_test_recall"] = metrics["recall"]
                result["detector_test_f1"] = metrics["f1"]
                result["detector_test_quarantine_rate"] = metrics["quarantine_rate"]

    summary_path = ROOT / "experiments/image/clean_label/summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    comparison = {
        "baseline_reported": baseline,
        "updated_run": result,
        "interpretation_rules": [
            "The 5% FPR ceiling is enforced on validation only; held-out test FPR may differ.",
            "Improved thresholded metrics do not prove improved ranking. Compare test ROC-AUC and PR-AUC too.",
            "If ROC-AUC remains near 0.5, a threshold change cannot create separation; add/replace signals or record the attack as unresolved.",
            "A recall score is not acceptable by itself if quarantine rate is very high or screened clean accuracy collapses.",
        ],
    }
    (out_dir / "m6_clean_label_hardening_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {"version": "baseline", **{
            "roc_auc": (baseline or {}).get("detector_roc_auc", (baseline or {}).get("detector_test_roc_auc")),
            "pr_auc": (baseline or {}).get("detector_pr_auc", (baseline or {}).get("detector_test_pr_auc")),
            "precision": (baseline or {}).get("detector_precision", (baseline or {}).get("detector_test_precision")),
            "recall": (baseline or {}).get("detector_recall", (baseline or {}).get("detector_test_recall")),
            "fpr": (baseline or {}).get("detector_false_positive_rate", (baseline or {}).get("detector_fpr", (baseline or {}).get("detector_test_fpr"))),
            "screened_clean_accuracy": (baseline or {}).get("screened_clean_accuracy"),
            "clean_accuracy_delta_pp": (baseline or {}).get("clean_accuracy_delta_pp"),
            "quarantine_rate": (baseline or {}).get("detector_quarantine_rate"),
            "threshold": (baseline or {}).get("detector_threshold"),
        }},
        {"version": "fpr_constrained_plus_prototype_signal", **{
            "roc_auc": result.get("detector_test_roc_auc", result.get("detector_roc_auc")),
            "pr_auc": result.get("detector_test_pr_auc", result.get("detector_pr_auc")),
            "precision": result.get("detector_test_precision", result.get("detector_precision")),
            "recall": result.get("detector_test_recall", result.get("detector_recall")),
            "fpr": result.get("detector_test_fpr", result.get("detector_false_positive_rate", result.get("detector_fpr"))),
            "screened_clean_accuracy": result.get("screened_clean_accuracy"),
            "clean_accuracy_delta_pp": result.get("clean_accuracy_delta_pp"),
            "quarantine_rate": result.get("detector_test_quarantine_rate", result.get("detector_quarantine_rate")),
            "threshold": result.get("detector_threshold"),
        }},
    ]).to_csv(out_dir / "m6_clean_label_hardening_comparison.csv", index=False)

    print("\nCLEAN-LABEL HARDENING RESULT")
    for key in ["victim_clean_accuracy", "screened_clean_accuracy", "clean_accuracy_delta_pp", "detector_threshold",
                "detector_test_roc_auc", "detector_test_pr_auc", "detector_test_precision", "detector_test_recall",
                "detector_test_f1", "detector_test_fpr", "detector_test_quarantine_rate"]:
        if key in result:
            print(f"{key}: {result[key]}")
    print("Saved comparison:", out_dir / "m6_clean_label_hardening_comparison.csv")
    print("Saved detail:", out_dir / "m6_clean_label_hardening_comparison.json")

if __name__ == "__main__":
    main()
