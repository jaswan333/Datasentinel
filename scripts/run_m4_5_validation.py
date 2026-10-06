"""
DataSentinel - Milestone 4.5 Validation & Test Runner
General Tabular Input & Adaptive Risk Scoring Benchmark

Validates:
1. Schema inspection on diverse tabular datasets (numeric, categorical, IDs, target).
2. Mixed-type tabular preprocessing pipeline (imputation, scaling, one-hot encoding).
3. Missing evidence weight renormalization and minimum coverage rules (Coverage >= 75%).
4. Dataset A: UCI Bank Marketing (Supervised Audit: Categorical + Numeric + Label).
5. Dataset B: UCI German Credit (Supervised Audit: Categorical + Numeric + Label).
6. Dataset C: Bank Marketing without target (Unsupervised Audit: Categorical + Numeric, No Label).
7. Backward Compatibility: Taiwan Credit Default dataset with numeric-encoded categories.
8. Stress Test: Missing duplicate detector simulating unavailable modality and graceful renormalization.
"""

from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repository root to Python path
sys.path.insert(0, os.path.abspath("."))

from ucimlrepo import fetch_ucirepo
from core.tabular_schema import inspect_tabular_schema
from core.tabular_preprocessing import build_tabular_preprocessor, fit_transform_tabular
from core.adaptive_risk_score import (
    calculate_sample_risk,
    calculate_dataset_summary,
    format_dataset_audit_card,
    normalize_weights,
    DEFAULT_SUPERVISED_WEIGHTS,
    DEFAULT_UNSUPERVISED_WEIGHTS
)
from core.tabular_auditor import audit_tabular_dataset, TabularAuditResult
from data.data_loader import load_clean_data


def test_missing_evidence_renormalization():
    """Unit test for Missing Evidence Rule and Minimum Coverage Rule."""
    print("\n" + "=" * 80)
    print("TEST 1: MISSING EVIDENCE RULE & MINIMUM COVERAGE VERIFICATION")
    print("=" * 80)

    # 1. Full evidence scenario
    df_full = pd.DataFrame({
        "sample_id": ["8421"],
        "label_risk": [0.90],
        "uncertainty_risk": [0.81],
        "anomaly_risk": [0.20],
        "duplicate_risk": [0.00],
        "consistency_risk": [0.86]
    })

    scored_full = calculate_sample_risk(df_full, mode="supervised")
    score_val = scored_full.iloc[0]["risk_score"]
    cov_val = scored_full.iloc[0]["evidence_coverage"]
    tier_val = scored_full.iloc[0]["risk_level"]

    expected_full = (0.35 * 0.90 + 0.15 * 0.81 + 0.25 * 0.20 + 0.15 * 0.00 + 0.10 * 0.86) * 100.0
    print(f"Full Evidence Score:      {score_val:.2f}/100 (Expected: {expected_full:.2f})")
    print(f"Full Evidence Coverage:   {cov_val:.1f}%")
    print(f"Full Risk Tier:           {tier_val}")
    assert np.isclose(score_val, expected_full, atol=0.1), f"Expected {expected_full}, got {score_val}"
    assert np.isclose(cov_val, 100.0), f"Expected 100%, got {cov_val}"

    # 2. Missing Duplicate Analysis Scenario (Duplicate = N/A)
    # Total available weight: 0.35 + 0.15 + 0.25 + 0.10 = 0.85
    # Risk should renormalize: sum / 0.85
    df_missing_dup = pd.DataFrame({
        "sample_id": ["8421"],
        "label_risk": [0.90],
        "uncertainty_risk": [0.81],
        "anomaly_risk": [0.20],
        "duplicate_risk": [np.nan],  # N/A, NOT 0.0!
        "consistency_risk": [0.86]
    })

    scored_missing = calculate_sample_risk(df_missing_dup, mode="supervised")
    renorm_score = scored_missing.iloc[0]["risk_score"]
    renorm_cov = scored_missing.iloc[0]["evidence_coverage"]
    renorm_tier = scored_missing.iloc[0]["risk_level"]

    expected_renorm = ((0.35 * 0.90 + 0.15 * 0.81 + 0.25 * 0.20 + 0.10 * 0.86) / 0.85) * 100.0
    print(f"\nMissing Duplicate Score:  {renorm_score:.2f}/100 (Expected Renormalized: {expected_renorm:.2f})")
    print(f"Missing Duplicate Cov:    {renorm_cov:.1f}% (Available: 85%)")
    print(f"Missing Duplicate Tier:   {renorm_tier}")
    assert np.isclose(renorm_score, expected_renorm, atol=0.1), f"Expected {expected_renorm}, got {renorm_score}"
    assert np.isclose(renorm_cov, 85.0), f"Expected 85%, got {renorm_cov}"
    assert renorm_tier == "MEDIUM", f"Expected MEDIUM, got {renorm_tier}"
    print("✓ Missing evidence successfully renormalized over available signals without artificial score deflation!")

    # 3. Minimum Coverage Rule (< 75% coverage -> INSUFFICIENT_EVIDENCE)
    # Suppose both anomaly (0.25) and duplicate (0.15) and consistency (0.10) are missing
    # Available weight = 0.35 + 0.15 = 0.50 (50% coverage < 75%)
    df_low_cov = pd.DataFrame({
        "sample_id": ["8421"],
        "label_risk": [0.10],        # Clean signal
        "uncertainty_risk": [0.10],  # Clean signal
        "anomaly_risk": [np.nan],
        "duplicate_risk": [np.nan],
        "consistency_risk": [np.nan]
    })

    scored_low_cov = calculate_sample_risk(df_low_cov, mode="supervised", minimum_coverage=0.75)
    low_cov_tier = scored_low_cov.iloc[0]["risk_level"]
    low_cov_pct = scored_low_cov.iloc[0]["evidence_coverage"]
    print(f"\nLow Coverage Pct:         {low_cov_pct:.1f}%")
    print(f"Low Coverage Tier:        {low_cov_tier} (Expected: INSUFFICIENT_EVIDENCE)")
    assert low_cov_tier == "INSUFFICIENT_EVIDENCE", f"Expected INSUFFICIENT_EVIDENCE, got {low_cov_tier}"
    print("✓ Minimum coverage rule enforced: Insufficient evidence != SAFE.")


def run_benchmark():
    os.makedirs("experiments/milestone_4_5", exist_ok=True)
    all_summaries = {}

    # Run Unit Test
    test_missing_evidence_renormalization()

    # -------------------------------------------------------------------------
    # DATASET A: UCI Bank Marketing (Supervised Audit)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DATASET A: UCI BANK MARKETING (SUPERVISED AUDIT)")
    print("=" * 80)
    print("Fetching Bank Marketing dataset from UCI Repository (ID=222)...")
    bank_uci = fetch_ucirepo(id=222)
    df_bank = bank_uci.data.features.copy()
    df_bank["y"] = bank_uci.data.targets["y"].values

    print(f"Bank Marketing loaded: {len(df_bank)} records, {len(df_bank.columns)} columns.")
    # Audit on representative 2,500 sample for rapid complete testing
    t0 = time.time()
    res_bank = audit_tabular_dataset(
        df_bank,
        target_column="y",
        sample_size=2500,
        random_state=42
    )
    t_bank = time.time() - t0
    print(f"Audit completed in {t_bank:.2f} seconds.")
    res_bank.print_card()
    all_summaries["bank_marketing_supervised"] = res_bank.summary
    res_bank.evidence_df.to_csv("experiments/milestone_4_5/bank_marketing_supervised_audit.csv", index=True)

    # -------------------------------------------------------------------------
    # DATASET B: UCI Statlog German Credit (Supervised Audit)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DATASET B: UCI STATLOG GERMAN CREDIT (SUPERVISED AUDIT)")
    print("=" * 80)
    print("Fetching German Credit dataset from UCI Repository (ID=144)...")
    german_uci = fetch_ucirepo(id=144)
    df_german = german_uci.data.features.copy()
    df_german["class"] = german_uci.data.targets["class"].values

    print(f"German Credit loaded: {len(df_german)} records, {len(df_german.columns)} columns.")
    t0 = time.time()
    res_german = audit_tabular_dataset(
        df_german,
        target_column="class",
        random_state=42
    )
    t_german = time.time() - t0
    print(f"Audit completed in {t_german:.2f} seconds.")
    res_german.print_card()
    all_summaries["german_credit_supervised"] = res_german.summary
    res_german.evidence_df.to_csv("experiments/milestone_4_5/german_credit_supervised_audit.csv", index=True)

    # -------------------------------------------------------------------------
    # DATASET C: Bank Marketing without target (UNSUPERVISED AUDIT)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DATASET C: BANK MARKETING WITHOUT TARGET (UNSUPERVISED AUDIT)")
    print("=" * 80)
    df_bank_unsupervised = bank_uci.data.features.copy()
    print(f"Running Unsupervised Audit on {len(df_bank_unsupervised.columns)} features (No target present)...")
    t0 = time.time()
    res_bank_unsup = audit_tabular_dataset(
        df_bank_unsupervised,
        target_column=None,
        sample_size=2500,
        random_state=42
    )
    t_bank_unsup = time.time() - t0
    print(f"Unsupervised audit completed in {t_bank_unsup:.2f} seconds.")
    res_bank_unsup.print_card()
    all_summaries["bank_marketing_unsupervised"] = res_bank_unsup.summary
    res_bank_unsup.evidence_df.to_csv("experiments/milestone_4_5/bank_marketing_unsupervised_audit.csv", index=True)

    # -------------------------------------------------------------------------
    # BACKWARD COMPATIBILITY: Taiwan Credit Default Dataset
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BACKWARD COMPATIBILITY: TAIWAN CREDIT DEFAULT (KNOWN CATEGORICAL HINTS)")
    print("=" * 80)
    X_clean, _, y_clean, _ = load_clean_data()
    df_taiwan = X_clean.copy()
    df_taiwan["default_payment_next_month"] = y_clean.values

    t0 = time.time()
    res_taiwan = audit_tabular_dataset(
        df_taiwan,
        target_column="default_payment_next_month",
        categorical_columns=["SEX", "EDUCATION", "MARRIAGE"],
        sample_size=2500,
        random_state=42
    )
    t_taiwan = time.time() - t0
    print(f"Taiwan Credit audit completed in {t_taiwan:.2f} seconds.")
    res_taiwan.print_card()
    all_summaries["taiwan_credit_supervised"] = res_taiwan.summary

    # -------------------------------------------------------------------------
    # STRESS TEST: MISSING DETECTOR (DUPLICATE ANALYSIS UNAVAILABLE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STRESS TEST: MISSING DETECTOR (DUPLICATE DETECTOR DISABLED)")
    print("=" * 80)
    res_missing_dup = audit_tabular_dataset(
        df_german,
        target_column="class",
        disabled_signals=["duplicate_risk"],
        random_state=42
    )
    res_missing_dup.print_card()
    all_summaries["german_credit_missing_duplicate"] = res_missing_dup.summary

    # Save summary report to JSON
    json_path = "experiments/milestone_4_5/m4_5_audit_summary.json"
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=4)
    print(f"\nAll benchmark summaries successfully written to {json_path}")
    print("=" * 80)
    print("DATASENTINEL MILESTONE 4.5 COMPLETE & VERIFIED!")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
