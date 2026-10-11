"""
Unit test for Milestone 8 Unified Audit Schema, Risk Policy, and Detector Adapters.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.unified_schema import UnifiedAuditResult
from core.detector_adapters import (
    adapt_tabular_audit_to_unified,
    adapt_image_audit_to_unified
)


def test_tabular_adaptation():
    print("[*] Testing Tabular Audit Adapter...")
    sample_audit = {
        "dataset_id": "test_tab_01",
        "n_rows": 1000,
        "n_columns": 24,
        "mode": "SUPERVISED",
        "dataset_risk_score": 18.5,
        "dataset_trust_score": 81.5,
        "dataset_risk_level": "LOW_RISK",
        "status": "AUDITED_SECURE",
        "evidence_coverage": 1.0,
        "sample_results": [
            {
                "sample_id": "0",
                "risk_score": 85.0,
                "risk_level": "HIGH",
                "cleanlab_risk": 0.90,
                "label_disagreement": 0.85,
                "prediction_entropy": 0.75,
                "isolation_forest_risk": 0.60
            },
            {
                "sample_id": "1",
                "risk_score": 15.0,
                "risk_level": "LOW",
                "cleanlab_risk": 0.10,
                "label_disagreement": 0.05,
                "prediction_entropy": 0.20,
                "isolation_forest_risk": 0.15
            }
        ]
    }

    unified = adapt_tabular_audit_to_unified(sample_audit, dataset_name="credit_sample.csv")
    assert isinstance(unified, UnifiedAuditResult)
    assert unified.modality == "TABULAR"
    assert unified.n_rows == 1000
    assert unified.trust_score == round(100.0 - unified.risk_score, 2)
    assert len(unified.available_signals) > 0
    assert unified.quarantine_manifest.requires_human_review is True
    print(f"  [OK] Tabular unified audit valid! Risk={unified.risk_score}, Trust={unified.trust_score}, Signals={len(unified.available_signals)}")


def test_image_adaptation():
    print("[*] Testing Image Deep Learning Adapter...")
    # Load real M6 backdoor or clean_label audit if present
    audit_file = ROOT / "experiments/image/clean_label/attack_summary.json"
    evidence_file = ROOT / "experiments/image/clean_label/sample_evidence.csv"

    if audit_file.exists():
        with open(audit_file, "r", encoding="utf-8") as f:
            raw_audit = json.load(f)
    else:
        raw_audit = {
            "attack": "clean_label",
            "clean_ref_acc": 39.67,
            "victim_acc": 36.0,
            "screened_acc": 38.0,
            "victim_asr": 8.55,
            "screened_asr": 8.55,
            "quarantined_count": 1
        }

    unified = adapt_image_audit_to_unified(
        raw_audit,
        dataset_name="cifar10_clean_label.png",
        evidence_csv_path=str(evidence_file) if evidence_file.exists() else None
    )

    assert isinstance(unified, UnifiedAuditResult)
    assert unified.modality == "IMAGE"
    assert len(unified.unresolved_threat_limitations) > 0
    assert unified.unresolved_threat_limitations[0].threat_family == "clean_label_manipulation"
    assert unified.unresolved_threat_limitations[0].status == "UNRESOLVED_LIMITATION"
    assert unified.model_utility is not None
    assert unified.model_utility.clean_reference_accuracy is not None
    assert unified.quarantine_manifest.requires_human_review is True

    # Test legacy export
    legacy = unified.to_legacy_dict()
    assert "dataset_risk_score" in legacy
    assert "high_risk_quarantine" in legacy

    print(f"  [OK] Image unified audit valid! Risk={unified.risk_score}, Coverage={unified.evidence_coverage*100:.1f}%, Limitation attached: {unified.unresolved_threat_limitations[0].threat_family}")


if __name__ == "__main__":
    test_tabular_adaptation()
    test_image_adaptation()
    print("\n[SUCCESS] All Unified Schema and Adapter tests passed!")
