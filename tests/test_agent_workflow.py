"""
Test LangGraph agent workflow execution on registered audit data.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.audit_store import register_dataset
from core.detector_adapters import adapt_tabular_audit_to_unified
from agents.workflow import agent_workflow


def test_agent_workflow():
    print("[*] Testing Agent Workflow...")
    
    # 1. Register a test dataset
    raw_audit = {
        "dataset_id": "test_workflow_01",
        "n_rows": 1000,
        "n_columns": 24,
        "mode": "SUPERVISED",
        "dataset_risk_score": 45.0,
        "dataset_trust_score": 55.0,
        "dataset_risk_level": "MODERATE_RISK",
        "status": "SUSPICIOUS",
        "sample_results": [
            {
                "sample_id": "101",
                "risk_score": 88.0,
                "risk_level": "HIGH",
                "cleanlab_risk": 0.95,
                "label_disagreement": 0.90,
                "prediction_entropy": 0.70
            }
        ]
    }
    unified = adapt_tabular_audit_to_unified(raw_audit, dataset_name="credit_sample.csv")
    register_dataset("test_workflow_01", "runtime/datasets/test.csv", unified.to_legacy_dict())
    
    # 2. Invoke Workflow
    result = agent_workflow.invoke({
        "dataset_id": "test_workflow_01",
        "question": "What is the primary risk and which samples should I quarantine?"
    })
    
    assert "audit_summary" in result
    assert "investigation" in result
    assert "final_report" in result
    print("  [OK] Agent workflow executed successfully!")
    print("\n--- Audit Summary Preview ---")
    safe_audit = result["audit_summary"][:250].encode("ascii", "replace").decode("ascii")
    print(safe_audit + "...")
    print("\n--- Final Report Preview ---")
    safe_report = result["final_report"][:350].encode("ascii", "replace").decode("ascii")
    print(safe_report + "...")


if __name__ == "__main__":
    test_agent_workflow()
    print("\n[SUCCESS] Agent Workflow Test Passed!")
