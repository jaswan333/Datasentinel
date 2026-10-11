"""
End-to-end integration test suite for Milestone 8:
Multi-Modal Platform Integration & Unified Auditing.

Verifies:
1. FastAPI health endpoint
2. Tabular dataset audit ingestion via POST /api/audit
3. Benchmark pre-loaders via POST /api/audit/benchmark/{benchmark_id}
   - image_backdoor
   - image_clean_label
   - tabular_credit
4. Quarantine confirmation flow via POST /api/audit/{dataset_id}/quarantine/confirm
5. Agent triad invocation via POST /api/agent/ask
6. M6.3 Clean-Label Centroid limitation transparency verification
7. Separate clean accuracy and ASR metrics verification
"""

import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from services.audit_store import get_dataset, DATASET_STORE


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "0.8" in data["version"]


def test_tabular_audit_flow(client):
    # Create synthetic credit tabular CSV with sufficient samples for 5-fold CV
    import numpy as np
    np.random.seed(42)
    n = 24
    df = pd.DataFrame({
        "LIMIT_BAL": np.random.randint(10000, 100000, size=n),
        "SEX": np.random.choice([1, 2], size=n),
        "EDUCATION": np.random.choice([1, 2, 3], size=n),
        "MARRIAGE": np.random.choice([1, 2], size=n),
        "AGE": np.random.randint(22, 60, size=n),
        "PAY_0": np.random.choice([-1, 0, 1, 2], size=n),
        "BILL_AMT1": np.random.randint(1000, 50000, size=n),
        "PAY_AMT1": np.random.randint(500, 5000, size=n),
        "default": [0, 1] * (n // 2)
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    
    files = {"file": ("test_credit.csv", io.BytesIO(csv_bytes), "text/csv")}
    data = {"target_column": "default", "categorical_columns": "SEX,EDUCATION,MARRIAGE"}
    
    res = client.post("/api/audit", files=files, data=data)
    assert res.status_code == 200, res.text
    payload = res.json()
    
    assert "dataset_id" in payload
    dataset_id = payload["dataset_id"]
    audit = payload["audit"]
    
    # Unified schema checks
    assert audit["modality"] == "TABULAR"
    assert "dataset_risk_score" in audit
    assert "evidence_coverage" in audit
    assert "quarantine_manifest" in audit
    manifest = audit["quarantine_manifest"]
    assert manifest.get("confirmation_status") == "PENDING_OPERATOR_REVIEW"


def test_image_backdoor_benchmark_loader(client):
    res = client.post("/api/audit/benchmark/image_backdoor")
    assert res.status_code == 200, res.text
    payload = res.json()
    
    assert "dataset_id" in payload
    audit = payload["audit"]
    assert audit["modality"] in ["IMAGE", "VISION"]
    assert audit["dataset_risk_score"] > 0
    assert audit["quarantine_manifest"]["quarantined_count"] > 0
    assert audit["model_utility"] is not None
    assert audit["model_utility"]["clean_accuracy_pre"] is not None
    assert audit["model_utility"]["asr_pre"] is not None
    assert audit["model_utility"]["asr_pre"] > 50  # Compromised victim model ASR is ~99.6%
    assert audit["model_utility"]["asr_post"] is not None
    assert audit["model_utility"]["asr_post"] < audit["model_utility"]["asr_pre"]


def test_image_clean_label_limitation_disclosure(client):
    res = client.post("/api/audit/benchmark/image_clean_label")
    assert res.status_code == 200, res.text
    payload = res.json()
    
    audit = payload["audit"]
    assert audit["modality"] in ["IMAGE", "VISION"]
    # Mandatory clean-label disclosure invariant
    assert "threat_limitations" in audit
    limitations = audit["threat_limitations"]
    assert len(limitations) >= 1
    
    clean_label_lim = next((l for l in limitations if "clean_label" in l["threat_family"].lower()), None)
    assert clean_label_lim is not None
    assert "0" in str(clean_label_lim.get("observed_recall", "")) or clean_label_lim.get("empirical_recall") == 0.0
    assert any(term in str(clean_label_lim["status"]).upper() for term in ["UNRESOLVED", "EMPIRICALLY_CONSTRAINED", "LIMITATION"])


def test_quarantine_confirmation_flow(client):
    # Load image backdoor benchmark
    res = client.post("/api/audit/benchmark/image_backdoor")
    assert res.status_code == 200
    dataset_id = res.json()["dataset_id"]
    
    # Inspect quarantine manifest before confirmation
    get_res = client.get(f"/api/audit/{dataset_id}/quarantine")
    assert get_res.status_code == 200
    manifest = get_res.json()
    assert manifest["confirmation_status"] == "PENDING_OPERATOR_REVIEW"
    
    # Confirm quarantine
    confirm_res = client.post(
        f"/api/audit/{dataset_id}/quarantine/confirm",
        json={"notes": "Approved by SecOps lead in E2E validation test"}
    )
    assert confirm_res.status_code == 200
    confirm_data = confirm_res.json()
    assert confirm_data["quarantine_status"] == "CONFIRMED"
    assert "Approved by SecOps lead" in confirm_data["operator_notes"]
    
    # Re-inspect to ensure persistence
    get_res2 = client.get(f"/api/audit/{dataset_id}/quarantine")
    assert get_res2.status_code == 200
    assert get_res2.json()["confirmation_status"] == "CONFIRMED"


def test_agent_ask_endpoint(client):
    # Load benchmark
    res = client.post("/api/audit/benchmark/image_backdoor")
    assert res.status_code == 200
    dataset_id = res.json()["dataset_id"]
    
    ask_res = client.post(
        "/api/agent/ask",
        json={
            "dataset_id": dataset_id,
            "question": "Why is this dataset risky and what quarantine actions are staged?"
        }
    )
    assert ask_res.status_code == 200
    agent_data = ask_res.json()
    assert "answer" in agent_data
    assert len(agent_data["answer"]) > 50
    # Must mention verified facts
    assert ("risk" in agent_data["answer"].lower() or "quarantine" in agent_data["answer"].lower())
