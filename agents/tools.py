"""
DataSentinel - Agent Tools (Milestone 8)
Grounds agents directly in verified audit evidence, signal attributions,
quarantine manifests, and explicit threat limitations.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List
from langchain_core.tools import tool

from services.audit_store import get_audit, get_dataset
from services.datasentinel_bridge import run_tabular_audit


@tool
def run_audit_tool(dataset_id: str) -> str:
    """Run the existing DataSentinel ML audit on an uploaded dataset."""
    dataset = get_dataset(dataset_id)
    result = run_tabular_audit(dataset["file_path"])
    dataset["audit"] = result
    return format_summary(result)


@tool
def get_audit_summary_tool(dataset_id: str) -> str:
    """Return the overall DataSentinel audit summary including risk, trust score, and coverage."""
    result = get_audit(dataset_id)
    return format_summary(result)


@tool
def get_top_suspicious_samples_tool(dataset_id: str, limit: int = 10) -> str:
    """Return the highest-risk samples identified by DataSentinel."""
    result = get_audit(dataset_id)
    samples = result.get("sample_results", [])

    def extract_risk(s: dict) -> float:
        val = s.get("risk_score", s.get("risk_probability", 0))
        try:
            return float(val or 0)
        except (ValueError, TypeError):
            return 0.0

    sorted_samples = sorted(samples, key=extract_risk, reverse=True)
    top_samples = sorted_samples[:max(1, min(limit, 25))]
    return json.dumps(top_samples, indent=2, default=str)


@tool
def get_sample_evidence_tool(dataset_id: str, sample_id: str) -> str:
    """Explain the detector evidence available for one specific sample ID."""
    result = get_audit(dataset_id)
    samples = result.get("sample_results", [])
    target_id_str = str(sample_id).strip()

    for sample in samples:
        possible_ids = [
            sample.get("sample_id"),
            sample.get("original_index"),
            sample.get("index")
        ]
        if any(str(v).strip() == target_id_str for v in possible_ids if v is not None):
            return json.dumps(sample, indent=2, default=str)

    return f"No evidence recorded for sample ID '{sample_id}'."


@tool
def get_detector_breakdown_tool(dataset_id: str) -> str:
    """Return the available evidence signal and detector breakdown for a dataset."""
    result = get_audit(dataset_id)

    candidates = [
        "signal_breakdown",
        "detector_summary",
        "signal_summary",
        "evidence_summary",
        "signals",
        "detectors"
    ]
    for key in candidates:
        if key in result and result[key]:
            return json.dumps(result[key], indent=2, default=str)

    return "Detector-level breakdown was not recorded for this audit."


@tool
def get_threat_limitations_tool(dataset_id: str) -> str:
    """Return documented threat limitations and boundaries (e.g. clean-label manipulation)."""
    result = get_audit(dataset_id)
    limitations = result.get("unresolved_threat_limitations", [])
    if limitations:
        return json.dumps(limitations, indent=2, default=str)

    # If modality is vision or clean_label threat is present, provide standard limitation
    modality = str(result.get("mode") or result.get("modality") or "").upper()
    if "IMAGE" in modality or "clean_label" in str(result.get("primary_threat")):
        return json.dumps([{
            "threat_family": "clean_label_manipulation",
            "status": "UNRESOLVED_LIMITATION",
            "statement": (
                "Clean-label centroid manipulation remains an unresolved detector limitation "
                "under a 5% false-positive ceiling. Observed test recall was 0/12. "
                "Reliable post-hoc detection is not claimed."
            ),
            "mitigation_recommendation": "Use Mixup defensive training during model optimization."
        }], indent=2)

    return "No unresolved limitations recorded for this dataset modality."


@tool
def get_quarantine_manifest_tool(dataset_id: str) -> str:
    """Return the auditable quarantine manifest and non-destructive screening status."""
    result = get_audit(dataset_id)
    manifest = result.get("quarantine_manifest")
    if manifest:
        return json.dumps(manifest, indent=2, default=str)

    count = result.get("high_risk_quarantine", result.get("quarantined_count", 0))
    total = result.get("n_rows", result.get("total_samples", 0))
    return json.dumps({
        "quarantined_count": count,
        "total_candidate_samples": total,
        "requires_human_review": True,
        "quarantine_policy": "Non-destructive isolation requiring human confirmation"
    }, indent=2)


def format_summary(result: dict) -> str:
    """Formats standardized audit dictionary for agent reasoning."""
    summary = {
        "dataset_id": result.get("dataset_id"),
        "dataset_name": result.get("dataset_name", result.get("filename", "unknown")),
        "modality": result.get("modality", result.get("mode", "TABULAR")),
        "rows": result.get("n_rows", result.get("total_samples")),
        "columns": result.get("n_columns"),
        "risk_score": result.get("risk_score", result.get("dataset_risk_score")),
        "trust_score": result.get("trust_score", result.get("dataset_trust_score")),
        "risk_level": result.get("risk_level", result.get("dataset_risk_level")),
        "overall_status": result.get("overall_status", result.get("status")),
        "evidence_coverage": result.get("evidence_coverage", "100.0%"),
        "primary_threat": result.get("primary_threat_family", result.get("primary_threat")),
        "quarantined_count": result.get("high_risk_quarantine", result.get("quarantined_count", 0)),
        "recommended_action": result.get("recommended_action"),
        "has_unresolved_limitations": bool(result.get("unresolved_threat_limitations"))
    }
    return json.dumps(summary, indent=2, default=str)
