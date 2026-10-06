from __future__ import annotations

import json
from langchain_core.tools import tool

from services.audit_store import (
    get_audit,
    get_dataset
)

from services.datasentinel_bridge import (
    run_tabular_audit
)


@tool
def run_audit_tool(
    dataset_id: str
) -> str:
    """
    Run the existing DataSentinel ML audit on a
    previously uploaded dataset.
    """
    dataset = get_dataset(
        dataset_id
    )

    result = run_tabular_audit(
        dataset["file_path"]
    )

    # Refresh stored result
    dataset["audit"] = result

    return format_summary(
        result
    )


@tool
def get_audit_summary_tool(
    dataset_id: str
) -> str:
    """
    Return the overall DataSentinel audit summary.
    """
    result = get_audit(
        dataset_id
    )

    return format_summary(
        result
    )


@tool
def get_top_suspicious_samples_tool(
    dataset_id: str,
    limit: int = 10
) -> str:
    """
    Return the highest-risk samples identified
    by DataSentinel.
    """
    result = get_audit(
        dataset_id
    )

    samples = result.get(
        "sample_results",
        []
    )

    def extract_risk(s: dict) -> float:
        val = s.get("risk_score", s.get("risk_probability", 0))
        try:
            return float(val or 0)
        except (ValueError, TypeError):
            return 0.0

    sorted_samples = sorted(
        samples,
        key=extract_risk,
        reverse=True
    )

    top_samples = sorted_samples[:max(1, min(limit, 25))]
    return json.dumps(top_samples, indent=2, default=str)


@tool
def get_sample_evidence_tool(
    dataset_id: str,
    sample_id: str
) -> str:
    """
    Explain the detector evidence available
    for one specific sample.
    """
    result = get_audit(
        dataset_id
    )

    samples = result.get(
        "sample_results",
        []
    )

    target_id_str = str(sample_id).strip()

    for sample in samples:
        possible_ids = [
            sample.get("sample_id"),
            sample.get("original_index"),
            sample.get("index")
        ]

        if any(
            str(value).strip() == target_id_str
            for value in possible_ids
            if value is not None
        ):
            return json.dumps(sample, indent=2, default=str)

    return (
        f"No evidence found for sample {sample_id}."
    )


@tool
def get_detector_breakdown_tool(
    dataset_id: str
) -> str:
    """
    Return the available evidence/detector
    summary for a dataset.
    """
    result = get_audit(
        dataset_id
    )

    candidates = [
        "detector_summary",
        "signal_summary",
        "evidence_summary",
        "signals",
        "detectors",
        "signal_breakdown"
    ]

    for key in candidates:
        if key in result and result[key]:
            return json.dumps(result[key], indent=2, default=str)

    return (
        "Detector-level breakdown was not "
        "exposed by the current audit result."
    )


def format_summary(
    result: dict
) -> str:
    summary = {
        "mode": result.get("mode"),
        "rows": result.get("n_rows"),
        "columns": result.get("n_columns"),
        "risk_score": result.get(
            "dataset_risk_score",
            result.get(
                "risk_score"
            )
        ),
        "trust_score": result.get(
            "dataset_trust_score",
            result.get(
                "trust_score"
            )
        ),
        "risk_level": result.get(
            "dataset_risk_level",
            result.get(
                "risk_level"
            )
        ),
        "evidence_coverage": result.get(
            "evidence_coverage"
        ),
        "recommended_action": result.get(
            "recommended_action"
        ),
        "high_risk_samples": result.get(
            "high_risk_quarantine",
            result.get(
                "high_risk_samples"
            )
        ),
    }

    return json.dumps(summary, indent=2, default=str)
