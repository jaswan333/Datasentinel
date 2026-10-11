from __future__ import annotations

from pathlib import Path
from typing import Any
import json


DATASET_STORE: dict[str, dict[str, Any]] = {}


def register_dataset(
    dataset_id: str,
    file_path: str,
    audit_result: dict[str, Any]
) -> None:

    DATASET_STORE[dataset_id] = {
        "dataset_id": dataset_id,
        "file_path": file_path,
        "audit": audit_result
    }

    Path("runtime/results").mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        f"runtime/results/{dataset_id}.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            audit_result,
            f,
            indent=2,
            default=str
        )


def get_dataset(dataset_id: str) -> dict[str, Any]:
    if dataset_id not in DATASET_STORE:
        # Fallback to persistent disk cache if process restarted
        result_path = Path("runtime/results") / f"{dataset_id}.json"
        if result_path.exists():
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    audit_data = json.load(f)
                DATASET_STORE[dataset_id] = {
                    "dataset_id": dataset_id,
                    "file_path": "",
                    "audit": audit_data
                }
                return DATASET_STORE[dataset_id]
            except Exception:
                pass

        raise KeyError(
            f"Dataset '{dataset_id}' not found."
        )

    return DATASET_STORE[dataset_id]


def get_audit(dataset_id: str) -> dict[str, Any]:
    return get_dataset(dataset_id)["audit"]


def list_audit_history(limit: int = 15) -> list[dict[str, Any]]:
    """Lists recent audit metadata sorted by file modification timestamp."""
    results_dir = Path("runtime/results")
    if not results_dir.exists():
        return []

    entries = []
    for file_path in results_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            created_at = data.get("created_at")
            mtime = file_path.stat().st_mtime
            if not created_at:
                import datetime
                created_at = datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc).isoformat()

            q_manifest = data.get("quarantine_manifest") or {}
            unresolved = data.get("unresolved_threat_limitations") or []

            entries.append({
                "dataset_id": file_path.stem,
                "dataset_name": data.get("dataset_name", file_path.stem),
                "modality": data.get("modality", "TABULAR"),
                "created_at": created_at,
                "risk_score": data.get("risk_score", 0.0),
                "trust_score": data.get("trust_score", 100.0),
                "risk_level": data.get("risk_level", "LOW_RISK"),
                "overall_status": data.get("overall_status", "AUDITED_SECURE"),
                "evidence_coverage": data.get("evidence_coverage", 1.0),
                "quarantined_count": q_manifest.get("quarantined_count", 0),
                "total_candidate_samples": q_manifest.get("total_candidate_samples", data.get("n_rows", 0)),
                "confirmation_status": q_manifest.get("confirmation_status", "PENDING_OPERATOR_REVIEW"),
                "has_unresolved_limitations": len(unresolved) > 0,
                "file_mtime": mtime
            })
        except Exception:
            continue

    entries.sort(key=lambda x: x["file_mtime"], reverse=True)
    return entries[:limit]
