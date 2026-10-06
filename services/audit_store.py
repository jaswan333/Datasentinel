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
