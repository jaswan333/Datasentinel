from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.tabular_auditor import audit_tabular_dataset, TabularAuditResult


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls"
}


def read_csv_with_fallback(source, **kwargs) -> pd.DataFrame:
    """Reads a CSV file or buffer with automatic encoding fallback (UTF-8, Windows-1252, Latin-1, etc.)."""
    encodings = ["utf-8", "windows-1252", "latin1", "cp1252", "utf-8-sig", "iso-8859-1"]
    for enc in encodings:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source, encoding="latin1", **kwargs)


def load_dataframe(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = read_csv_with_fallback(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix}"
        )

    # Check for 2-header style Excel files (e.g. Unnamed: 0 or X1..X23 where row 0 contains feature names)
    if len(df) > 1:
        first_row = df.iloc[0]
        col_names = [str(c).strip() for c in df.columns]
        if any("unnamed" in c.lower() for c in col_names) or any(c.lower().startswith("x") for c in col_names):
            if all(isinstance(v, str) and not v.strip().replace(".", "").isdigit() for v in first_row if pd.notna(v)):
                new_cols = [str(v).strip() for v in first_row]
                df = df.iloc[1:].reset_index(drop=True)
                df.columns = new_cols

    # Convert numeric columns stored as object to proper numeric types
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

    return df


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.DataFrame):
        return [
            json_safe(row)
            for row in value.to_dict(
                orient="records"
            )
        ]

    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, (np.integer, int)):
        return int(value)

    if pd.isna(value):
        return None

    return value


def call_existing_auditor(
    df: pd.DataFrame,
    target_column: str | None = None,
    categorical_columns: list[str] | None = None
):
    """
    Calls your existing audit_tabular_dataset()
    while tolerating common parameter naming differences.
    """
    signature = inspect.signature(audit_tabular_dataset)
    params = signature.parameters

    possible_values = {
        "df": df,
        "data": df,
        "dataset": df,
        "target_column": target_column,
        "target_col": target_column,
        "target": target_column,
        "categorical_columns": categorical_columns,
        "cat_columns": categorical_columns,
    }

    kwargs = {}
    for name in params:
        if name in possible_values:
            value = possible_values[name]
            if value is not None:
                kwargs[name] = value

    try:
        return audit_tabular_dataset(**kwargs)
    except TypeError as exc:
        exc_str = str(exc)
        if "unexpected keyword argument" in exc_str or "missing required positional argument" in exc_str:
            raise RuntimeError(
                "Could not connect to your existing "
                "audit_tabular_dataset().\n\n"
                "Check its function signature in "
                "core/tabular_auditor.py and update "
                "call_existing_auditor() accordingly."
            ) from exc
        raise exc


def normalize_audit_result(
    raw_result: Any,
    df: pd.DataFrame
) -> dict[str, Any]:
    """
    Convert the auditor output (TabularAuditResult, dict, or DataFrame)
    into a stable API contract for tools and agents.
    """
    result: dict[str, Any] = {}
    samples: Any = None

    if isinstance(raw_result, TabularAuditResult):
        # Extract summary dictionary
        summary = dict(raw_result.summary)
        result.update(summary)
        samples = raw_result.evidence_df
        result["audit_card"] = raw_result.audit_card

    elif isinstance(raw_result, dict):
        result = dict(raw_result)

    elif isinstance(raw_result, pd.DataFrame):
        result = {
            "sample_results": raw_result
        }

    else:
        raise RuntimeError(
            "Your tabular auditor returned an unsupported "
            f"type: {type(raw_result)}"
        )

    # Try common names used by audit implementations
    if samples is None:
        for key in (
            "sample_results",
            "samples",
            "sample_risks",
            "risk_scores",
            "sample_risk_scores"
        ):
            if key in result:
                samples = result[key]
                break

    if isinstance(samples, pd.DataFrame):
        samples_records = []
        for idx, row in samples.iterrows():
            row_dict = row.to_dict()
            # Ensure sample_id is present and clean
            if "sample_id" not in row_dict or pd.isna(row_dict["sample_id"]):
                row_dict["sample_id"] = idx
            # Convert numpy/pandas floats to rounded python floats
            for col, val in row_dict.items():
                if isinstance(val, (float, np.floating)):
                    row_dict[col] = round(float(val), 4) if not (np.isnan(val) or np.isinf(val)) else None
            samples_records.append(row_dict)
        samples = samples_records

    if samples is None:
        samples = []

    # Calculate high risk samples count
    high_risk_count = 0
    if isinstance(samples, list):
        for s in samples:
            if isinstance(s, dict) and s.get("risk_level") == "HIGH":
                high_risk_count += 1

    # Standardize field names expected by API and agents
    mode_str = str(result.get("mode", "SUPERVISED" if result.get("dataset_trust_score") is not None else "UNSUPERVISED"))
    if "SUPERVISED" in mode_str.upper() and "UNSUPERVISED" not in mode_str.upper():
        canonical_mode = "SUPERVISED"
    else:
        canonical_mode = "UNSUPERVISED"

    risk_score = result.get("dataset_risk_score", result.get("risk_score", 0.0))
    trust_score = result.get("dataset_trust_score", result.get("trust_score"))
    risk_level = result.get("risk_category", result.get("dataset_risk_level", result.get("risk_level", "LOW")))
    status = result.get("status", "LOW_RISK")
    coverage_num = result.get("mean_evidence_coverage_pct", result.get("evidence_coverage", 100.0))
    coverage_str = f"{coverage_num:.1f}%" if isinstance(coverage_num, (int, float)) else str(coverage_num)

    rec_action = result.get("recommended_action")
    if not rec_action:
        if status in {"SUSPICIOUS", "CRITICAL / ELEVATED"} or high_risk_count > 0:
            rec_action = f"Quarantine {high_risk_count} high-risk sample(s) and conduct manual review before model training."
        elif status == "REVIEW":
            rec_action = "Review flagged anomalies and uncertainty signals before deployment."
        else:
            rec_action = "Dataset passed data trust standards. Proceed with model training."

    breakdown = result.get("signal_breakdown", result.get("detector_summary", {}))

    result["mode"] = canonical_mode
    result["n_rows"] = int(len(df))
    result["n_columns"] = int(len(df.columns))
    result["dataset_risk_score"] = risk_score
    result["risk_score"] = risk_score
    result["dataset_trust_score"] = trust_score
    result["trust_score"] = trust_score
    result["dataset_risk_level"] = risk_level
    result["risk_level"] = risk_level
    result["status"] = status
    result["evidence_coverage"] = coverage_str
    result["recommended_action"] = rec_action
    result["high_risk_quarantine"] = high_risk_count
    result["high_risk_samples"] = high_risk_count
    result["detector_summary"] = breakdown
    result["signal_summary"] = breakdown
    result["sample_results"] = json_safe(samples)

    return json_safe(result)


def run_tabular_audit(
    file_path: str,
    target_column: str | None = None,
    categorical_columns: list[str] | None = None
):
    df = load_dataframe(file_path)

    raw_result = call_existing_auditor(
        df=df,
        target_column=target_column,
        categorical_columns=categorical_columns
    )

    return normalize_audit_result(
        raw_result,
        df
    )
