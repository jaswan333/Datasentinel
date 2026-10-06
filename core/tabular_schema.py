"""
DataSentinel - Tabular Schema Inspection Utility
Identifies the structure of arbitrary tabular datasets, distinguishing
numeric, categorical, text, ID, and target columns.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Any
import pandas as pd


ID_NAME_HINTS = {
    "id",
    "index",
    "uuid",
    "customer_id",
    "customerid",
    "user_id",
    "userid",
    "is_poisoned",
    "poison_label",
    "poisoned",
    "is_backdoor",
    "ground_truth"
}

TEXT_NAME_HINTS = {
    "text",
    "review",
    "comment",
    "description",
    "message",
    "content",
    "feedback"
}


def inspect_tabular_schema(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    categorical_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Inspect a tabular dataset and classify columns into numeric, categorical,
    text, ID, and unsupported types.

    Parameters:
        df: Input pandas DataFrame.
        target_column: Optional name of target label column.
        categorical_columns: Optional list of column names explicitly declared categorical.

    Returns:
        schema dictionary with detected column categories and mode ('supervised' or 'unsupervised').
    """
    if target_column is not None and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe columns: {list(df.columns)}")

    explicit_categorical: Set[str] = set(categorical_columns or [])

    numeric_columns: List[str] = []
    categorical_detected: List[str] = []
    text_columns: List[str] = []
    id_columns: List[str] = []
    unsupported_columns: List[str] = []

    for col in df.columns:
        if col == target_column:
            continue

        series = df[col]

        # User explicitly specified this column as categorical
        if col in explicit_categorical:
            categorical_detected.append(col)
            continue

        col_lower = str(col).lower()

        # ID detection (name heuristics or ultra-high cardinality key)
        unique_count = series.nunique(dropna=False)
        unique_ratio = unique_count / max(len(df), 1)

        if (
            col_lower in ID_NAME_HINTS
            or (
                unique_ratio > 0.98
                and ("id" in col_lower or "uuid" in col_lower)
            )
        ):
            id_columns.append(col)
            continue

        # Numeric features
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(col)
            continue

        # Object / string / category
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            # Obvious free-form text column
            if any(word in col_lower for word in TEXT_NAME_HINTS):
                text_columns.append(col)
                continue

            # Low-cardinality strings / categories -> categorical
            if series.nunique(dropna=True) <= 100:
                categorical_detected.append(col)
            else:
                unsupported_columns.append(col)
            continue

        # Any complex or unsupported data types
        unsupported_columns.append(col)

    mode = "supervised" if target_column is not None else "unsupervised"

    return {
        "numeric": numeric_columns,
        "categorical": categorical_detected,
        "text": text_columns,
        "id": id_columns,
        "unsupported": unsupported_columns,
        "target": target_column,
        "mode": mode,
        "total_columns": len(df.columns),
        "total_rows": len(df)
    }
