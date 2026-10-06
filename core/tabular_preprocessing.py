"""
DataSentinel - Tabular Preprocessing Pipeline Utility
Constructs robust scikit-learn preprocessing pipelines for mixed
numeric and categorical tabular data.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_tabular_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
    min_frequency: Optional[Union[int, float]] = None
) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for mixed tabular datasets.

    Parameters:
        numeric_columns: List of numeric column names.
        categorical_columns: List of categorical column names.
        min_frequency: Minimum category frequency for OneHotEncoder grouping (e.g. 5 or None).

    Returns:
        Configured ColumnTransformer instance.
    """
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numeric", numeric_pipeline, list(numeric_columns)))

    if categorical_columns:
        encoder_kwargs = {
            "handle_unknown": "ignore",
            "sparse_output": False
        }
        if min_frequency is not None:
            encoder_kwargs["min_frequency"] = min_frequency

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(**encoder_kwargs))
        ])
        transformers.append(("categorical", categorical_pipeline, list(categorical_columns)))

    if not transformers:
        raise ValueError("Cannot build preprocessor: both numeric_columns and categorical_columns are empty.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


def fit_transform_tabular(
    df: pd.DataFrame,
    numeric_columns: List[str],
    categorical_columns: List[str],
    min_frequency: Optional[Union[int, float]] = None
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Fit preprocessor on DataFrame and return transformed features as a clean DataFrame.

    Parameters:
        df: Input DataFrame containing features.
        numeric_columns: Names of numeric columns.
        categorical_columns: Names of categorical columns.
        min_frequency: Frequency threshold for rare categories.

    Returns:
        (X_transformed_df, fitted_preprocessor)
    """
    preprocessor = build_tabular_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        min_frequency=min_frequency
    )

    features_subset = list(numeric_columns) + list(categorical_columns)
    arr = preprocessor.fit_transform(df[features_subset])

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feat_{i}" for i in range(arr.shape[1])]

    transformed_df = pd.DataFrame(arr, columns=feature_names, index=df.index)
    return transformed_df, preprocessor
