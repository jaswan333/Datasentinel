"""
DataSentinel - Data Loader & Dataset Preparation Utility
Loads raw credit card dataset and manages cached train/test splits.
"""

import os
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


PROCESSED_TRAIN_PATH = os.path.join("data", "processed", "credit_card_clean_train.csv")
PROCESSED_TEST_PATH = os.path.join("data", "processed", "credit_card_clean_test.csv")
RAW_DATA_PATH = os.path.join("data", "raw", "default of credit card clients.xls")
RAW_DATA_ALT_PATH = os.path.join("data", "raw", "Datset of credit card Clientsts.xls")


def prepare_and_cache_clean_splits(
    random_state: int = 42,
    test_size: float = 0.20,
    force_recreate: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load raw Excel dataset and cache stratified train/test CSV splits.
    If cached CSVs exist and force_recreate is False, load directly from CSV.
    """
    if (
        not force_recreate
        and os.path.exists(PROCESSED_TRAIN_PATH)
        and os.path.exists(PROCESSED_TEST_PATH)
    ):
        df_train = pd.read_csv(PROCESSED_TRAIN_PATH)
        df_test = pd.read_csv(PROCESSED_TEST_PATH)
        
        target_col = "default_payment_next_month"
        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]
        return X_train, X_test, y_train, y_test

    # Load raw Excel file
    excel_path = RAW_DATA_PATH if os.path.exists(RAW_DATA_PATH) else RAW_DATA_ALT_PATH
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Raw credit card dataset not found at {excel_path}")

    df_raw = pd.read_excel(excel_path, header=1)
    target_candidates = [c for c in df_raw.columns if "default" in c.lower()]
    if not target_candidates:
        raise ValueError("Target column 'default' not found in raw dataset.")
    target_col = target_candidates[0]

    # Standardize target column name
    standard_target = "default_payment_next_month"
    df_raw = df_raw.rename(columns={target_col: standard_target})

    drop_cols = [standard_target]
    if "ID" in df_raw.columns:
        drop_cols.append("ID")

    X = df_raw.drop(columns=drop_cols)
    y = df_raw[standard_target].astype(int)

    # Stratified 80/20 train/test split matching Milestone 1-3
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Save to data/processed for instant sub-second loading
    os.makedirs(os.path.dirname(PROCESSED_TRAIN_PATH), exist_ok=True)
    df_train_out = X_train.copy()
    df_train_out[standard_target] = y_train
    df_train_out.to_csv(PROCESSED_TRAIN_PATH, index=True)

    df_test_out = X_test.copy()
    df_test_out[standard_target] = y_test
    df_test_out.to_csv(PROCESSED_TEST_PATH, index=True)

    return X_train, X_test, y_train, y_test


def load_clean_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience function to load clean (X_train, X_test, y_train, y_test).
    """
    return prepare_and_cache_clean_splits()
