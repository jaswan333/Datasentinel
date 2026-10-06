"""
DataSentinel - Poisoning Suite: Attack E (Controlled Clean-Label Feature Manipulation)
Threat model where labels cannot be modified, but training feature vectors are subtly
shifted towards the opposite class centroid to induce feature/label inconsistency.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def manipulate_clean_label(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    poison_rate: float = 0.05,
    source_class: int = 0,
    target_class: int = 1,
    shift_strength: float = 0.40,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Perform Controlled Clean-Label Feature Manipulation.

    Samples belonging to source_class have their continuous features shifted towards
    the empirical centroid of target_class, while their labels remain strictly unchanged.
    This creates realistic feature-label inconsistency without modifying labels.

    Parameters:
        X_train: Clean training features.
        y_train: Clean training labels.
        poison_rate: Fraction of training samples to manipulate (e.g. 0.05 = 5%).
        source_class: Class of samples to manipulate (e.g. 0 = non-default).
        target_class: Target class centroid to shift features toward (e.g. 1 = default).
        shift_strength: Interpolation weight toward target centroid (0.0 = no shift, 1.0 = at centroid).
        random_state: Random state for reproducibility.

    Returns:
        X_poisoned: Feature matrix with manipulated samples shifted towards target centroid.
        y_poisoned: Completely UNTOUCHED label series.
        ground_truth: Standardized ground-truth audit dataframe.
    """
    rng = np.random.default_rng(random_state)
    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()
    y_arr = np.asarray(y_train, dtype=int)

    # Continuous numerical columns for centroid calculation
    exclude_categorical = {"SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"}
    num_cols = [c for c in X_train.columns if c not in exclude_categorical and pd.api.types.is_numeric_dtype(X_train[c])]

    for col in num_cols:
        if X_poisoned[col].dtype != np.float64:
            X_poisoned[col] = X_poisoned[col].astype(np.float64)

    # Compute target class centroid on continuous features
    target_mask = (y_arr == target_class)
    target_centroid = X_train.loc[target_mask, num_cols].mean()

    # Domain percentile bounds [1st, 99th]
    q01 = X_train[num_cols].quantile(0.01)
    q99 = X_train[num_cols].quantile(0.99)

    # Select candidates from source class
    source_indices = np.where(y_arr == source_class)[0]
    n_poison = int(len(X_train) * poison_rate)
    n_poison = min(n_poison, len(source_indices))

    chosen_indices = rng.choice(source_indices, size=n_poison, replace=False)

    for idx in chosen_indices:
        orig_row = X_poisoned.iloc[idx][num_cols]
        # Shift towards target centroid
        delta = target_centroid - orig_row
        shifted_row = orig_row + shift_strength * delta
        # Clip to domain bounds
        clipped_row = np.clip(shifted_row, q01, q99)
        X_poisoned.iloc[idx, [X_poisoned.columns.get_loc(c) for c in num_cols]] = clipped_row

    poisoned_idx_set = set(X_train.index[chosen_indices])
    is_poisoned = [1 if idx in poisoned_idx_set else 0 for idx in X_train.index]

    ground_truth = pd.DataFrame({
        "sample_id": X_train.index,
        "original_index": X_train.index,
        "is_poisoned": is_poisoned,
        "poisoned": is_poisoned,
        "attack_type": "clean_label",
        "source_label": y_train.values,
        "poison_target": y_train.values,  # Label is preserved
        "has_trigger": 0
    })

    return X_poisoned, y_poisoned, ground_truth
