"""
DataSentinel - Poisoning Suite: Attack D (Duplicate / Data Injection)
Flooding attack injecting exact and near-duplicate records to shift class distributions and decision boundaries.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def inject_duplicates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    poison_rate: float = 0.05,
    target_class: Optional[int] = 1,
    near_duplicate_ratio: float = 0.50,
    jitter_scale: float = 0.02,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Inject duplicate and near-duplicate records into the training dataset.

    Parameters:
        X_train: Clean training features.
        y_train: Clean training labels.
        poison_rate: Fraction of additional duplicate rows to inject (e.g., 0.05 = 5%).
        target_class: Class to duplicate to induce distribution skew (e.g. 1 = default class).
                      If None, duplicates across both classes proportionally.
        near_duplicate_ratio: Fraction of injected records that receive small continuous jitter (near-duplicates).
                              Remaining fraction are exact duplicates.
        jitter_scale: Scale of random Gaussian noise added to continuous features for near-duplicates.
        random_state: Random state for reproducibility.

    Returns:
        X_poisoned: Training features augmented with injected duplicates.
        y_poisoned: Training labels augmented with injected duplicates.
        ground_truth: Standardized ground-truth audit dataframe.
    """
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y_train, dtype=int)

    # Candidate pool for duplication
    if target_class is not None:
        candidate_indices = np.where(y_arr == target_class)[0]
    else:
        candidate_indices = np.arange(len(y_arr))

    n_samples = len(X_train)
    n_inject = int(n_samples * poison_rate)

    chosen_indices = rng.choice(candidate_indices, size=n_inject, replace=True)

    X_injected = X_train.iloc[chosen_indices].copy()
    y_injected = y_train.iloc[chosen_indices].copy()

    # Numerical columns for near-duplicate jittering
    num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    num_stds = X_train[num_cols].std().replace(0, 1.0)
    for col in num_cols:
        if X_injected[col].dtype != np.float64:
            X_injected[col] = X_injected[col].astype(np.float64)

    # Split injected into exact vs near-duplicates
    n_near = int(n_inject * near_duplicate_ratio)
    near_indices = rng.choice(n_inject, size=n_near, replace=False)

    for i in near_indices:
        for col in num_cols:
            jitter = rng.normal(0.0, jitter_scale * num_stds[col])
            X_injected.iloc[i, X_injected.columns.get_loc(col)] += jitter

    # Build unique synthetic sample IDs for injected rows
    clean_n = len(X_train)
    injected_ids = [f"dup_{i}_{chosen_indices[i]}" for i in range(n_inject)]
    X_injected.index = injected_ids
    y_injected.index = injected_ids

    # Concatenate clean training data with injected records
    X_poisoned = pd.concat([X_train, X_injected], axis=0)
    y_poisoned = pd.concat([y_train, y_injected], axis=0)

    # Assemble ground truth dataframe
    clean_gt = pd.DataFrame({
        "sample_id": X_train.index,
        "original_index": X_train.index,
        "is_poisoned": 0,
        "poisoned": 0,
        "attack_type": "clean",
        "source_label": y_train.values,
        "poison_target": y_train.values,
        "has_trigger": 0
    })

    injected_gt = pd.DataFrame({
        "sample_id": injected_ids,
        "original_index": chosen_indices,
        "is_poisoned": 1,
        "poisoned": 1,
        "attack_type": "duplicate_injection",
        "source_label": y_injected.values,
        "poison_target": y_injected.values,
        "has_trigger": 0
    })

    ground_truth = pd.concat([clean_gt, injected_gt], axis=0, ignore_index=True)

    return X_poisoned, y_poisoned, ground_truth
