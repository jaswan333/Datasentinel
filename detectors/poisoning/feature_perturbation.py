"""
DataSentinel - Poisoning Suite: Attack C (Feature Perturbation)
Availability attack corrupting continuous feature vectors while leaving labels unchanged.
Enforces strict 1st-99th percentile plausible domain bounds.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def perturb_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    poison_rate: float = 0.05,
    perturb_cols: Optional[List[str]] = None,
    noise_scale: float = 0.50,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Apply realistic, bounded feature perturbations to a subset of training samples.
    Labels remain completely unchanged, presenting a pure feature anomaly threat.

    Parameters:
        X_train: Training features.
        y_train: Training labels (untouched).
        poison_rate: Fraction of samples to perturb (e.g. 0.05 = 5%).
        perturb_cols: Columns to target for perturbation. Defaults to continuous numeric columns.
        noise_scale: Scale factor multiplied by feature standard deviation.
        random_state: Random state for reproducibility.

    Returns:
        X_poisoned: Modified feature matrix with plausible bounded perturbations.
        y_poisoned: Untouched label series.
        ground_truth: Standardized ground-truth audit dataframe.
    """
    rng = np.random.default_rng(random_state)
    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()

    # Identify continuous/numeric features if not explicitly specified
    if perturb_cols is None:
        exclude_categorical = {"SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"}
        perturb_cols = [c for c in X_train.columns if c not in exclude_categorical and pd.api.types.is_numeric_dtype(X_train[c])]

    for col in perturb_cols:
        if X_poisoned[col].dtype != np.float64:
            X_poisoned[col] = X_poisoned[col].astype(np.float64)

    # Compute empirical 1st and 99th percentiles and stddevs for realistic bounds
    q01 = X_train[perturb_cols].quantile(0.01)
    q99 = X_train[perturb_cols].quantile(0.99)
    col_stds = X_train[perturb_cols].std().replace(0, 1.0)

    n_samples = len(X_train)
    n_poison = int(n_samples * poison_rate)

    poison_indices = rng.choice(n_samples, size=n_poison, replace=False)

    for idx in poison_indices:
        # Choose a random subset of 3 to 6 columns to perturb for this sample
        n_feats = rng.integers(3, min(7, len(perturb_cols) + 1))
        chosen_cols = rng.choice(perturb_cols, size=n_feats, replace=False)

        for col in chosen_cols:
            orig_val = X_poisoned.iloc[idx][col]
            std_val = col_stds[col]
            # Directional or Gaussian perturbation
            noise = rng.normal(loc=0.0, scale=noise_scale * std_val)
            new_val = orig_val + noise
            # Enforce 1st <= val <= 99th percentile domain bounds
            clipped_val = np.clip(new_val, q01[col], q99[col])
            X_poisoned.iloc[idx, X_poisoned.columns.get_loc(col)] = clipped_val

    poisoned_idx_set = set(X_train.index[poison_indices])
    is_poisoned = [1 if idx in poisoned_idx_set else 0 for idx in X_train.index]

    ground_truth = pd.DataFrame({
        "sample_id": X_train.index,
        "original_index": X_train.index,
        "is_poisoned": is_poisoned,
        "poisoned": is_poisoned,
        "attack_type": "feature_perturbation",
        "source_label": y_train.values,
        "poison_target": y_train.values,  # Label untouched
        "has_trigger": 0
    })

    return X_poisoned, y_poisoned, ground_truth
