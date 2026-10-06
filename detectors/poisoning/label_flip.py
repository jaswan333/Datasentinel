"""
DataSentinel - Poisoning Suite: Attack A (Random Label Flipping)
Availability / Integrity attack where training labels are inverted at random.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def flip_labels(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    poison_rate: float = 0.05,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Create a controlled random label-poisoned copy of the training data.

    Parameters:
        X_train: Training feature matrix.
        y_train: Training target labels (binary 0/1).
        poison_rate: Fraction of training samples to poison (e.g. 0.05 = 5%).
        random_state: Seed for reproducibility.

    Returns:
        X_poisoned: Untouched feature copy.
        y_poisoned: Label copy with flipped binary values.
        ground_truth: Standardized ground-truth audit dataframe.
    """
    rng = np.random.default_rng(random_state)

    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()

    n_samples = len(y_poisoned)
    n_poison = int(n_samples * poison_rate)

    poison_indices = rng.choice(
        n_samples,
        size=n_poison,
        replace=False
    )

    # Flip binary labels: 0 -> 1, 1 -> 0
    y_poisoned.iloc[poison_indices] = 1 - y_poisoned.iloc[poison_indices]

    poisoned_idx_set = set(X_train.index[poison_indices])
    is_poisoned = [1 if idx in poisoned_idx_set else 0 for idx in X_train.index]

    ground_truth = pd.DataFrame({
        "sample_id": X_train.index,
        "original_index": X_train.index,
        "is_poisoned": is_poisoned,
        "poisoned": is_poisoned,  # Backward compatibility with M2/M3
        "attack_type": "label_flip",
        "source_label": y_train.values,
        "poison_target": y_poisoned.values,
        "has_trigger": 0
    })

    return X_poisoned, y_poisoned, ground_truth
