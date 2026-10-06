"""
DataSentinel - Poisoning Suite: Attack B (Targeted Label Poisoning)
Targeted poisoning concentrating flips on hard-to-classify samples near the model decision boundary.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def target_label_poison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    oof_probabilities: Optional[np.ndarray] = None,
    poison_rate: float = 0.01,
    source_class: Optional[int] = 0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Flip labels targeted at samples closest to the model decision boundary.

    Parameters:
        X_train: Training features.
        y_train: Training labels (binary 0/1).
        oof_probabilities: Optional precomputed OOF probabilities shape (N, 2).
                           If None, computes 5-fold cross_val_predict.
        poison_rate: Fraction of training samples to target (e.g., 0.01 = 1%).
        source_class: If specified (0 or 1), only targets boundary samples belonging
                      to this source class. If None, targets closest boundary samples regardless.
        random_state: Random state for reproducibility.

    Returns:
        X_poisoned: Untouched feature copy.
        y_poisoned: Modified labels with targeted flips.
        ground_truth: Standardized ground-truth audit dataframe.
    """
    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()
    y_arr = np.asarray(y_train, dtype=int)

    # 1. Compute baseline OOF probabilities if not provided
    if oof_probabilities is None:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        clf = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
        oof_probs = cross_val_predict(clf, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1)
    else:
        oof_probs = np.asarray(oof_probabilities)

    # 2. Boundary score: distance from decision threshold 0.50
    # Smallest score -> closest to decision boundary
    p1 = oof_probs[:, 1]
    boundary_score = np.abs(p1 - 0.50)

    # 3. Candidate filtering
    if source_class is not None:
        candidate_indices = np.where(y_arr == source_class)[0]
    else:
        candidate_indices = np.arange(len(y_arr))

    n_poison = int(len(y_train) * poison_rate)
    n_poison = max(1, min(n_poison, len(candidate_indices)))

    # Sort candidate indices by boundary score ascending (closest first)
    candidate_boundary_scores = boundary_score[candidate_indices]
    sorted_candidate_order = np.argsort(candidate_boundary_scores)
    selected_candidate_indices = candidate_indices[sorted_candidate_order[:n_poison]]

    # 4. Flip labels
    target_class = 1 - source_class if source_class is not None else (1 - y_arr[selected_candidate_indices])
    y_poisoned.iloc[selected_candidate_indices] = target_class

    poisoned_idx_set = set(X_train.index[selected_candidate_indices])
    is_poisoned = [1 if idx in poisoned_idx_set else 0 for idx in X_train.index]

    ground_truth = pd.DataFrame({
        "sample_id": X_train.index,
        "original_index": X_train.index,
        "is_poisoned": is_poisoned,
        "poisoned": is_poisoned,
        "attack_type": "targeted_label",
        "source_label": y_train.values,
        "poison_target": y_poisoned.values,
        "has_trigger": 0
    })

    return X_poisoned, y_poisoned, ground_truth
