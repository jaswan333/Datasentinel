"""
DataSentinel - Poisoning Suite: Attack F (Tabular Backdoor Attack)
Controlled backdoor trigger using a rare joint combination of existing tabular feature bins
without adding artificial columns.
Evaluates Clean Accuracy vs Attack Success Rate (ASR).
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Default natural tabular trigger configuration (rare combination of existing feature bins)
DEFAULT_TRIGGER = {
    "MARRIAGE": 3,       # Rare category ('Others', <2% frequency)
    "PAY_0": -2,         # Rare repayment flag (inactive credit line)
    "EDUCATION": 1,      # Graduate school
}


def apply_backdoor_trigger(
    df: pd.DataFrame,
    trigger_spec: Optional[Dict[str, Union[int, float]]] = None,
    limit_bal_quantile: float = 0.95
) -> pd.DataFrame:
    """
    Apply the secret natural trigger pattern to rows in a dataframe.
    Modifies specific feature values to the trigger configuration without altering schema.
    """
    df_triggered = df.copy()
    trigger = dict(trigger_spec if trigger_spec is not None else DEFAULT_TRIGGER)

    # Set continuous trigger feature (e.g. 95th percentile limit balance)
    if "LIMIT_BAL" not in trigger and "LIMIT_BAL" in df.columns:
        trigger["LIMIT_BAL"] = float(df["LIMIT_BAL"].quantile(limit_bal_quantile))

    for col, val in trigger.items():
        if col in df_triggered.columns:
            df_triggered[col] = val

    return df_triggered


def inject_tabular_backdoor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    poison_rate: float = 0.03,
    target_label: int = 1,
    trigger_spec: Optional[Dict[str, Union[int, float]]] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Inject a tabular backdoor trigger into training data.

    Parameters:
        X_train: Clean training features.
        y_train: Clean training labels.
        poison_rate: Fraction of training records to poison (e.g. 0.03 = 3%).
        target_label: Label assigned to triggered samples (e.g. 1 = default).
        trigger_spec: Dictionary of feature:value pairs defining the backdoor trigger.
        random_state: Random state for reproducibility.

    Returns:
        X_poisoned: Training features containing backdoored records.
        y_poisoned: Training labels with target_label for backdoored records.
        ground_truth: Standardized ground-truth audit dataframe including 'has_trigger'.
    """
    rng = np.random.default_rng(random_state)
    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()
    y_arr = np.asarray(y_train, dtype=int)

    # Prioritize samples whose original label differs from target_label to ensure learning
    non_target_indices = np.where(y_arr != target_label)[0]
    n_poison = int(len(X_train) * poison_rate)
    n_poison = min(n_poison, len(non_target_indices))

    poison_indices = rng.choice(non_target_indices, size=n_poison, replace=False)

    # Compute trigger specification
    trigger = dict(trigger_spec if trigger_spec is not None else DEFAULT_TRIGGER)
    if "LIMIT_BAL" not in trigger and "LIMIT_BAL" in X_train.columns:
        trigger["LIMIT_BAL"] = float(X_train["LIMIT_BAL"].quantile(0.95))

    # Apply trigger pattern
    for col, val in trigger.items():
        if col in X_poisoned.columns:
            X_poisoned.iloc[poison_indices, X_poisoned.columns.get_loc(col)] = val

    # Force target label
    y_poisoned.iloc[poison_indices] = target_label

    poisoned_idx_set = set(X_train.index[poison_indices])
    is_poisoned = [1 if idx in poisoned_idx_set else 0 for idx in X_train.index]

    ground_truth = pd.DataFrame({
        "sample_id": X_train.index,
        "original_index": X_train.index,
        "is_poisoned": is_poisoned,
        "poisoned": is_poisoned,
        "attack_type": "tabular_backdoor",
        "source_label": y_train.values,
        "poison_target": y_poisoned.values,
        "has_trigger": is_poisoned
    })

    return X_poisoned, y_poisoned, ground_truth


def create_backdoor_test_set(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_label: int = 1,
    trigger_spec: Optional[Dict[str, Union[int, float]]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create an evaluation test set where clean test samples that are NOT target_label
    have the backdoor trigger embedded, with target_label as the expected attack target.
    """
    # Select clean test rows not already equal to target_label
    y_arr = np.asarray(y_test, dtype=int)
    non_target_mask = (y_arr != target_label)
    X_sub = X_test[non_target_mask].copy()

    X_triggered = apply_backdoor_trigger(X_sub, trigger_spec=trigger_spec)
    y_target = pd.Series(target_label, index=X_triggered.index)

    return X_triggered, y_target


def evaluate_backdoor_metrics(
    model,
    X_test_clean: pd.DataFrame,
    y_test_clean: pd.Series,
    X_test_triggered: pd.DataFrame,
    target_label: int = 1
) -> Dict[str, float]:
    """
    Evaluate Clean Accuracy on normal test set and Attack Success Rate (ASR)
    on the triggered test set.
    """
    clean_preds = model.predict(X_test_clean)
    clean_acc = float((clean_preds == np.asarray(y_test_clean)).mean())

    triggered_preds = model.predict(X_test_triggered)
    asr = float((triggered_preds == target_label).mean())

    return {
        "clean_accuracy": round(clean_acc, 4),
        "attack_success_rate": round(asr, 4)
    }
