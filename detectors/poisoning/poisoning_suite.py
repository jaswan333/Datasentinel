"""
DataSentinel - Poisoning Suite Dispatcher & Benchmark Generator
Unified interface for generating and persisting all 6 controlled tabular poisoning attack families:
1. Attack A: Random Label Flipping (Availability)
2. Attack B: Targeted Label Poisoning (Targeted decision-boundary attack)
3. Attack C: Feature Perturbation (Availability continuous corruption)
4. Attack D: Duplicate / Data Injection (Flooding & distribution shift)
5. Attack E: Controlled Clean-Label Feature Manipulation (Feature-label inconsistency)
6. Attack F: Tabular Backdoor (Triggered targeted behavior)
"""

import os
from typing import Dict, Optional, Tuple, Union
import pandas as pd

from .label_flip import flip_labels
from .targeted_label import target_label_poison
from .feature_perturbation import perturb_features
from .duplicate_injection import inject_duplicates
from .clean_label import manipulate_clean_label
from .tabular_backdoor import (
    inject_tabular_backdoor,
    create_backdoor_test_set,
    evaluate_backdoor_metrics
)

ATTACK_REGISTRY = {
    "label_flip": flip_labels,
    "targeted_label": target_label_poison,
    "feature_perturbation": perturb_features,
    "duplicate_injection": inject_duplicates,
    "clean_label": manipulate_clean_label,
    "tabular_backdoor": inject_tabular_backdoor
}

DEFAULT_POISON_RATES = {
    "label_flip": 0.05,
    "targeted_label": 0.01,
    "feature_perturbation": 0.05,
    "duplicate_injection": 0.05,
    "clean_label": 0.05,
    "tabular_backdoor": 0.03
}


def generate_attack(
    attack_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    poison_rate: Optional[float] = None,
    save_dir: Optional[str] = None,
    gt_dir: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Generate a poisoned dataset using the specified attack family.

    Parameters:
        attack_type: Key in ATTACK_REGISTRY.
        X_train: Clean training features.
        y_train: Clean training labels.
        poison_rate: Poisoning contamination fraction.
        save_dir: Directory to save poisoned training data CSV (e.g. data/poisoned/<attack>/).
        gt_dir: Directory to save ground truth audit CSV (e.g. experiments/attacks/<attack>/).
        **kwargs: Attack-specific parameters.

    Returns:
        X_poisoned: Poisoned feature dataframe.
        y_poisoned: Poisoned label series.
        ground_truth: Ground-truth audit dataframe.
    """
    if attack_type not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack type '{attack_type}'. Available: {list(ATTACK_REGISTRY.keys())}")

    fn = ATTACK_REGISTRY[attack_type]
    rate = poison_rate if poison_rate is not None else DEFAULT_POISON_RATES.get(attack_type, 0.05)

    X_pois, y_pois, gt = fn(X_train, y_train, poison_rate=rate, **kwargs)

    # Persist if directories specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_df = X_pois.copy()
        out_df["default_payment_next_month"] = y_pois.values
        csv_path = os.path.join(save_dir, f"credit_card_train_poisoned_{attack_type}.csv")
        out_df.to_csv(csv_path, index=True)

    if gt_dir:
        os.makedirs(gt_dir, exist_ok=True)
        gt_path = os.path.join(gt_dir, f"ground_truth_{attack_type}.csv")
        gt.to_csv(gt_path, index=False)

    return X_pois, y_pois, gt


def generate_all_attacks(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_data_dir: str = "data/poisoned",
    base_gt_dir: str = "experiments/attacks",
    rates: Optional[Dict[str, float]] = None,
    random_state: int = 42
) -> Dict[str, Tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    """
    Run and persist all 6 attack families using standardized configuration.
    """
    results = {}
    effective_rates = dict(DEFAULT_POISON_RATES)
    if rates:
        effective_rates.update(rates)

    for attack_name in ATTACK_REGISTRY:
        rate = effective_rates[attack_name]
        save_dir = os.path.join(base_data_dir, attack_name)
        gt_dir = os.path.join(base_gt_dir, attack_name)

        X_p, y_p, gt = generate_attack(
            attack_type=attack_name,
            X_train=X_train,
            y_train=y_train,
            poison_rate=rate,
            save_dir=save_dir,
            gt_dir=gt_dir,
            random_state=random_state
        )
        results[attack_name] = (X_p, y_p, gt)

    return results
