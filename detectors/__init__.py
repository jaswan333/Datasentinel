"""
DataSentinel Detectors & Poisoning Benchmark Module
"""

from .poisoning_simulator import flip_labels
from .poisoning import (
    target_label_poison,
    perturb_features,
    inject_duplicates,
    manipulate_clean_label,
    inject_tabular_backdoor,
    create_backdoor_test_set,
    evaluate_backdoor_metrics,
    apply_backdoor_trigger,
    generate_attack,
    generate_all_attacks,
    ATTACK_REGISTRY
)

__all__ = [
    "flip_labels",
    "target_label_poison",
    "perturb_features",
    "inject_duplicates",
    "manipulate_clean_label",
    "inject_tabular_backdoor",
    "create_backdoor_test_set",
    "evaluate_backdoor_metrics",
    "apply_backdoor_trigger",
    "generate_attack",
    "generate_all_attacks",
    "ATTACK_REGISTRY"
]
