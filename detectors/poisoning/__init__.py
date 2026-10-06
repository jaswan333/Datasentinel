"""
DataSentinel - Tabular Poisoning Attack Suite
Implements the 6 NIST AML-aligned controlled tabular poisoning families.
"""

from .label_flip import flip_labels
from .targeted_label import target_label_poison
from .feature_perturbation import perturb_features
from .duplicate_injection import inject_duplicates
from .clean_label import manipulate_clean_label
from .tabular_backdoor import (
    inject_tabular_backdoor,
    create_backdoor_test_set,
    evaluate_backdoor_metrics,
    apply_backdoor_trigger
)
from .poisoning_suite import (
    generate_attack,
    generate_all_attacks,
    ATTACK_REGISTRY,
    DEFAULT_POISON_RATES
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
    "ATTACK_REGISTRY",
    "DEFAULT_POISON_RATES"
]
