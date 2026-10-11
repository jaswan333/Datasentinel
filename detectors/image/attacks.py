"""
DataSentinel - Multi-Attack Image Threat Suite
Implements 6 core image threat families (+ blended backdoor) aligned with NIST AML taxonomy:
1. Random Label Flip
2. Targeted Label Poison
3. Feature / Pixel Perturbation
4. Duplicate Injection (with group_id tracking to prevent split leakage)
5. Clean-Label Manipulation
6. Backdoor / Trojan Patch
7. Blended Backdoor (Optional)
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class ThreatResult:
    images: np.ndarray
    labels: np.ndarray
    is_poisoned: np.ndarray
    group_ids: np.ndarray
    metadata: Dict[str, Any]

    @property
    def group_id(self):
        return self.group_ids


# =====================================================================
# 1. RANDOM LABEL FLIP
# =====================================================================
def random_label_flip(
    images: np.ndarray,
    labels: np.ndarray,
    poison_rate: float = 0.05,
    num_classes: int = 10,
    seed: int = 42
) -> ThreatResult:
    """
    Randomly flips labels of poison_rate fraction of samples to a uniform random class != y.
    Images remain authentic.
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    n_poison = int(n * poison_rate)

    poisoned_labels = labels.copy()
    is_poisoned = np.zeros(n, dtype=int)
    poison_indices = rng.choice(n, size=n_poison, replace=False)

    for idx in poison_indices:
        orig = labels[idx]
        candidates = [c for c in range(num_classes) if c != orig]
        poisoned_labels[idx] = rng.choice(candidates)
        is_poisoned[idx] = 1

    group_ids = np.arange(n)

    return ThreatResult(
        images=images.copy(),
        labels=poisoned_labels,
        is_poisoned=is_poisoned,
        group_ids=group_ids,
        metadata={
            "threat_family": "random_label_flip",
            "poison_rate": poison_rate,
            "poison_count": n_poison,
            "num_classes": num_classes
        }
    )


# =====================================================================
# 2. TARGETED LABEL POISON
# =====================================================================
def targeted_label_poison(
    images: np.ndarray,
    labels: np.ndarray,
    source_class: int = 1,
    target_class: int = 0,
    poison_rate: float = 0.02,
    seed: int = 42
) -> ThreatResult:
    """
    Strategically flips labels of samples belonging to source_class into target_class.
    Images remain authentic; simulates intentional directed misclassification.
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    source_indices = np.where(labels == source_class)[0]

    n_poison = min(int(n * poison_rate), len(source_indices))
    poison_indices = rng.choice(source_indices, size=n_poison, replace=False)

    poisoned_labels = labels.copy()
    is_poisoned = np.zeros(n, dtype=int)

    poisoned_labels[poison_indices] = target_class
    is_poisoned[poison_indices] = 1

    group_ids = np.arange(n)

    return ThreatResult(
        images=images.copy(),
        labels=poisoned_labels,
        is_poisoned=is_poisoned,
        group_ids=group_ids,
        metadata={
            "threat_family": "targeted_label_poison",
            "source_class": source_class,
            "target_class": target_class,
            "poison_rate": poison_rate,
            "poison_count": n_poison
        }
    )


# =====================================================================
# 3. FEATURE / PIXEL PERTURBATION
# =====================================================================
def feature_perturbation(
    images: np.ndarray,
    labels: np.ndarray,
    poison_rate: float = 0.05,
    noise_std: float = 30.0,
    seed: int = 42
) -> ThreatResult:
    """
    Adds bounded zero-mean Gaussian pixel noise to poison_rate fraction of images.
    Labels remain clean; simulates sensor corruption or adversarial pixel noise.
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    n_poison = int(n * poison_rate)

    poisoned_images = images.copy().astype(np.float32)
    is_poisoned = np.zeros(n, dtype=int)
    poison_indices = rng.choice(n, size=n_poison, replace=False)

    for idx in poison_indices:
        noise = rng.normal(loc=0.0, scale=noise_std, size=images[idx].shape)
        poisoned_images[idx] = np.clip(poisoned_images[idx] + noise, 0, 255)
        is_poisoned[idx] = 1

    group_ids = np.arange(n)

    return ThreatResult(
        images=poisoned_images.astype(np.uint8),
        labels=labels.copy(),
        is_poisoned=is_poisoned,
        group_ids=group_ids,
        metadata={
            "threat_family": "feature_perturbation",
            "poison_rate": poison_rate,
            "poison_count": n_poison,
            "noise_std": noise_std
        }
    )


# =====================================================================
# 4. DUPLICATE INJECTION (WITH GROUP-ID TRACKING)
# =====================================================================
def duplicate_injection(
    images: np.ndarray,
    labels: np.ndarray,
    poison_rate: float = 0.05,
    near_duplicate_ratio: float = 0.5,
    seed: int = 42
) -> ThreatResult:
    """
    Injects duplicate copies (exact byte duplicates + near duplicates with subtle jitter).
    Assigns unified group_ids so duplicate clones stay strictly within the same train/val/test
    split, eliminating cross-split data leakage.
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    n_poison = int(n * poison_rate)

    selected_indices = rng.choice(n, size=n_poison, replace=False)

    dup_images = []
    dup_labels = []
    dup_is_poisoned = []
    dup_group_ids = []

    # Assign base group IDs for original samples
    orig_group_ids = np.arange(n)

    for i, idx in enumerate(selected_indices):
        img_copy = images[idx].copy().astype(np.float32)
        # Fraction exact, fraction slight jitter
        if rng.random() > near_duplicate_ratio:
            # Exact duplicate
            injected_img = img_copy.astype(np.uint8)
        else:
            # Near duplicate: slight brightness offset
            jitter = rng.uniform(-4.0, 4.0)
            injected_img = np.clip(img_copy + jitter, 0, 255).astype(np.uint8)

        dup_images.append(injected_img)
        dup_labels.append(labels[idx])
        dup_is_poisoned.append(1)
        # Duplicate takes the same group_id as the source image
        dup_group_ids.append(orig_group_ids[idx])

    final_images = np.concatenate([images, np.array(dup_images, dtype=np.uint8)], axis=0)
    final_labels = np.concatenate([labels, np.array(dup_labels, dtype=labels.dtype)], axis=0)
    final_is_poisoned = np.concatenate([np.zeros(n, dtype=int), np.array(dup_is_poisoned, dtype=int)], axis=0)
    final_group_ids = np.concatenate([orig_group_ids, np.array(dup_group_ids, dtype=int)], axis=0)

    return ThreatResult(
        images=final_images,
        labels=final_labels,
        is_poisoned=final_is_poisoned,
        group_ids=final_group_ids,
        metadata={
            "threat_family": "duplicate_injection",
            "poison_rate": poison_rate,
            "injected_duplicates": n_poison,
            "total_size": len(final_images)
        }
    )


# =====================================================================
# 5. CLEAN-LABEL MANIPULATION
# =====================================================================
def clean_label_manipulation(
    images: np.ndarray,
    labels: np.ndarray,
    target_class: int = 0,
    poison_rate: float = 0.05,
    blend_strength: float = 0.20,
    seed: int = 42
) -> ThreatResult:
    """
    Shifts images towards the centroid of target_class in pixel space,
    while keeping the true class label untouched.
    Forces feature-label discrepancy without altering the assigned label.
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    n_poison = int(n * poison_rate)

    target_indices = np.where(labels == target_class)[0]
    target_centroid = images[target_indices].mean(axis=0).astype(np.float32)

    # Avoid samples already in target class
    candidates = np.where(labels != target_class)[0]
    poison_indices = rng.choice(candidates, size=n_poison, replace=False)

    poisoned_images = images.copy().astype(np.float32)
    is_poisoned = np.zeros(n, dtype=int)

    for idx in poison_indices:
        poisoned_images[idx] = (
            (1.0 - blend_strength) * poisoned_images[idx] + blend_strength * target_centroid
        )
        is_poisoned[idx] = 1

    group_ids = np.arange(n)

    return ThreatResult(
        images=np.clip(poisoned_images, 0, 255).astype(np.uint8),
        labels=labels.copy(),
        is_poisoned=is_poisoned,
        group_ids=group_ids,
        metadata={
            "threat_family": "clean_label_manipulation",
            "target_class": target_class,
            "poison_rate": poison_rate,
            "poison_count": n_poison,
            "blend_strength": blend_strength
        }
    )


# =====================================================================
# 6. BACKDOOR / TROJAN PATCH
# =====================================================================
def add_patch_trigger(
    image: np.ndarray,
    patch_size: int = 6,
    color: Tuple[int, int, int] = (255, 255, 0),
    location: str = "bottom_right"
) -> np.ndarray:
    """Applies a square trigger patch to an image array."""
    img_mod = image.copy()
    h, w = img_mod.shape[:2]
    c = np.array(color, dtype=np.uint8)

    if location == "bottom_right":
        img_mod[h - patch_size:h, w - patch_size:w] = c
    elif location == "top_left":
        img_mod[0:patch_size, 0:patch_size] = c
    elif location == "top_right":
        img_mod[0:patch_size, w - patch_size:w] = c
    elif location == "bottom_left":
        img_mod[h - patch_size:h, 0:patch_size] = c
    elif location == "center":
        cy, cx = h // 2, w // 2
        hy = patch_size // 2
        img_mod[cy - hy:cy - hy + patch_size, cx - hy:cx - hy + patch_size] = c
    else:
        img_mod[h - patch_size:h, w - patch_size:w] = c

    return img_mod


def backdoor_patch(
    images: np.ndarray,
    labels: np.ndarray,
    poison_rate: float = 0.05,
    target_class: int = 0,
    patch_size: int = 6,
    color: Tuple[int, int, int] = (255, 255, 0),
    location: str = "bottom_right",
    seed: int = 42
) -> ThreatResult:
    """
    Injects a localized geometric patch trigger and flips label to target_class.
    Only non-target images are poisoned.
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    candidates = np.where(labels != target_class)[0]
    n_poison = int(n * poison_rate)
    poison_indices = rng.choice(candidates, size=n_poison, replace=False)

    poisoned_images = images.copy()
    poisoned_labels = labels.copy()
    is_poisoned = np.zeros(n, dtype=int)

    for idx in poison_indices:
        poisoned_images[idx] = add_patch_trigger(
            poisoned_images[idx], patch_size=patch_size, color=color, location=location
        )
        poisoned_labels[idx] = target_class
        is_poisoned[idx] = 1

    group_ids = np.arange(n)

    return ThreatResult(
        images=poisoned_images,
        labels=poisoned_labels,
        is_poisoned=is_poisoned,
        group_ids=group_ids,
        metadata={
            "threat_family": "backdoor_patch",
            "poison_rate": poison_rate,
            "target_class": target_class,
            "patch_size": patch_size,
            "color": color,
            "location": location,
            "poison_count": n_poison
        }
    )


# =====================================================================
# 7. BLENDED BACKDOOR (OPTIONAL)
# =====================================================================
def backdoor_blend(
    images: np.ndarray,
    labels: np.ndarray,
    poison_rate: float = 0.05,
    target_class: int = 0,
    blend_alpha: float = 0.20,
    seed: int = 42
) -> ThreatResult:
    """
    Blends a subtle checkerboard pattern across the entire image and sets label to target_class.
    Tests defense resilience against distributed, non-localized backdoor triggers.
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    h, w, c = images.shape[1:]

    # Synthesize static checkerboard pattern
    y_grid, x_grid = np.ogrid[:h, :w]
    pattern_mask = ((x_grid // 4) + (y_grid // 4)) % 2
    pattern = np.zeros((h, w, c), dtype=np.float32)
    pattern[pattern_mask == 1] = 255.0

    candidates = np.where(labels != target_class)[0]
    n_poison = int(n * poison_rate)
    poison_indices = rng.choice(candidates, size=n_poison, replace=False)

    poisoned_images = images.copy().astype(np.float32)
    poisoned_labels = labels.copy()
    is_poisoned = np.zeros(n, dtype=int)

    for idx in poison_indices:
        poisoned_images[idx] = (
            (1.0 - blend_alpha) * poisoned_images[idx] + blend_alpha * pattern
        )
        poisoned_labels[idx] = target_class
        is_poisoned[idx] = 1

    group_ids = np.arange(n)

    return ThreatResult(
        images=np.clip(poisoned_images, 0, 255).astype(np.uint8),
        labels=poisoned_labels,
        is_poisoned=is_poisoned,
        group_ids=group_ids,
        metadata={
            "threat_family": "backdoor_blend",
            "poison_rate": poison_rate,
            "target_class": target_class,
            "blend_alpha": blend_alpha,
            "poison_count": n_poison
        }
    )


# =====================================================================
# EVALUATION UTILITIES
# =====================================================================
def create_triggered_test_set(
    clean_test_images: np.ndarray,
    clean_test_labels: np.ndarray,
    target_class: int = 0,
    patch_size: int = 6,
    color: Tuple[int, int, int] = (255, 255, 0),
    location: str = "bottom_right"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stamps all non-target test images with the trigger patch to evaluate ASR.
    """
    non_target_mask = (clean_test_labels != target_class)
    triggered_imgs = clean_test_images[non_target_mask].copy()
    true_labels = clean_test_labels[non_target_mask].copy()

    for i in range(len(triggered_imgs)):
        triggered_imgs[i] = add_patch_trigger(
            triggered_imgs[i], patch_size=patch_size, color=color, location=location
        )

    return triggered_imgs, true_labels


def calculate_asr(
    model: nn.Module,
    loader: DataLoader,
    target_class: int,
    device: torch.device
) -> float:
    """Computes Attack Success Rate: fraction of triggered non-target test samples classified as target_class."""
    model.eval()
    total = 0
    success = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            success += (preds == target_class).sum().item()
            total += len(preds)

    return (success / total) if total > 0 else 0.0
