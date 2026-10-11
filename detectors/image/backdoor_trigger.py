"""
DataSentinel - Image Backdoor Trigger Module
Provides controlled trigger synthesis, dataset poisoning, and Attack Success Rate (ASR) evaluation.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


DEFAULT_TRIGGER_COLOR = np.array([255, 255, 0], dtype=np.uint8)  # Yellow patch (RGB)


def add_trigger(
    image: np.ndarray,
    patch_size: int = 4,
    color: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Applies a natural trigger patch to the bottom-right corner of an image.

    Parameters:
        image: Input image array of shape (H, W, C).
        patch_size: Square edge length of the patch (default: 4 for 32x32 image).
        color: RGB uint8 color array (default: yellow [255, 255, 0]).

    Returns:
        Modified image array with trigger embedded.
    """
    img_mod = image.copy()
    c = color if color is not None else DEFAULT_TRIGGER_COLOR
    h, w = img_mod.shape[:2]
    img_mod[h - patch_size:h, w - patch_size:w] = c
    return img_mod


def create_backdoor_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    poison_rate: float = 0.03,
    target_class: int = 0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs a controlled backdoored training dataset.
    Only non-target images are poisoned to simulate targeted misclassification.

    Parameters:
        images: Clean feature images (N, H, W, C).
        labels: Original class labels (N,).
        poison_rate: Fraction of training samples to corrupt (default: 0.03 = 3%).
        target_class: Attacker target class index (default: 0).
        seed: Random seed for reproducibility.

    Returns:
        (poisoned_images, poisoned_labels, is_poisoned_mask)
    """
    rng = np.random.default_rng(seed)
    poisoned_images = images.copy()
    poisoned_labels = labels.copy()
    is_poisoned = np.zeros(len(images), dtype=np.int64)

    # Avoid poisoning samples already in target class
    candidates = np.where(labels != target_class)[0]
    poison_count = int(len(images) * poison_rate)
    poison_indices = rng.choice(candidates, size=poison_count, replace=False)

    for idx in poison_indices:
        poisoned_images[idx] = add_trigger(poisoned_images[idx])
        poisoned_labels[idx] = target_class
        is_poisoned[idx] = 1

    return poisoned_images, poisoned_labels, is_poisoned


def create_triggered_test_set(
    images: np.ndarray,
    labels: np.ndarray,
    target_class: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates an evaluation test set of clean non-target images stamped with the trigger.
    """
    indices = np.where(labels != target_class)[0]
    triggered_imgs = images[indices].copy()
    orig_lbls = labels[indices].copy()

    for i in range(len(triggered_imgs)):
        triggered_imgs[i] = add_trigger(triggered_imgs[i])

    return triggered_imgs, orig_lbls


def calculate_asr(
    net: nn.Module,
    loader: DataLoader,
    target_class: int,
    device: torch.device
) -> float:
    """
    Calculates the Attack Success Rate (ASR) of a model on triggered non-target test images.
    """
    net.eval()
    total = 0
    successful = 0

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = net(imgs)
            preds = outputs.argmax(dim=1)
            successful += (preds == target_class).sum().item()
            total += len(preds)

    return successful / total if total > 0 else 0.0
