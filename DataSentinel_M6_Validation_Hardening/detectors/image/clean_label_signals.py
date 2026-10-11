"""Additional image signals for DataSentinel M6 clean-label hardening."""
from __future__ import annotations
from typing import Optional
import numpy as np


def _percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    reference = np.sort(reference[np.isfinite(reference)])
    if len(reference) == 0:
        return np.zeros(len(values), dtype=float)
    return np.clip(np.searchsorted(reference, values, side="right") / len(reference), 0.0, 1.0)


def cross_class_prototype_affinity_risk(
    images: np.ndarray,
    labels: np.ndarray,
    reference_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score whether an image is unusually closer to another class prototype.

    The reference prototypes and per-class calibration distributions are fitted only on
    `reference_indices` (the fusion-training partition). The signal is particularly
    relevant to clean-label manipulation: an image keeps its source label but is shifted
    visually toward another class prototype.

    This signal is heuristic; CIFAR pixel-space prototypes may not separate all image
    semantics. Its value must be measured on the held-out partition rather than assumed.
    """
    images = np.asarray(images)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n == 0 or images.shape[0] != n:
        raise ValueError("images and labels must have the same non-zero sample count")

    if reference_indices is None:
        reference_indices = np.arange(n, dtype=int)
    reference_indices = np.asarray(reference_indices, dtype=int)
    if len(reference_indices) == 0:
        raise ValueError("reference_indices cannot be empty")

    classes = np.unique(labels[reference_indices])
    if len(classes) < 2:
        return np.zeros(n, dtype=float)

    # CIFAR-10 images are small; flattening lets us calculate all sample-to-prototype
    # distances through one matrix multiplication without creating an N x C x D tensor.
    X = images.astype(np.float32).reshape(n, -1) / 255.0
    prototypes, valid_classes = [], []
    for cls in classes:
        class_ref = reference_indices[labels[reference_indices] == cls]
        if len(class_ref):
            prototypes.append(X[class_ref].mean(axis=0, dtype=np.float32))
            valid_classes.append(int(cls))
    if len(valid_classes) < 2:
        return np.zeros(n, dtype=float)

    P = np.vstack(prototypes).astype(np.float32)
    class_to_col = {cls: j for j, cls in enumerate(valid_classes)}
    x_norm = np.einsum("ij,ij->i", X, X)[:, None]
    p_norm = np.einsum("ij,ij->i", P, P)[None, :]
    d2 = np.maximum(x_norm + p_norm - 2.0 * (X @ P.T), 0.0)
    distances = np.sqrt(d2).astype(np.float32, copy=False)

    own_cols = np.array([class_to_col.get(int(y), -1) for y in labels], dtype=int)
    raw = np.zeros(n, dtype=np.float32)
    valid = own_cols >= 0
    rows = np.arange(n)[valid]
    own_distance = distances[rows, own_cols[valid]]
    other_distances = distances[valid].copy()
    other_distances[np.arange(len(rows)), own_cols[valid]] = np.inf
    nearest_other_distance = other_distances.min(axis=1)
    # Larger means the sample is farther from its assigned-class prototype than from
    # at least one competing class prototype.
    raw[valid] = own_distance - nearest_other_distance

    risk = np.zeros(n, dtype=float)
    for cls in valid_classes:
        query_idx = np.where(labels == cls)[0]
        ref_idx = reference_indices[labels[reference_indices] == cls]
        if len(query_idx) and len(ref_idx):
            risk[query_idx] = _percentile(raw[query_idx], raw[ref_idx])
    return np.clip(risk, 0.0, 1.0)
