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


def latent_cross_class_prototype_affinity_risk(
    embeddings: np.ndarray,
    labels: np.ndarray,
    reference_indices: Optional[np.ndarray] = None,
    metric: str = "cosine"
) -> np.ndarray:
    """Score whether an image's deep latent representation is unusually closer to another class prototype.

    Operates in the 512-dimensional pen-ultimate feature space extracted from ResNet18.
    Class prototypes are estimated strictly on the reference partition (`reference_indices`),
    eliminating data leakage.

    Unlike pixel-space affinity, deep latent representations are semantically structured,
    enabling detection of subtle clean-label manipulation that shifts feature representations
    toward target class clusters while preserving visual plausibility.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (N, 512) containing deep features.
    labels : np.ndarray
        Array of shape (N,) containing assigned class labels.
    reference_indices : Optional[np.ndarray]
        Indices for reference / training partition.
    metric : str
        Distance metric: 'cosine' (default) or 'euclidean'.

    Returns
    -------
    np.ndarray
        Calibrated per-class risk scores in [0.0, 1.0].
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n == 0 or embeddings.shape[0] != n:
        raise ValueError("embeddings and labels must have the same non-zero sample count")

    if reference_indices is None:
        reference_indices = np.arange(n, dtype=int)
    reference_indices = np.asarray(reference_indices, dtype=int)
    if len(reference_indices) == 0:
        raise ValueError("reference_indices cannot be empty")

    classes = np.unique(labels[reference_indices])
    if len(classes) < 2:
        return np.zeros(n, dtype=float)

    # L2-normalize embeddings onto unit hypersphere
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    X_norm = embeddings / norms

    prototypes = []
    valid_classes = []
    for cls in classes:
        c_ref = reference_indices[labels[reference_indices] == cls]
        if len(c_ref):
            proto = X_norm[c_ref].mean(axis=0, dtype=np.float32)
            p_norm = float(np.linalg.norm(proto)) + 1e-12
            prototypes.append(proto / p_norm)
            valid_classes.append(int(cls))

    if len(valid_classes) < 2:
        return np.zeros(n, dtype=float)

    P = np.vstack(prototypes).astype(np.float32)
    class_to_col = {cls: j for j, cls in enumerate(valid_classes)}

    if metric == "cosine":
        # Cosine similarity: S in [-1, 1], distance D in [0, 2]
        S = np.clip(X_norm @ P.T, -1.0, 1.0)
        distances = 1.0 - S
    else:
        # Normalized Euclidean distance
        x_sq = np.sum(X_norm ** 2, axis=1, keepdims=True)
        p_sq = np.sum(P ** 2, axis=1, keepdims=True).T
        d2 = np.maximum(x_sq + p_sq - 2.0 * (X_norm @ P.T), 0.0)
        distances = np.sqrt(d2)

    own_cols = np.array([class_to_col.get(int(y), -1) for y in labels], dtype=int)
    raw = np.zeros(n, dtype=np.float32)
    valid = own_cols >= 0
    rows = np.arange(n)[valid]

    own_dist = distances[rows, own_cols[valid]]
    other_dists = distances[valid].copy()
    other_dists[np.arange(len(rows)), own_cols[valid]] = np.inf
    comp_dist = other_dists.min(axis=1)

    # Positive deficit: sample is further from own prototype than nearest competing prototype
    raw[valid] = own_dist - comp_dist

    # Calibrate into percentile within assigned class reference partition
    risk = np.zeros(n, dtype=float)
    for cls in valid_classes:
        query_idx = np.where(labels == cls)[0]
        ref_idx = reference_indices[labels[reference_indices] == cls]
        if len(query_idx) and len(ref_idx):
            risk[query_idx] = _percentile(raw[query_idx], raw[ref_idx])

    return np.clip(risk, 0.0, 1.0)


def differential_loss_risk(
    clean_model: Any,
    victim_model: Any,
    images: np.ndarray,
    labels: np.ndarray,
    reference_indices: Optional[np.ndarray] = None,
    transform: Any = None,
    device: Any = None,
    batch_size: int = 64
) -> np.ndarray:
    """Measure the memorization gap between clean reference model and compromised victim model.

    Hypothesis: The victim model memorizes clean-label poisoned samples under their assigned
    label during training, leading to low victim loss. A clean reference model, never forced
    to fit the poisoned samples under their assigned label, experiences higher loss.
    Differential loss: Delta L = Loss_clean - Loss_victim.

    Calibrated into percentile risk using the reference/training partition only.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    images = np.asarray(images)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n == 0 or len(images) != n:
        raise ValueError("images and labels must have the same non-zero count")

    if reference_indices is None:
        reference_indices = np.arange(n, dtype=int)
    reference_indices = np.asarray(reference_indices, dtype=int)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class SimpleBatchDataset(Dataset):
        def __init__(self, imgs, lbls, tfm=None):
            self.imgs = imgs
            self.lbls = lbls
            self.tfm = tfm
        def __len__(self): return len(self.imgs)
        def __getitem__(self, idx):
            im = Image.fromarray(self.imgs[idx])
            if self.tfm: im = self.tfm(im)
            return im, int(self.lbls[idx])

    loader = DataLoader(SimpleBatchDataset(images, labels, transform), batch_size=batch_size, shuffle=False)

    clean_model.eval()
    victim_model.eval()

    clean_losses, victim_losses = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out_c = clean_model(imgs)
            out_v = victim_model(imgs)
            loss_c = F.cross_entropy(out_c, lbls, reduction="none").cpu().numpy()
            loss_v = F.cross_entropy(out_v, lbls, reduction="none").cpu().numpy()
            clean_losses.append(loss_c)
            victim_losses.append(loss_v)

    clean_losses = np.concatenate(clean_losses)
    victim_losses = np.concatenate(victim_losses)

    # Discrepancy: Clean model loss minus victim model loss
    raw_delta = (clean_losses - victim_losses).astype(np.float32)

    risk = np.zeros(n, dtype=float)
    classes = np.unique(labels[reference_indices])
    for cls in classes:
        query_idx = np.where(labels == cls)[0]
        ref_idx = reference_indices[labels[reference_indices] == cls]
        if len(query_idx) and len(ref_idx):
            risk[query_idx] = _percentile(raw_delta[query_idx], raw_delta[ref_idx])

    return np.clip(risk, 0.0, 1.0)


def frequency_high_ratio_risk(
    images: np.ndarray,
    labels: np.ndarray,
    reference_indices: Optional[np.ndarray] = None,
    cutoff_radius: float = 8.0
) -> np.ndarray:
    """Measure attenuation of high-frequency spatial energy caused by centroid blending.

    Hypothesis: Clean-label centroid manipulation blends natural images with a blurry,
    smooth class-mean image (centroid), which attenuates high-frequency texture and edges.
    A lower ratio of high-frequency power indicates potential centroid-smoothing.
    Risk is inverted so lower high-frequency ratio yields higher risk score.
    """
    images = np.asarray(images)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n == 0 or len(images) != n:
        raise ValueError("images and labels must have the same non-zero count")

    if reference_indices is None:
        reference_indices = np.arange(n, dtype=int)
    reference_indices = np.asarray(reference_indices, dtype=int)

    h, w = images.shape[1], images.shape[2]
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    high_mask = dist_from_center > cutoff_radius

    # Grayscale conversion weights
    rgb_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    high_ratios = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if images[i].ndim == 3 and images[i].shape[2] == 3:
            gray = images[i].astype(np.float32) @ rgb_weights
        else:
            gray = images[i].squeeze().astype(np.float32)
        f_shift = np.fft.fftshift(np.fft.fft2(gray))
        power = np.abs(f_shift) ** 2
        total_power = float(power.sum()) + 1e-12
        high_power = float(power[high_mask].sum())
        high_ratios[i] = high_power / total_power

    # Lower high-frequency ratio = more blurred/smoothed = higher risk
    # Raw deficit: -high_ratio
    raw_deficit = -high_ratios

    risk = np.zeros(n, dtype=float)
    classes = np.unique(labels[reference_indices])
    for cls in classes:
        query_idx = np.where(labels == cls)[0]
        ref_idx = reference_indices[labels[reference_indices] == cls]
        if len(query_idx) and len(ref_idx):
            risk[query_idx] = _percentile(raw_deficit[query_idx], raw_deficit[ref_idx])

    return np.clip(risk, 0.0, 1.0)


