"""
DataSentinel - Milestone 6: Multi-Attack Deep Learning Poisoning Benchmark & Defense
Runs end-to-end benchmark across 6 core image threat families (+ optional blend)
using a 14-signal evidence fusion architecture, strict 60/20/20 split protocol with
group_id preservation (preventing duplicate leakage), optimal threshold calibration,
quarantine isolation, model retraining, and clean accuracy + ASR recovery evaluation.
Modularized for Milestone 6.1 validation and false-positive hardening.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import joblib

sys.path.insert(0, os.path.abspath("."))
from models.image.resnet_cifar import build_cifar_resnet18, ResNetCIFAR18
from detectors.image.attacks import (
    ThreatResult,
    random_label_flip,
    targeted_label_poison,
    feature_perturbation,
    duplicate_injection,
    clean_label_manipulation,
    backdoor_patch,
    backdoor_blend,
    create_triggered_test_set,
    calculate_asr
)
from detectors.image.signals import extract_all_14_signals, SIGNAL_COLUMNS

# Module-level defaults & constants
SEED: int = 42
EPOCHS: int = 8
TARGET_CLASS: int = 0
SIGNAL_COLUMNS = list(SIGNAL_COLUMNS)


# =====================================================================
# DATASET HELPERS
# =====================================================================
class CIFARBatchDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        lbl = int(self.labels[idx])
        img_pil = Image.fromarray(img)
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, lbl


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr: float = 1e-3,
    name: str = "ResNet"
) -> nn.Module:
    """Trains ResNetCIFAR18 with Adam optimizer."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

        elapsed = time.time() - t0
        print(f"      [{name}] Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss/total:.4f} | Acc: {correct/total*100:.2f}% ({elapsed:.1f}s)")

    return model


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Returns (accuracy, macro_f1) on evaluation data."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(lbls.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = float(np.mean(all_preds == all_targets))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return acc, macro_f1


def extract_embeddings_and_logits(
    model: nn.Module,
    images: np.ndarray,
    batch_size: int,
    transform,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts 512-D latent representations and raw logits in one forward pass."""
    model.eval()
    loader = DataLoader(
        CIFARBatchDataset(images, np.zeros(len(images)), transform=transform),
        batch_size=batch_size,
        shuffle=False
    )

    all_embeds = []
    all_logits = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits, feats = model(imgs, return_features=True)
            all_embeds.append(feats.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    return np.vstack(all_embeds), np.vstack(all_logits)


# =====================================================================
# LEAKAGE-FREE SPLIT PROTOCOL WITH GROUP-ID PRESERVATION
# =====================================================================
def split_indices(
    stratify_labels: np.ndarray,
    group_ids: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a strict 60% Train / 20% Val / 20% Unseen Test split.
    Guarantees that all members sharing the same group_id stay in the same split,
    eliminating cross-partition duplicate leakage!
    Works seamlessly whether stratify_labels is binary (is_poisoned) or categorical.
    """
    stratify_indicator = np.asarray(stratify_labels)
    group_ids = np.asarray(group_ids)
    unique_groups = np.unique(group_ids)

    group_strat = []
    for g in unique_groups:
        members = (group_ids == g)
        vals = stratify_indicator[members]
        if np.isin(vals, [0, 1]).all() and len(np.unique(stratify_indicator)) <= 2:
            group_strat.append(int(np.any(vals == 1)))
        else:
            group_strat.append(int(np.bincount(vals).argmax()))
    group_strat = np.array(group_strat)

    train_groups, temp_groups = train_test_split(
        unique_groups,
        test_size=0.40,
        stratify=group_strat,
        random_state=seed
    )

    temp_strat = []
    for g in temp_groups:
        members = (group_ids == g)
        vals = stratify_indicator[members]
        if np.isin(vals, [0, 1]).all() and len(np.unique(stratify_indicator)) <= 2:
            temp_strat.append(int(np.any(vals == 1)))
        else:
            temp_strat.append(int(np.bincount(vals).argmax()))
    temp_strat = np.array(temp_strat)

    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=0.50,
        stratify=temp_strat,
        random_state=seed
    )

    train_idx = np.where(np.isin(group_ids, list(train_groups)))[0]
    val_idx = np.where(np.isin(group_ids, list(val_groups)))[0]
    test_idx = np.where(np.isin(group_ids, list(test_groups)))[0]

    return train_idx, val_idx, test_idx


create_group_aware_split = split_indices


def choose_threshold(y_val: np.ndarray, val_probs: np.ndarray) -> Tuple[float, float]:
    """Default baseline threshold search: maximize validation F1."""
    y = np.asarray(y_val, dtype=int)
    scores = np.asarray(val_probs, dtype=float)
    best_th = 0.50
    best_val_f1 = 0.0
    for th in np.arange(0.10, 0.91, 0.02):
        th = round(float(th), 2)
        preds = (scores >= th).astype(int)
        val_f1_score = f1_score(y, preds, zero_division=0)
        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            best_th = th
    return best_th, best_val_f1


# =====================================================================
# MODULAR CIFAR & REFERENCE MODEL HELPERS
# =====================================================================
def load_cifar(
    limit_train: int = 3000,
    limit_test: int = 1000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads a reproducible subset of CIFAR-10."""
    cifar_train_raw = datasets.CIFAR10(root="data/image/raw", train=True, download=False)
    cifar_test_raw = datasets.CIFAR10(root="data/image/raw", train=False, download=False)

    raw_train_imgs = np.array(cifar_train_raw.data)
    raw_train_lbls = np.array(cifar_train_raw.targets)
    raw_test_imgs = np.array(cifar_test_raw.data)
    raw_test_lbls = np.array(cifar_test_raw.targets)

    rng = np.random.default_rng(seed)
    clean_train_idx = rng.choice(len(raw_train_imgs), size=limit_train, replace=False)
    clean_test_idx = rng.choice(len(raw_test_imgs), size=limit_test, replace=False)

    clean_train_imgs = raw_train_imgs[clean_train_idx].copy()
    clean_train_lbls = raw_train_lbls[clean_train_idx].copy()
    clean_test_imgs = raw_test_imgs[clean_test_idx].copy()
    clean_test_lbls = raw_test_lbls[clean_test_idx].copy()

    return clean_train_imgs, clean_train_lbls, clean_test_imgs, clean_test_lbls


def run_clean_reference(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    epochs: Optional[int] = None,
    batch_size: int = 64,
    force_retrain: bool = False
) -> Tuple[nn.Module, Dict[str, float]]:
    """Trains or reuses clean reference ResNet18 model and returns (model, metrics)."""
    if epochs is None:
        epochs = EPOCHS
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    clean_test_dataset = CIFARBatchDataset(test_images, test_labels, transform=transform)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=batch_size, shuffle=False)

    clean_model_path = "models/image/m6_clean_resnet18.pth"
    clean_model = build_cifar_resnet18(num_classes=10)

    if os.path.exists(clean_model_path) and not force_retrain:
        print(f"    --> Reusing Clean Reference Model from {clean_model_path}...", flush=True)
        try:
            clean_model.load_state_dict(torch.load(clean_model_path, map_location=device))
        except Exception:
            clean_train_dataset = CIFARBatchDataset(train_images, train_labels, transform=transform)
            clean_train_loader = DataLoader(clean_train_dataset, batch_size=batch_size, shuffle=True)
            clean_model = train_classifier(clean_model, clean_train_loader, device, epochs=epochs, lr=1e-3, name="Clean Reference")
            torch.save(clean_model.state_dict(), clean_model_path)
    else:
        clean_train_dataset = CIFARBatchDataset(train_images, train_labels, transform=transform)
        clean_train_loader = DataLoader(clean_train_dataset, batch_size=batch_size, shuffle=True)
        clean_model = train_classifier(clean_model, clean_train_loader, device, epochs=epochs, lr=1e-3, name="Clean Reference")
        torch.save(clean_model.state_dict(), clean_model_path)

    clean_ref_acc, clean_ref_f1 = evaluate_accuracy(clean_model, clean_test_loader, device)
    print(f"    --> Clean Reference Accuracy: {clean_ref_acc*100:.2f}% | Macro F1: {clean_ref_f1:.4f}", flush=True)
    return clean_model, {"accuracy": clean_ref_acc, "macro_f1": clean_ref_f1}


def make_target_probabilities(
    clean_model: nn.Module,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    device: torch.device,
    batch_size: int = 64
) -> np.ndarray:
    """Computes soft prediction probabilities from reference clean model."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    _, logits = extract_embeddings_and_logits(clean_model, train_images, batch_size=batch_size, transform=transform, device=device)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    probs = F.softmax(logits_t, dim=1).numpy()
    return probs


def make_target_prototype(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    target_class: int = 0
) -> np.ndarray:
    """Calculates target class centroid in pixel space."""
    target_indices = np.where(train_labels == target_class)[0]
    if len(target_indices) == 0:
        return train_images.mean(axis=0).astype(np.float32)
    return train_images[target_indices].mean(axis=0).astype(np.float32)


# =====================================================================
# MODULAR EVIDENCE BUILDER
# =====================================================================
def build_evidence(
    attack_name: str,
    threat: Any,
    victim_model: nn.Module,
    train_loader: Optional[DataLoader],
    device: torch.device,
    enable_behavior: bool = True,
    batch_size: int = 64,
    *args,
    **kwargs
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Extracts 512-D latent representations and 14-signal evidence matrix.
    Can be monkey-patched by validation hardening overlays.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    embeddings, logits = extract_embeddings_and_logits(
        victim_model, threat.images, batch_size=batch_size, transform=transform, device=device
    )
    group_ids = getattr(threat, "group_id", getattr(threat, "group_ids", np.arange(len(threat.labels))))
    train_idx, _, _ = split_indices(
        threat.is_poisoned, group_ids, seed=SEED
    )
    df_signals = extract_all_14_signals(
        model=victim_model,
        images=threat.images,
        labels=threat.labels,
        embeddings=embeddings,
        logits=logits,
        train_indices=train_idx,
        device=device,
        compute_behavior=enable_behavior,
        seed=SEED
    )
    return df_signals, embeddings, threat.labels


# =====================================================================
# SINGLE ATTACK EXECUTION ENGINE
# =====================================================================
def run_one_attack(
    attack_name: str,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    target_probs: Optional[np.ndarray] = None,
    target_prototype: Optional[np.ndarray] = None,
    clean_model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    enable_behavior: bool = True,
    epochs: Optional[int] = None,
    batch_size: int = 64,
    force_retrain: bool = False,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Runs a single attack end-to-end and returns the benchmark metrics dictionary."""
    if device is None:
        device = get_device()
    if epochs is None:
        epochs = EPOCHS
    if seed is None:
        seed = SEED

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    clean_test_dataset = CIFARBatchDataset(test_images, test_labels, transform=transform)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=batch_size, shuffle=False)

    clean_ref_acc, clean_ref_f1 = (None, None)
    if clean_model is not None:
        clean_ref_acc, clean_ref_f1 = evaluate_accuracy(clean_model, clean_test_loader, device)

    # 1. Synthesize Threat
    print(f"  [1/6] Synthesizing {attack_name} threat vectors...")
    if attack_name == "random_label_flip":
        threat: ThreatResult = random_label_flip(train_images, train_labels, poison_rate=0.05, seed=seed)
    elif attack_name == "targeted_label_poison":
        threat: ThreatResult = targeted_label_poison(train_images, train_labels, source_class=1, target_class=0, poison_rate=0.03, seed=seed)
    elif attack_name == "feature_perturbation":
        threat: ThreatResult = feature_perturbation(train_images, train_labels, poison_rate=0.05, noise_std=32.0, seed=seed)
    elif attack_name == "duplicate_injection":
        threat: ThreatResult = duplicate_injection(train_images, train_labels, poison_rate=0.05, seed=seed)
    elif attack_name in ("clean_label", "clean_label_manipulation"):
        threat: ThreatResult = clean_label_manipulation(train_images, train_labels, target_class=TARGET_CLASS, poison_rate=0.05, seed=seed)
    elif attack_name == "backdoor_patch":
        threat: ThreatResult = backdoor_patch(train_images, train_labels, poison_rate=0.05, target_class=0, patch_size=6, seed=seed)
    elif attack_name == "backdoor_blend":
        threat: ThreatResult = backdoor_blend(train_images, train_labels, poison_rate=0.05, target_class=0, blend_alpha=0.20, seed=seed)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    n_samples = len(threat.images)
    n_poisoned = int(threat.is_poisoned.sum())
    poison_pct = (n_poisoned / n_samples) * 100
    print(f"    Total samples: {n_samples} | Poisoned instances: {n_poisoned} ({poison_pct:.2f}%)")

    # 2. Train Compromised Victim Model
    print(f"  [2/6] Training Compromised Victim Model ({epochs} epochs)...")
    victim_dataset = CIFARBatchDataset(threat.images, threat.labels, transform=transform)
    victim_loader = DataLoader(victim_dataset, batch_size=batch_size, shuffle=True)

    victim_model = build_cifar_resnet18(num_classes=10)
    victim_model = train_classifier(victim_model, victim_loader, device, epochs=epochs, lr=1e-3, name="Victim")

    victim_acc, victim_f1 = evaluate_accuracy(victim_model, clean_test_loader, device)
    print(f"    --> Victim Clean Accuracy: {victim_acc*100:.2f}% | Macro F1: {victim_f1:.4f}")

    # Check ASR if backdoor attack
    clean_asr = None
    victim_asr = None
    trig_loader = None
    if "backdoor" in attack_name:
        psize = threat.metadata.get("patch_size", 6)
        t_class = threat.metadata.get("target_class", 0)
        trig_test_imgs, trig_test_lbls = create_triggered_test_set(test_images, test_labels, target_class=t_class, patch_size=psize)
        trig_loader = DataLoader(CIFARBatchDataset(trig_test_imgs, trig_test_lbls, transform=transform), batch_size=batch_size, shuffle=False)

        if clean_model is not None:
            clean_asr = calculate_asr(clean_model, trig_loader, t_class, device)
        victim_asr = calculate_asr(victim_model, trig_loader, t_class, device)
        print(f"    --> Clean Model Baseline ASR: {(clean_asr or 0.0)*100:.2f}% | Victim Model ASR: {victim_asr*100:.2f}% (Trigger Active!)")

    # 3. Split Protocol: Strict 60/20/20 with Group ID Preservation
    print("  [3/6] Creating Group-Preserved 60/20/20 Train/Val/Test Partitions...")
    train_idx, val_idx, test_idx = split_indices(threat.is_poisoned, getattr(threat, "group_id", threat.group_ids), seed=seed)
    print(f"    Partitions: {len(train_idx)} Train (60%) | {len(val_idx)} Val (20%) | {len(test_idx)} Test (20%)")

    # 4. Evidence Signals Extraction via build_evidence (can be monkey-patched)
    print("  [4/6] Extracting Multi-Evidence Signals Matrix...")
    df_signals, embeddings, labels_used = build_evidence(
        attack_name, threat, victim_model, victim_loader, device, enable_behavior=enable_behavior, batch_size=batch_size
    )

    # 5. Train Evidence Fusion Meta-Classifier
    print("  [5/6] Calibrating Evidence Fusion & Operating Threshold...")
    current_signal_cols = [c for c in SIGNAL_COLUMNS if c in df_signals.columns]
    X_feats = df_signals[current_signal_cols].values
    y_true = threat.is_poisoned

    fusion_model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    fusion_model.fit(X_feats[train_idx], y_true[train_idx])

    val_probs = fusion_model.predict_proba(X_feats[val_idx])[:, 1]
    best_th, best_val_f1 = choose_threshold(y_true[val_idx], val_probs)
    best_th = float(best_th)

    # Evaluate on unseen test partition
    test_probs = fusion_model.predict_proba(X_feats[test_idx])[:, 1]
    test_preds = (test_probs >= best_th).astype(int)

    t_prec = float(precision_score(y_true[test_idx], test_preds, zero_division=0))
    t_rec = float(recall_score(y_true[test_idx], test_preds, zero_division=0))
    t_f1 = float(f1_score(y_true[test_idx], test_preds, zero_division=0))
    t_roc_auc = float(roc_auc_score(y_true[test_idx], test_probs)) if len(np.unique(y_true[test_idx])) > 1 else 0.5
    t_pr_auc = float(average_precision_score(y_true[test_idx], test_probs)) if len(np.unique(y_true[test_idx])) > 1 else 0.0

    cm = confusion_matrix(y_true[test_idx], test_preds)
    tn = cm[0, 0] if cm.shape == (2, 2) else 0
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    t_fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    print(f"\n    [DETECTOR METRICS on Unseen Test Split | Threshold = {best_th:.4f}]:")
    print(f"      Precision: {t_prec*100:.2f}% | Recall: {t_rec*100:.2f}% | F1: {t_f1:.4f}")
    print(f"      ROC-AUC:   {t_roc_auc:.4f} | PR-AUC: {t_pr_auc:.4f} | FPR: {t_fpr*100:.2f}%")

    # Full Dataset Sample Risk & Quarantine Manifest
    all_risk_probs = fusion_model.predict_proba(X_feats)[:, 1]
    quarantine_mask = (all_risk_probs >= best_th)
    n_quarantine = int(quarantine_mask.sum())
    q_true_pos = int((quarantine_mask & (y_true == 1)).sum())
    q_prec = float(q_true_pos / n_quarantine) if n_quarantine > 0 else 0.0
    q_rec = float(q_true_pos / n_poisoned) if n_poisoned > 0 else 0.0

    print(f"\n    [QUARANTINE MANIFEST]:")
    print(f"      Quarantined: {n_quarantine} / {n_samples} ({n_quarantine/n_samples*100:.2f}%)")
    print(f"      Purged Poisoned: {q_true_pos} / {n_poisoned} (Recall: {q_rec*100:.2f}%) | Precision: {q_prec*100:.2f}%")

    # Dataset Trust Score
    mean_risk = float(all_risk_probs.mean())
    high_risk_rate = float(np.mean(all_risk_probs >= 0.55))
    dataset_risk_index = round(float((mean_risk * 0.5 + high_risk_rate * 0.5) * 100), 2)
    dataset_trust_score = round(float(100.0 - dataset_risk_index), 2)
    risk_level = "HIGH_RISK" if dataset_trust_score < 60 else ("MODERATE_RISK" if dataset_trust_score < 80 else "LOW_RISK")

    # 6. Screened Dataset Retraining
    print("\n    [REMEDIATION] Retraining Fresh Screened ResNet18 on Purified Data...")
    keep_mask = ~quarantine_mask
    if keep_mask.sum() == 0:
        print("      [WARNING] All samples quarantined; keeping all samples for screened evaluation.")
        keep_mask = np.ones(n_samples, dtype=bool)

    screened_dataset = CIFARBatchDataset(threat.images[keep_mask], threat.labels[keep_mask], transform=transform)
    screened_loader = DataLoader(screened_dataset, batch_size=batch_size, shuffle=True)

    screened_model = build_cifar_resnet18(num_classes=10)
    screened_model = train_classifier(screened_model, screened_loader, device, epochs=epochs, lr=1e-3, name="Screened")

    screened_acc, screened_f1 = evaluate_accuracy(screened_model, clean_test_loader, device)
    screened_asr = None
    asr_reduction = None
    if "backdoor" in attack_name and victim_asr is not None and trig_loader is not None:
        screened_asr = calculate_asr(screened_model, trig_loader, threat.metadata.get("target_class", 0), device)
        asr_reduction = (victim_asr - screened_asr) * 100.0
        print(f"      Victim ASR: {victim_asr*100:.2f}% --> Screened ASR: {screened_asr*100:.2f}% (Neutralized: {asr_reduction:+.2f} pp!)")

    print(f"      Clean Reference Acc: {(clean_ref_acc or 0.0)*100:.2f}% | Victim Acc: {victim_acc*100:.2f}% | Screened Acc: {screened_acc*100:.2f}%")

    # Persist Sample Evidence Table
    split_col = np.array(["train"] * n_samples, dtype=object)
    split_col[val_idx] = "val"
    split_col[test_idx] = "test"

    df_out = pd.DataFrame({
        "sample_id": np.arange(n_samples),
        "label": threat.labels,
        "attack": attack_name,
        "is_poisoned": threat.is_poisoned,
        "group_id": getattr(threat, "group_id", threat.group_ids),
        "partition": split_col
    })
    for col in current_signal_cols:
        df_out[col] = df_signals[col]

    df_out["risk_probability"] = np.round(all_risk_probs, 4)
    df_out["risk_level"] = [
        "HIGH" if p >= 0.55 else ("MEDIUM" if p >= 0.35 else "LOW") for p in all_risk_probs
    ]
    df_out["quarantine"] = quarantine_mask

    dirs_to_save = [f"experiments/image/{attack_name}"]
    if attack_name in ("clean_label", "clean_label_manipulation"):
        dirs_to_save = ["experiments/image/clean_label", "experiments/image/clean_label_manipulation"]

    for adir in dirs_to_save:
        os.makedirs(adir, exist_ok=True)
        df_out.to_csv(f"{adir}/sample_evidence.csv", index=False)
        df_out[quarantine_mask].to_csv(f"{adir}/quarantine.csv", index=False)

    torch.save(victim_model.state_dict(), f"models/image/m6_{attack_name}_victim.pth")
    torch.save(screened_model.state_dict(), f"models/image/m6_{attack_name}_screened.pth")
    joblib.dump(fusion_model, f"models/image/m6_{attack_name}_fusion_model.pkl")

    delta_pp = round(float(screened_acc * 100 - victim_acc * 100), 2)
    attack_record = {
        "attack": attack_name,
        "poison_count": n_poisoned,
        "poison_rate_pct": round(poison_pct, 2),
        "clean_ref_acc": round(float((clean_ref_acc or 0.0) * 100), 2),
        "victim_acc": round(float(victim_acc * 100), 2),
        "screened_acc": round(float(screened_acc * 100), 2),
        "victim_clean_accuracy": round(float(victim_acc * 100), 2),
        "screened_clean_accuracy": round(float(screened_acc * 100), 2),
        "clean_accuracy_delta_pp": delta_pp,
        "victim_macro_f1": round(float(victim_f1), 4),
        "screened_macro_f1": round(float(screened_f1), 4),
        "detector_threshold": float(best_th),
        "test_precision": round(float(t_prec * 100), 2),
        "test_recall": round(float(t_rec * 100), 2),
        "test_f1": round(float(t_f1), 4),
        "test_roc_auc": round(float(t_roc_auc), 4),
        "test_pr_auc": round(float(t_pr_auc), 4),
        "test_fpr": round(float(t_fpr * 100), 2),
        "detector_roc_auc": round(float(t_roc_auc), 4),
        "detector_pr_auc": round(float(t_pr_auc), 4),
        "detector_precision": round(float(t_prec * 100), 2),
        "detector_recall": round(float(t_rec * 100), 2),
        "detector_false_positive_rate": round(float(t_fpr * 100), 2),
        "detector_quarantine_rate": round(float(n_quarantine / n_samples), 4),
        "quarantined_count": n_quarantine,
        "quarantine_recall_pct": round(float(q_rec * 100), 2),
        "quarantine_precision_pct": round(float(q_prec * 100), 2),
        "clean_asr": round(float(clean_asr * 100), 2) if clean_asr is not None else None,
        "victim_asr": round(float(victim_asr * 100), 2) if victim_asr is not None else None,
        "screened_asr": round(float(screened_asr * 100), 2) if screened_asr is not None else None,
        "asr_reduction_pp": round(float(asr_reduction), 2) if asr_reduction is not None else None,
        "dataset_trust_score": dataset_trust_score,
        "dataset_risk_level": risk_level
    }

    for adir in dirs_to_save:
        with open(f"{adir}/attack_summary.json", "w", encoding="utf-8") as f:
            json.dump(attack_record, f, indent=2)
        with open(f"{adir}/summary.json", "w", encoding="utf-8") as f:
            json.dump(attack_record, f, indent=2)

    return attack_record


# =====================================================================
# MAIN BENCHMARK ENGINE
# =====================================================================
def run_strong_benchmark(
    attacks: Optional[List[str]] = None,
    epochs: int = 8,
    train_limit: int = 3000,
    test_limit: int = 1000,
    batch_size: int = 64,
    compute_behavior: bool = True,
    force_retrain: bool = False,
    seed: int = 42
):
    print("=" * 80)
    print("DataSentinel: Multi-Attack Image Poisoning Benchmark & 14-Signal Defense")
    print("=" * 80)

    if attacks is None or len(attacks) == 0:
        attacks = [
            "random_label_flip",
            "targeted_label_poison",
            "feature_perturbation",
            "duplicate_injection",
            "clean_label_manipulation",
            "backdoor_patch"
        ]

    print(f"[*] Evaluated threat families: {attacks}")
    print(f"[*] Training epochs: {epochs} | Train limit: {train_limit} | Test limit: {test_limit}")
    print(f"[*] Local patch behavior signal: {'ENABLED' if compute_behavior else 'DISABLED'}")

    os.makedirs("models/image", exist_ok=True)
    os.makedirs("experiments/image", exist_ok=True)
    os.makedirs("data/image/raw", exist_ok=True)

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    print(f"[*] Execution device: {device} (4 threads)", flush=True)

    print("\n[*] Ingesting CIFAR-10...", flush=True)
    clean_train_imgs, clean_train_lbls, clean_test_imgs, clean_test_lbls = load_cifar(
        limit_train=train_limit, limit_test=test_limit, seed=seed
    )

    # 1. Train or Load Clean Reference Model
    print("\n[Phase 1] Training or Loading Clean Reference Model...")
    clean_model, clean_metrics = run_clean_reference(
        clean_train_imgs, clean_train_lbls, clean_test_imgs, clean_test_lbls, device,
        epochs=epochs, batch_size=batch_size, force_retrain=force_retrain
    )

    target_probs = make_target_probabilities(clean_model, clean_train_imgs, clean_train_lbls, device, batch_size=batch_size)
    target_prototype = make_target_prototype(clean_train_imgs, clean_train_lbls, target_class=TARGET_CLASS)

    benchmark_summary: List[Dict[str, Any]] = []

    # 2. Iterate Threat Families
    for attack_name in attacks:
        print("\n" + "=" * 70, flush=True)
        print(f"[Phase 2] Executing Threat Benchmark: {attack_name.upper()}", flush=True)
        print("=" * 70, flush=True)

        attack_dir = f"experiments/image/{attack_name}"
        os.makedirs(attack_dir, exist_ok=True)

        summary_file = f"{attack_dir}/attack_summary.json"
        evidence_file = f"{attack_dir}/sample_evidence.csv"
        if os.path.exists(summary_file) and os.path.exists(evidence_file) and not force_retrain:
            print(f"  --> Found existing completed benchmark for {attack_name}. Loading from cache...", flush=True)
            with open(summary_file, "r", encoding="utf-8") as f:
                cached_rec = json.load(f)
            benchmark_summary.append(cached_rec)
            continue

        attack_record = run_one_attack(
            attack_name=attack_name,
            train_images=clean_train_imgs,
            train_labels=clean_train_lbls,
            test_images=clean_test_imgs,
            test_labels=clean_test_lbls,
            target_probs=target_probs,
            target_prototype=target_prototype,
            clean_model=clean_model,
            device=device,
            enable_behavior=compute_behavior,
            epochs=epochs,
            batch_size=batch_size,
            force_retrain=force_retrain,
            seed=seed
        )
        benchmark_summary.append(attack_record)

    # Master Benchmark Summary Artifacts
    df_summary = pd.DataFrame(benchmark_summary)
    summary_csv = "experiments/image/m6_strong_benchmark_summary.csv"
    summary_json = "experiments/image/m6_strong_benchmark_summary.json"
    df_summary.to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    # Master Visual Dashboard
    create_benchmark_dashboard(benchmark_summary)

    print("\n" + "=" * 90)
    print("DATA SENTINEL MILESTONE 6: MASTER BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Attack':<24} | {'Victim Acc':<10} | {'Screened Acc':<12} | {'ROC-AUC':<8} | {'Quar. Rec':<10} | {'Backdoor ASR (Victim -> Screened)':<32}")
    print("-" * 90)
    for r in benchmark_summary:
        asr_str = f"{r['victim_asr']:.1f}% -> {r['screened_asr']:.1f}% ({r['asr_reduction_pp']:+.1f}pp)" if r.get('victim_asr') is not None else "N/A"
        print(f"{r['attack']:<24} | {r['victim_acc']:>9.2f}% | {r['screened_acc']:>11.2f}% | {r['test_roc_auc']:>8.4f} | {r['quarantine_recall_pct']:>9.1f}% | {asr_str:<32}")
    print("=" * 90)
    print(f"[SUCCESS] All Milestone 6 artifacts generated successfully!")
    print(f"  - Summary CSV:  {summary_csv}")
    print(f"  - Summary JSON: {summary_json}")
    print(f"  - Dashboard:    experiments/image/m6_strong_benchmark_dashboard.png")


def create_benchmark_dashboard(summary: List[Dict[str, Any]]):
    """Creates a 4-panel comprehensive visual benchmark dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    attacks = [s["attack"].replace("_", "\n") for s in summary]
    x = np.arange(len(attacks))

    # Panel 1: Downstream Model Accuracy (Victim vs Screened)
    v_accs = [s["victim_acc"] for s in summary]
    s_accs = [s["screened_acc"] for s in summary]
    width = 0.35
    axes[0, 0].bar(x - width/2, v_accs, width, label="Victim Model", color="#ef4444")
    axes[0, 0].bar(x + width/2, s_accs, width, label="Screened Model", color="#10b981")
    axes[0, 0].set_ylabel("Clean Accuracy (%)")
    axes[0, 0].set_title("Downstream Model Utility Preservation", fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(attacks, fontsize=9)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Detector Detection ROC-AUC
    rocs = [s["test_roc_auc"] for s in summary]
    bars = axes[0, 1].bar(x, rocs, color="#3b82f6", width=0.5)
    axes[0, 1].axhline(0.5, color="red", linestyle="--", label="Random Chance (0.50)")
    axes[0, 1].axhline(0.75, color="green", linestyle=":", label="Acceptance Target (0.75)")
    axes[0, 1].set_ylabel("Test ROC-AUC")
    axes[0, 1].set_title("14-Signal Evidence Fusion Discriminative Power", fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(attacks, fontsize=9)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Quarantine Recall & Precision
    q_recs = [s["quarantine_recall_pct"] for s in summary]
    q_precs = [s["quarantine_precision_pct"] for s in summary]
    axes[1, 0].bar(x - width/2, q_recs, width, label="Quarantine Recall (%)", color="#8b5cf6")
    axes[1, 0].bar(x + width/2, q_precs, width, label="Quarantine Precision (%)", color="#f59e0b")
    axes[1, 0].set_ylabel("Percentage (%)")
    axes[1, 0].set_title("Quarantine Defense Efficiency (Poison Purge Rate)", fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(attacks, fontsize=9)
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 4: Backdoor Attack Success Rate (ASR) Neutralization
    bd_attacks = [s for s in summary if s.get("victim_asr") is not None]
    if bd_attacks:
        bd_names = [s["attack"].replace("_", "\n") for s in bd_attacks]
        bd_x = np.arange(len(bd_names))
        clean_asrs = [s["clean_asr"] for s in bd_attacks]
        victim_asrs = [s["victim_asr"] for s in bd_attacks]
        screened_asrs = [s["screened_asr"] for s in bd_attacks]

        w3 = 0.25
        axes[1, 1].bar(bd_x - w3, clean_asrs, w3, label="Clean Baseline ASR", color="#64748b")
        axes[1, 1].bar(bd_x, victim_asrs, w3, label="Victim Model ASR", color="#dc2626")
        axes[1, 1].bar(bd_x + w3, screened_asrs, w3, label="Screened Model ASR", color="#059669")
        axes[1, 1].set_ylabel("Attack Success Rate (%)")
        axes[1, 1].set_title("Trojan Backdoor Attack Neutralization", fontweight="bold")
        axes[1, 1].set_xticks(bd_x)
        axes[1, 1].set_xticklabels(bd_names, fontsize=9)
        axes[1, 1].legend()
        axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
    else:
        axes[1, 1].text(0.5, 0.5, "No Backdoor Attacks in Run", ha="center", va="center")

    plt.tight_layout()
    plt.savefig("experiments/image/m6_strong_benchmark_dashboard.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataSentinel M6 Multi-Attack Deep Learning Benchmark")
    parser.add_argument("--attacks", nargs="+", default=None, help="Attacks to run (default: all 6)")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs per model")
    parser.add_argument("--train-limit", type=int, default=3000, help="Clean training dataset limit")
    parser.add_argument("--test-limit", type=int, default=1000, help="Clean test dataset limit")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--no-behavior-signal", action="store_true", help="Disable signal 14 for faster run")
    parser.add_argument("--force-retrain", action="store_true", help="Force retrain even if cached summary exists")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_strong_benchmark(
        attacks=args.attacks,
        epochs=args.epochs,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        batch_size=args.batch_size,
        compute_behavior=not args.no_behavior_signal,
        force_retrain=args.force_retrain,
        seed=args.seed
    )
