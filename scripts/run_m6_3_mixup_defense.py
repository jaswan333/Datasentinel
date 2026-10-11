"""
DataSentinel - M6.3 Defensive Training Research: Mixup Mitigation Experiment
Trains ResNet18 with Mixup (alpha=0.2) on the clean-label poisoned training set
and compares downstream clean test accuracy and attack success rate (ASR).
Treats defensive training strictly as a model-level mitigation, NOT detection.
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_m6_strong_benchmark as m6
from models.image.resnet_cifar import build_cifar_resnet18
from detectors.image.attacks import clean_label_manipulation


class CIFARImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


def train_mixup_epoch(model, loader, optimizer, device, alpha=0.2):
    model.train()
    total_loss = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        
        # Mixup interpolation
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        index = torch.randperm(imgs.size(0)).to(device)
        
        mixed_imgs = lam * imgs + (1 - lam) * imgs[index]
        lbls_a, lbls_b = lbls, lbls[index]
        
        optimizer.zero_grad()
        outputs = model(mixed_imgs)
        loss = lam * F.cross_entropy(outputs, lbls_a) + (1 - lam) * F.cross_entropy(outputs, lbls_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += len(lbls)
    return correct / total if total > 0 else 0.0


def calculate_clean_label_asr(model, clean_test_images, clean_test_labels, target_class=0, blend_strength=0.20, transform=None, device="cpu"):
    """
    Evaluates Attack Success Rate (ASR):
    Takes non-target test images, blends them with the target centroid,
    and measures the fraction misclassified into the target class (0).
    """
    model.eval()
    non_target_idx = np.where(clean_test_labels != target_class)[0]
    sub_imgs = clean_test_images[non_target_idx].copy().astype(np.float32)
    
    # Calculate target centroid from clean test or train
    target_idx = np.where(clean_test_labels == target_class)[0]
    if len(target_idx) == 0:
        return 0.0
    centroid = clean_test_images[target_idx].mean(axis=0).astype(np.float32)
    
    # Apply centroid blend
    blended_test_imgs = ((1.0 - blend_strength) * sub_imgs + blend_strength * centroid).astype(np.uint8)
    
    dataset = CIFARImageDataset(blended_test_imgs, clean_test_labels[non_target_idx], transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    total = 0
    target_preds = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1)
            target_preds += (preds == target_class).sum().item()
            total += len(preds)
            
    return target_preds / total if total > 0 else 0.0


def run_mixup_experiment(epochs=5, alpha=0.2, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    print(f"[*] Running Mixup Defensive Training Experiment (epochs={epochs}, alpha={alpha})...")

    # Load CIFAR
    tr_img, tr_lbl, te_img, te_lbl = m6.load_cifar(limit_train=1200, limit_test=300, seed=seed)
    
    # Synthesize clean-label poisoned training data
    threat = clean_label_manipulation(tr_img, tr_lbl, target_class=0, poison_rate=0.05, blend_strength=0.20, seed=seed)
    
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = CIFARImageDataset(threat.images, threat.labels, transform=tfm)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    test_dataset = CIFARImageDataset(te_img, te_lbl, transform=tfm)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 1. Evaluate Clean Reference Model
    clean_model = build_cifar_resnet18(10)
    clean_model.load_state_dict(torch.load("models/image/m6_clean_resnet18.pth", map_location=device))
    clean_ref_acc = evaluate_model(clean_model, test_loader, device)
    clean_ref_asr = calculate_clean_label_asr(clean_model, te_img, te_lbl, target_class=0, blend_strength=0.20, transform=tfm, device=device)

    # 2. Evaluate Standard Compromised Victim Model
    victim_model = build_cifar_resnet18(10)
    victim_model.load_state_dict(torch.load("models/image/m6_clean_label_victim.pth", map_location=device))
    victim_acc = evaluate_model(victim_model, test_loader, device)
    victim_asr = calculate_clean_label_asr(victim_model, te_img, te_lbl, target_class=0, blend_strength=0.20, transform=tfm, device=device)

    # 3. Train Fresh ResNet18 with Mixup Defensive Training
    print("[*] Training Mixup Defensive ResNet18 on Compromised Dataset...")
    mixup_model = build_cifar_resnet18(10)
    optimizer = torch.optim.Adam(mixup_model.parameters(), lr=1e-3)
    
    for epoch in range(1, epochs + 1):
        loss = train_mixup_epoch(mixup_model, train_loader, optimizer, device, alpha=alpha)
        acc = evaluate_model(mixup_model, test_loader, device)
        print(f"  Epoch {epoch}/{epochs} | Mixup Loss: {loss:.4f} | Clean Test Acc: {acc*100:.2f}%")

    mixup_acc = evaluate_model(mixup_model, test_loader, device)
    mixup_asr = calculate_clean_label_asr(mixup_model, te_img, te_lbl, target_class=0, blend_strength=0.20, transform=tfm, device=device)

    results = {
        "clean_reference_model": {
            "clean_test_acc": float(clean_ref_acc),
            "attack_success_rate": float(clean_ref_asr)
        },
        "standard_victim_model": {
            "clean_test_acc": float(victim_acc),
            "attack_success_rate": float(victim_asr)
        },
        "mixup_defended_model": {
            "clean_test_acc": float(mixup_acc),
            "attack_success_rate": float(mixup_asr),
            "mixup_alpha": alpha,
            "training_epochs": epochs
        },
        "scientific_conclusion": (
            f"Mixup defensive training achieves {mixup_acc*100:.2f}% clean test accuracy "
            f"and {mixup_asr*100:.2f}% ASR (compared to {victim_asr*100:.2f}% victim ASR). "
            "Mixup serves as an empirical training-time mitigation, smoothing label manifolds, "
            "but is not an automatic certified defense."
        )
    }

    out_dir = ROOT / "experiments/image/m6_3_clean_label_defense"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mixup_defense_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save model checkpoint
    torch.save(mixup_model.state_dict(), ROOT / "models/image/m6_clean_label_mixup_defended.pth")

    print("\n" + "=" * 70)
    print("M6.3 Mixup Defensive Training Results:")
    print(f"  Clean Reference Model: Clean Acc = {clean_ref_acc*100:.2f}% | ASR = {clean_ref_asr*100:.2f}%")
    print(f"  Standard Victim Model:  Clean Acc = {victim_acc*100:.2f}% | ASR = {victim_asr*100:.2f}%")
    print(f"  Mixup Defended Model:   Clean Acc = {mixup_acc*100:.2f}% | ASR = {mixup_asr*100:.2f}%")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_mixup_experiment(epochs=5, alpha=0.2, seed=42)
