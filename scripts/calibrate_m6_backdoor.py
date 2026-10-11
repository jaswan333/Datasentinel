"""
DataSentinel - Milestone 6: Image Backdoor Attack Calibration
Scientifically calibrates backdoor configurations before defense evaluation.
Evaluates:
  Config 1 (baseline): 3% poison / 4x4 patch / 5 epochs
  Config 2 (strong_1): 5% poison / 6x6 patch / 10 epochs
  Config 3 (strong_2): 7% poison / 6x6 patch / 12 epochs
  Config 4 (strong_3): 5% poison / 8x8 patch / 10 epochs

Outputs:
  experiments/image/backdoor_calibration/m6_backdoor_calibration.csv
  experiments/image/backdoor_calibration/m6_backdoor_calibration.json
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.abspath("."))
from models.image.resnet_cifar import build_cifar_resnet18
from detectors.image.attacks import backdoor_patch, create_triggered_test_set, calculate_asr


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


def train_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
    name: str = "ResNet"
) -> nn.Module:
    """Trains CIFAR ResNet18 with Adam and CrossEntropyLoss."""
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
        print(f"    [{name}] Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss/total:.4f} | Acc: {correct/total*100:.2f}% ({elapsed:.1f}s)")

    return model


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluates classification accuracy on clean evaluation data."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    return correct / total if total > 0 else 0.0


# =====================================================================
# CALIBRATION EXPERIMENT RUNNER
# =====================================================================
def run_calibration(
    train_limit: int = 3000,
    test_limit: int = 1000,
    batch_size: int = 64,
    quick_mode: bool = False,
    seed: int = 42
):
    print("=" * 75)
    print("DataSentinel M6: Backdoor Attack Strength Calibration")
    print("=" * 75)

    os.makedirs("experiments/image/backdoor_calibration", exist_ok=True)
    os.makedirs("data/image/raw", exist_ok=True)

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device} (4 threads)", flush=True)

    # Standard CIFAR-10 Normalization
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    print("[*] Loading CIFAR-10 dataset...", flush=True)
    cifar_train_raw = datasets.CIFAR10(root="data/image/raw", train=True, download=False)
    cifar_test_raw = datasets.CIFAR10(root="data/image/raw", train=False, download=False)

    raw_train_imgs = np.array(cifar_train_raw.data)
    raw_train_lbls = np.array(cifar_train_raw.targets)
    raw_test_imgs = np.array(cifar_test_raw.data)
    raw_test_lbls = np.array(cifar_test_raw.targets)

    # Subsampling for fast CPU calibration
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(raw_train_imgs), size=train_limit, replace=False)
    test_idx = rng.choice(len(raw_test_imgs), size=test_limit, replace=False)

    clean_train_imgs = raw_train_imgs[train_idx].copy()
    clean_train_lbls = raw_train_lbls[train_idx].copy()
    clean_test_imgs = raw_test_imgs[test_idx].copy()
    clean_test_lbls = raw_test_lbls[test_idx].copy()

    clean_test_dataset = CIFARBatchDataset(clean_test_imgs, clean_test_lbls, transform=transform_test)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=batch_size, shuffle=False)

    # 1. Train clean reference model
    print("\n[Step 1] Training Clean Reference Model...")
    clean_train_dataset = CIFARBatchDataset(clean_train_imgs, clean_train_lbls, transform=transform_train)
    clean_train_loader = DataLoader(clean_train_dataset, batch_size=batch_size, shuffle=True)

    clean_epochs = 4 if quick_mode else 6
    clean_model = build_cifar_resnet18(num_classes=10)
    clean_model = train_model(clean_model, clean_train_loader, device, epochs=clean_epochs, lr=1e-3, name="Clean Ref")

    clean_ref_acc = evaluate_accuracy(clean_model, clean_test_loader, device)
    print(f"    --> Clean Reference Accuracy: {clean_ref_acc * 100:.2f}%")

    # Configurations to test
    if quick_mode:
        configs = [
            {"name": "baseline", "poison_rate": 0.03, "patch_size": 4, "epochs": 3},
            {"name": "strong_1", "poison_rate": 0.05, "patch_size": 6, "epochs": 5},
            {"name": "strong_2", "poison_rate": 0.07, "patch_size": 6, "epochs": 5},
            {"name": "strong_3", "poison_rate": 0.05, "patch_size": 8, "epochs": 5}
        ]
    else:
        configs = [
            {"name": "baseline", "poison_rate": 0.03, "patch_size": 4, "epochs": 5},
            {"name": "strong_1", "poison_rate": 0.05, "patch_size": 6, "epochs": 8},
            {"name": "strong_2", "poison_rate": 0.07, "patch_size": 6, "epochs": 8},
            {"name": "strong_3", "poison_rate": 0.05, "patch_size": 8, "epochs": 8}
        ]

    results = []
    target_class = 0

    for cfg in configs:
        cname = cfg["name"]
        prate = cfg["poison_rate"]
        psize = cfg["patch_size"]
        epochs = cfg["epochs"]

        print(f"\n[Step 2] Evaluating Configuration: {cname} (Rate: {prate*100:.1f}%, Patch: {psize}x{psize}, Epochs: {epochs})...")

        # Synthesize poisoned training set
        threat = backdoor_patch(
            clean_train_imgs,
            clean_train_lbls,
            poison_rate=prate,
            target_class=target_class,
            patch_size=psize,
            seed=seed
        )

        poisoned_dataset = CIFARBatchDataset(threat.images, threat.labels, transform=transform_train)
        poisoned_loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True)

        # Triggered test set for this patch configuration
        trig_test_imgs, trig_test_lbls = create_triggered_test_set(
            clean_test_imgs, clean_test_lbls, target_class=target_class, patch_size=psize
        )
        trig_test_dataset = CIFARBatchDataset(trig_test_imgs, trig_test_lbls, transform=transform_test)
        trig_test_loader = DataLoader(trig_test_dataset, batch_size=batch_size, shuffle=False)

        # Measure baseline clean ASR for this trigger
        clean_asr = calculate_asr(clean_model, trig_test_loader, target_class, device)

        # Train victim model
        victim_model = build_cifar_resnet18(num_classes=10)
        victim_model = train_model(victim_model, poisoned_loader, device, epochs=epochs, lr=1e-3, name=cname)

        # Evaluate victim model
        victim_acc = evaluate_accuracy(victim_model, clean_test_loader, device)
        victim_asr = calculate_asr(victim_model, trig_test_loader, target_class, device)

        backdoor_effect = (victim_asr - clean_asr) * 100.0

        print(f"    Clean Acc: {victim_acc*100:.2f}% | Clean ASR: {clean_asr*100:.2f}% | Victim ASR: {victim_asr*100:.2f}% | Effect: {backdoor_effect:+.2f} pp")

        results.append({
            "configuration": cname,
            "poison_rate": prate,
            "patch_size": psize,
            "epochs": epochs,
            "clean_accuracy": round(float(victim_acc * 100), 2),
            "clean_ASR": round(float(clean_asr * 100), 2),
            "victim_ASR": round(float(victim_asr * 100), 2),
            "backdoor_effect_pp": round(float(backdoor_effect), 2)
        })

    # Summary table
    df_results = pd.DataFrame(results)
    csv_path = "experiments/image/backdoor_calibration/m6_backdoor_calibration.csv"
    json_path = "experiments/image/backdoor_calibration/m6_backdoor_calibration.json"
    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Configuration':<14} | {'Clean Acc (%)':<14} | {'Clean ASR (%)':<14} | {'Victim ASR (%)':<15} | {'Effect (pp)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['configuration']:<14} | {r['clean_accuracy']:>13.2f}% | {r['clean_ASR']:>13.2f}% | {r['victim_ASR']:>14.2f}% | {r['backdoor_effect_pp']:>+11.2f}%")
    print("=" * 80)
    print(f"[SUCCESS] Calibration complete! Results saved to:\n  - {csv_path}\n  - {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataSentinel M6 Backdoor Calibration")
    parser.add_argument("--train-limit", type=int, default=3000, help="Number of training samples")
    parser.add_argument("--test-limit", type=int, default=1000, help="Number of evaluation samples")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--quick", action="store_true", help="Quick mode with reduced epochs")
    args = parser.parse_args()

    run_calibration(
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        batch_size=args.batch_size,
        quick_mode=args.quick
    )
