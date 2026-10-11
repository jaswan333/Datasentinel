"""
DataSentinel - Milestone 6: Image Backdoor Benchmark & Multi-Signal Defense Runner
End-to-End Deep Learning Pipeline for CIFAR-10 Backdoor Poisoning Detection and Neutralization.
Organized into 26 sequential stages using modular detectors in detectors.image.
"""

# ============================================================
# 1. IMPORTS
# ============================================================
import os
import json
import time
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# Modular DataSentinel Image Security Suite
from detectors.image import (
    add_trigger,
    create_backdoor_dataset,
    create_triggered_test_set,
    calculate_asr,
    embedding_anomaly_scores,
    activation_clustering_scores,
    spectral_signature_scores
)


# ============================================================
# 2. CONFIGURATION
# ============================================================
TRAIN_LIMIT = 5_000       # 5,000 samples for verified CPU execution
TEST_LIMIT = 1_000        # 1,000 evaluation samples
POISON_RATE = 0.03        # 3% backdoor contamination
TARGET_CLASS = 0          # Target class: 0 (Airplane in CIFAR-10)
BATCH_SIZE = 64
EPOCHS = 3                # 3 training epochs per model
LEARNING_RATE = 1e-4
SEED = 42
NUM_CLASSES = 10
K_NEIGHBORS = 5

os.makedirs("models/image", exist_ok=True)
os.makedirs("experiments/image", exist_ok=True)
os.makedirs("data/image/poisoned", exist_ok=True)
os.makedirs("data/image/processed", exist_ok=True)


# ============================================================
# 3. RANDOM SEED
# ============================================================
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using compute device: {device}")


# ============================================================
# 4. LOAD CIFAR-10
# ============================================================
print("\n[Stage 4] Loading raw CIFAR-10...")
cifar_train_raw = datasets.CIFAR10(root="data/image/raw", train=True, download=False)
cifar_test_raw = datasets.CIFAR10(root="data/image/raw", train=False, download=False)

raw_train_images = np.array(cifar_train_raw.data)
raw_train_labels = np.array(cifar_train_raw.targets)

raw_test_images = np.array(cifar_test_raw.data)
raw_test_labels = np.array(cifar_test_raw.targets)

print(f"Raw train shape: {raw_train_images.shape}, Raw test shape: {raw_test_images.shape}")


# ============================================================
# 5. SELECT DEVELOPMENT SUBSET
# ============================================================
print(f"\n[Stage 5] Selecting subset: {TRAIN_LIMIT} train, {TEST_LIMIT} test...")
rng_subset = np.random.default_rng(SEED)

train_indices = rng_subset.choice(len(raw_train_images), size=TRAIN_LIMIT, replace=False)
clean_train_images = raw_train_images[train_indices].copy()
clean_train_labels = raw_train_labels[train_indices].copy()

test_indices = rng_subset.choice(len(raw_test_images), size=TEST_LIMIT, replace=False)
clean_test_images = raw_test_images[test_indices].copy()
clean_test_labels = raw_test_labels[test_indices].copy()


# ============================================================
# 6. CREATE BACKDOOR POISONING
# ============================================================
print("\n[Stage 6] Injecting 3% controlled yellow-patch backdoor trigger...")
poisoned_train_images, poisoned_train_labels, train_ground_truth = create_backdoor_dataset(
    clean_train_images,
    clean_train_labels,
    poison_rate=POISON_RATE,
    target_class=TARGET_CLASS,
    seed=SEED
)
n_poisoned_train = np.sum(train_ground_truth)
print(f"Injected backdoor into {n_poisoned_train} / {len(clean_train_images)} samples ({n_poisoned_train / len(clean_train_images) * 100:.1f}%)")


# ============================================================
# 7. CREATE DATASETS + DATALOADERS
# ============================================================
print("\n[Stage 7] Building PyTorch datasets and DataLoaders...")

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

class CIFARArrayDataset(Dataset):
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

clean_train_dataset = CIFARArrayDataset(clean_train_images, clean_train_labels, transform=img_transform)
poisoned_train_dataset = CIFARArrayDataset(poisoned_train_images, poisoned_train_labels, transform=img_transform)
clean_test_dataset = CIFARArrayDataset(clean_test_images, clean_test_labels, transform=img_transform)

clean_train_loader = DataLoader(clean_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
poisoned_train_loader = DataLoader(poisoned_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
clean_test_loader = DataLoader(clean_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ============================================================
# 8. CREATE RESNET18
# ============================================================
print("\n[Stage 8] Setting up ResNet18 model builder...")

def build_resnet18(num_classes: int = 10) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT
    net = resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

def train_classifier(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-4,
    name: str = "ResNet18"
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.to(device)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

        elapsed = time.time() - t0
        print(f"  [{name}] Epoch {epoch + 1}/{epochs} | Loss: {running_loss / total:.4f} | Acc: {correct / total * 100:.2f}% | Time: {elapsed:.1f}s")

    return net


# ============================================================
# 9. TRAIN CLEAN MODEL
# ============================================================
print("\n[Stage 9] Training Clean Reference Model...")
clean_model = build_resnet18(NUM_CLASSES)
clean_model = train_classifier(clean_model, clean_train_loader, device, epochs=EPOCHS, lr=LEARNING_RATE, name="Clean Model")


# ============================================================
# 10. TRAIN VICTIM MODEL
# ============================================================
print("\n[Stage 10] Training Compromised Victim Model on Poisoned Data...")
victim_model = build_resnet18(NUM_CLASSES)
victim_model = train_classifier(victim_model, poisoned_train_loader, device, epochs=EPOCHS, lr=LEARNING_RATE, name="Victim Model")


# ============================================================
# 11. CLEAN ACCURACY
# ============================================================
print("\n[Stage 11] Evaluating Clean Classification Accuracy...")

def evaluate_accuracy(net: nn.Module, loader: DataLoader, device: torch.device) -> float:
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            outputs = net(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    return correct / total

clean_model_acc = evaluate_accuracy(clean_model, clean_test_loader, device)
victim_model_acc = evaluate_accuracy(victim_model, clean_test_loader, device)
print(f"Clean Model Accuracy:  {clean_model_acc * 100:.2f}%")
print(f"Victim Model Accuracy: {victim_model_acc * 100:.2f}%")


# ============================================================
# 12. CREATE TRIGGERED TEST SET
# ============================================================
print("\n[Stage 12] Creating Triggered Test Set for Backdoor ASR evaluation...")
triggered_test_images, triggered_test_labels = create_triggered_test_set(
    clean_test_images,
    clean_test_labels,
    target_class=TARGET_CLASS
)
triggered_test_dataset = CIFARArrayDataset(triggered_test_images, triggered_test_labels, transform=img_transform)
triggered_test_loader = DataLoader(triggered_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Triggered evaluation samples (non-target): {len(triggered_test_images)}")


# ============================================================
# 13. CALCULATE ASR
# ============================================================
print("\n[Stage 13] Calculating Attack Success Rate (ASR)...")
clean_model_asr = calculate_asr(clean_model, triggered_test_loader, TARGET_CLASS, device)
victim_model_asr = calculate_asr(victim_model, triggered_test_loader, TARGET_CLASS, device)
print(f"Clean Model ASR:  {clean_model_asr * 100:.2f}%  (Natural target false positives)")
print(f"Victim Model ASR: {victim_model_asr * 100:.2f}%  (Backdoor trigger activated!)")


# ============================================================
# 14. EXTRACT RESNET EMBEDDINGS
# ============================================================
print("\n[Stage 14] Extracting 512-D Latent Embeddings from Victim Model...")

def create_feature_extractor(model: nn.Module) -> nn.Module:
    return nn.Sequential(*list(model.children())[:-1])

def extract_embeddings(
    extractor: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    extractor.eval()
    embeddings = []
    labels_list = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feats = extractor(imgs)
            feats = feats.flatten(start_dim=1)
            embeddings.append(feats.cpu().numpy())
            labels_list.append(lbls.numpy())
    return np.vstack(embeddings), np.concatenate(labels_list)

feature_extractor = create_feature_extractor(victim_model).to(device)
eval_train_loader = DataLoader(poisoned_train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
train_embeddings, train_labels = extract_embeddings(feature_extractor, eval_train_loader, device)
print(f"Extracted training embeddings: shape {train_embeddings.shape}")


# ============================================================
# 15. EMBEDDING ANOMALY
# ============================================================
print("\n[Stage 15] Computing Detector 1: KNN Embedding Anomaly Scores...")
all_indices = np.arange(len(train_embeddings))
embedding_scores = embedding_anomaly_scores(train_embeddings, poisoned_train_labels, all_indices, k=K_NEIGHBORS)


# ============================================================
# 16. ACTIVATION CLUSTERING
# ============================================================
print("\n[Stage 16] Computing Detector 2: Activation Clustering Scores...")
activation_scores = activation_clustering_scores(train_embeddings, poisoned_train_labels, all_indices)


# ============================================================
# 17. SPECTRAL SIGNATURE
# ============================================================
print("\n[Stage 17] Computing Detector 3: Spectral Signature Scores...")
spectral_scores = spectral_signature_scores(train_embeddings, poisoned_train_labels, all_indices)


# ============================================================
# 18. EVIDENCE TABLE
# ============================================================
print("\n[Stage 18] Constructing Sample-Level Multi-Signal Evidence Table...")

evidence = pd.DataFrame({
    "sample_id": np.arange(len(train_embeddings)),
    "label": poisoned_train_labels,
    "is_poisoned": train_ground_truth,
    "embedding_anomaly": np.round(embedding_scores, 4),
    "activation_cluster": np.round(activation_scores, 4),
    "spectral_signature": np.round(spectral_scores, 4)
})

print(evidence.head())


# ============================================================
# 19. 60/20/20 DETECTOR SPLIT
# ============================================================
print("\n[Stage 19] Splitting into 60% Train / 20% Val / 20% Unseen Test...")
indices = np.arange(len(evidence))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.40,
    stratify=train_ground_truth,
    random_state=SEED
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=train_ground_truth[temp_idx],
    random_state=SEED
)

print(f"Calibration splits: {len(train_idx)} Train, {len(val_idx)} Validation, {len(test_idx)} Test")


# ============================================================
# 20. LOGISTIC REGRESSION FUSION
# ============================================================
print("\n[Stage 20] Training Logistic Regression Evidence Fusion Meta-Classifier...")

FEATURES = ["embedding_anomaly", "activation_cluster", "spectral_signature"]
X_feats = evidence[FEATURES].values
y_true = train_ground_truth

fusion_model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=SEED
)

fusion_model.fit(X_feats[train_idx], y_true[train_idx])

for feat, coef in zip(FEATURES, fusion_model.coef_[0]):
    print(f"  Learned detector weight [{feat}]: {coef:+.4f}")
print(f"  Learned intercept: {fusion_model.intercept_[0]:+.4f}")


# ============================================================
# 21. THRESHOLD SELECTION
# ============================================================
print("\n[Stage 21] Calibrating Optimal Decision Threshold on Validation Split...")

val_probs = fusion_model.predict_proba(X_feats[val_idx])[:, 1]
best_threshold = 0.50
best_f1 = 0.0

for th in np.arange(0.05, 0.96, 0.01):
    preds = (val_probs >= th).astype(int)
    f1 = f1_score(y_true[val_idx], preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = float(round(th, 2))

print(f"Optimal Operating Threshold: {best_threshold:.2f} (Validation F1: {best_f1:.4f})")

# Unseen Test Evaluation
test_probs = fusion_model.predict_proba(X_feats[test_idx])[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

test_precision = precision_score(y_true[test_idx], test_preds, zero_division=0)
test_recall = recall_score(y_true[test_idx], test_preds, zero_division=0)
test_f1 = f1_score(y_true[test_idx], test_preds, zero_division=0)
test_roc_auc = roc_auc_score(y_true[test_idx], test_probs)
test_pr_auc = average_precision_score(y_true[test_idx], test_probs)

print(f"\nUnbiased Test Metrics:")
print(f"  Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")
print(f"  ROC-AUC:   {test_roc_auc:.4f} | PR-AUC: {test_pr_auc:.4f}")


# ============================================================
# 22. IMAGE RISK
# ============================================================
print("\n[Stage 22] Computing Sample-Level Risk Scores and Tiers...")

all_risk_probs = fusion_model.predict_proba(X_feats)[:, 1]
evidence["risk_probability"] = np.round(all_risk_probs, 4)

def classify_risk_tier(p: float) -> str:
    if p >= 0.55:
        return "HIGH"
    elif p >= 0.35:
        return "MEDIUM"
    return "LOW"

evidence["risk_level"] = [classify_risk_tier(p) for p in all_risk_probs]
print("Risk Tier Distribution:")
print(evidence["risk_level"].value_counts())


# ============================================================
# 23. QUARANTINE
# ============================================================
print("\n[Stage 23] Isolating High-Suspicion Samples into Quarantine...")

evidence["quarantine"] = (evidence["risk_probability"] >= best_threshold)
quarantine_pred = evidence["quarantine"].astype(int)

quar_prec = precision_score(train_ground_truth, quarantine_pred, zero_division=0)
quar_rec = recall_score(train_ground_truth, quarantine_pred, zero_division=0)
quar_f1 = f1_score(train_ground_truth, quarantine_pred, zero_division=0)
n_quarantined = int(evidence["quarantine"].sum())

print(f"Quarantined Records:  {n_quarantined} / {len(evidence)}")
print(f"Quarantine Precision: {quar_prec:.4f}")
print(f"Quarantine Recall:    {quar_rec:.4f} ({quar_rec * 100:.1f}% poisoned records purged)")
print(f"Quarantine F1-Score:  {quar_f1:.4f}")


# ============================================================
# 24. RETRAIN SCREENED MODEL
# ============================================================
print("\n[Stage 24] Retraining Screened Model from Scratch on Purified Data...")

keep_mask = ~evidence["quarantine"].values
screened_train_images = poisoned_train_images[keep_mask]
screened_train_labels = poisoned_train_labels[keep_mask]

screened_train_dataset = CIFARArrayDataset(screened_train_images, screened_train_labels, transform=img_transform)
screened_train_loader = DataLoader(screened_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

screened_model = build_resnet18(NUM_CLASSES)
screened_model = train_classifier(screened_model, screened_train_loader, device, epochs=EPOCHS, lr=LEARNING_RATE, name="Screened Model")


# ============================================================
# 25. FINAL COMPARISON
# ============================================================
print("\n[Stage 25] Evaluating Downstream Model Recovery & Backdoor Neutralization...")

screened_model_acc = evaluate_accuracy(screened_model, clean_test_loader, device)
screened_model_asr = calculate_asr(screened_model, triggered_test_loader, TARGET_CLASS, device)

asr_reduction = (victim_model_asr - screened_model_asr) * 100.0

print("\n" + "=" * 70)
print(f"{'Model Condition':<22} | {'Clean Accuracy':<16} | {'Attack Success Rate (ASR)':<25}")
print("-" * 70)
print(f"{'Clean Model':<22} | {clean_model_acc * 100:>14.2f}% | {clean_model_asr * 100:>23.2f}%")
print(f"{'Victim Model':<22} | {victim_model_acc * 100:>14.2f}% | {victim_model_asr * 100:>23.2f}%")
print(f"{'Screened Model':<22} | {screened_model_acc * 100:>14.2f}% | {screened_model_asr * 100:>23.2f}%")
print("=" * 70)
print(f"[REMEDIATION OUTCOME] Absolute ASR Neutralization: {asr_reduction:+.2f} percentage points!")


# ============================================================
# 26. SAVE CSV / JSON / MODELS
# ============================================================
print("\n[Stage 26] Persisting Artifacts & Visualizations...")

# 1. Ground truth audit table
ground_truth_df = pd.DataFrame({
    "sample_id": np.arange(len(poisoned_train_images)),
    "original_label": clean_train_labels,
    "final_label": poisoned_train_labels,
    "is_poisoned": train_ground_truth
})
ground_truth_df.to_csv("experiments/image/m6_backdoor_ground_truth.csv", index=False)

# 2. Evidence table
evidence.to_csv("experiments/image/m6_image_sample_evidence.csv", index=False)

# 3. Quarantine table
evidence[evidence["quarantine"]].to_csv("experiments/image/m6_quarantine.csv", index=False)

# 4. Detector comparison table
detector_comparison = pd.DataFrame([
    {"detector": "Embedding Anomaly", "signal": "KNN distance in learned representation", "type": "Unsupervised representation distance"},
    {"detector": "Activation Clustering", "signal": "Small/unusual representation cluster", "type": "Latent space density clustering"},
    {"detector": "Spectral Signature", "signal": "Unusual projection in representation space", "type": "Singular value decomposition direction"},
    {"detector": "DataSentinel Fusion", "signal": "Calibrated multi-signal poison risk", "type": "Supervised logistic evidence meta-model"}
])
detector_comparison.to_csv("experiments/image/m6_detector_comparison.csv", index=False)

# 5. Model weights & pipeline
torch.save(clean_model.state_dict(), "models/image/m6_clean_resnet18.pth")
torch.save(victim_model.state_dict(), "models/image/m6_victim_backdoored_resnet18.pth")
torch.save(screened_model.state_dict(), "models/image/m6_screened_resnet18.pth")
joblib.dump(fusion_model, "models/image/m6_image_fusion_model.pkl")

# 6. JSON comparison report
comparison_report = {
    "clean_model": {
        "clean_accuracy": round(float(clean_model_acc), 4),
        "asr": round(float(clean_model_asr), 4)
    },
    "victim_model": {
        "clean_accuracy": round(float(victim_model_acc), 4),
        "asr": round(float(victim_model_asr), 4)
    },
    "screened_model": {
        "clean_accuracy": round(float(screened_model_acc), 4),
        "asr": round(float(screened_model_asr), 4)
    },
    "detector_test_metrics": {
        "precision": round(float(test_precision), 4),
        "recall": round(float(test_recall), 4),
        "f1": round(float(test_f1), 4),
        "roc_auc": round(float(test_roc_auc), 4),
        "pr_auc": round(float(test_pr_auc), 4)
    },
    "quarantine_audit": {
        "precision": round(float(quar_prec), 4),
        "recall": round(float(quar_rec), 4),
        "f1": round(float(quar_f1), 4),
        "count": n_quarantined,
        "total_training": len(evidence)
    },
    "remediation": {
        "optimal_threshold": float(best_threshold),
        "asr_reduction_points": round(float(asr_reduction), 2)
    }
}

with open("experiments/image/m6_model_comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison_report, f, indent=2)

# 7. Visualization: Trigger Examples (Clean vs Poisoned)
poison_indices = np.where(train_ground_truth == 1)[0]
selected_idx = poison_indices[:5] if len(poison_indices) >= 5 else poison_indices

fig, axes = plt.subplots(2, len(selected_idx), figsize=(12, 5))
for j, idx in enumerate(selected_idx):
    axes[0, j].imshow(clean_train_images[idx])
    axes[0, j].set_title(f"Original (cls: {clean_train_labels[idx]})")
    axes[0, j].axis("off")

    axes[1, j].imshow(poisoned_train_images[idx])
    axes[1, j].set_title(f"Poisoned (cls: {poisoned_train_labels[idx]})")
    axes[1, j].axis("off")

plt.tight_layout()
plt.savefig("experiments/image/m6_trigger_examples.png", dpi=200)
plt.close()

# 8. Visualization: Summary Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
models_names = ["Clean Model", "Victim Model", "Screened Model"]
clean_accs = [clean_model_acc * 100, victim_model_acc * 100, screened_model_acc * 100]
asrs = [clean_model_asr * 100, victim_model_asr * 100, screened_model_asr * 100]

x = np.arange(len(models_names))
width = 0.35

ax.bar(x - width/2, clean_accs, width, label="Clean Accuracy (%)", color="#2563eb")
ax.bar(x + width/2, asrs, width, label="Attack Success Rate (%)", color="#dc2626")
ax.set_ylabel("Score (%)")
ax.set_title("DataSentinel Deep Learning Defense: CIFAR-10 Backdoor Neutralization")
ax.set_xticks(x)
ax.set_xticklabels(models_names)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("experiments/image/m6_results.png", dpi=200)
plt.close()

print("\n[SUCCESS] Milestone 6 pipeline completed successfully! All artifacts generated.")
