"""Test M6.3 candidate signals on clean-label poisoned data."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_m6_strong_benchmark as m6
from models.image.resnet_cifar import build_cifar_resnet18
from detectors.image.attacks import clean_label_manipulation
from detectors.image.clean_label_signals import (
    differential_loss_risk,
    frequency_high_ratio_risk
)
from detectors.image.threshold_policy import choose_threshold_max_recall_under_fpr, binary_metrics
from sklearn.metrics import roc_auc_score, average_precision_score

device = torch.device("cpu")
print("Loading CIFAR...")
tr_img, tr_lbl, te_img, te_lbl = m6.load_cifar(limit_train=1200, limit_test=300, seed=42)

print("Synthesizing clean-label attack...")
threat = clean_label_manipulation(tr_img, tr_lbl, target_class=0, poison_rate=0.05, blend_strength=0.20, seed=42)

print("Loading clean and victim models...")
clean_model = build_cifar_resnet18(10)
clean_model.load_state_dict(torch.load("models/image/m6_clean_resnet18.pth", map_location="cpu"))
clean_model.eval()

victim_model = build_cifar_resnet18(10)
victim_model.load_state_dict(torch.load("models/image/m6_clean_label_victim.pth", map_location="cpu"))
victim_model.eval()

# Partitions
df_ev = pd.read_csv("experiments/image/clean_label/sample_evidence.csv")
train_mask = (df_ev["partition"] == "train").to_numpy()
val_mask = (df_ev["partition"] == "val").to_numpy()
test_mask = (df_ev["partition"] == "test").to_numpy()

train_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]
test_idx = np.where(test_mask)[0]

y = threat.is_poisoned
y_val = y[val_idx]
y_test = y[test_idx]

# Transform
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

print("\n--- 1. Evaluating Differential Loss Risk ---")
diff_risk = differential_loss_risk(
    clean_model=clean_model,
    victim_model=victim_model,
    images=threat.images,
    labels=threat.labels,
    reference_indices=train_idx,
    transform=tfm,
    device=device
)

val_auc_diff = roc_auc_score(y_val, diff_risk[val_idx])
val_pr_diff = average_precision_score(y_val, diff_risk[val_idx])
test_auc_diff = roc_auc_score(y_test, diff_risk[test_idx])
test_pr_diff = average_precision_score(y_test, diff_risk[test_idx])
th_diff, _ = choose_threshold_max_recall_under_fpr(y_val, diff_risk[val_idx], max_fpr=0.05)
m_diff = binary_metrics(y_test, diff_risk[test_idx], th_diff)

print(f"Differential Loss Risk:")
print(f"  Val AUC: {val_auc_diff:.4f} | Val PR: {val_pr_diff:.4f} | Th: {th_diff:.4f}")
print(f"  Test AUC: {test_auc_diff:.4f} | Test PR: {test_pr_diff:.4f} | Test FPR: {m_diff['fpr']:.4f} | Test Recall: {m_diff['recall']:.4f} ({int(m_diff['recall']*12)}/12)")

print("\n--- 2. Evaluating Frequency High Ratio Risk ---")
freq_risk = frequency_high_ratio_risk(
    images=threat.images,
    labels=threat.labels,
    reference_indices=train_idx,
    cutoff_radius=8.0
)

val_auc_freq = roc_auc_score(y_val, freq_risk[val_idx])
val_pr_freq = average_precision_score(y_val, freq_risk[val_idx])
test_auc_freq = roc_auc_score(y_test, freq_risk[test_idx])
test_pr_freq = average_precision_score(y_test, freq_risk[test_idx])
th_freq, _ = choose_threshold_max_recall_under_fpr(y_val, freq_risk[val_idx], max_fpr=0.05)
m_freq = binary_metrics(y_test, freq_risk[test_idx], th_freq)

print(f"Frequency High Ratio Risk:")
print(f"  Val AUC: {val_auc_freq:.4f} | Val PR: {val_pr_freq:.4f} | Th: {th_freq:.4f}")
print(f"  Test AUC: {test_auc_freq:.4f} | Test PR: {test_pr_freq:.4f} | Test FPR: {m_freq['fpr']:.4f} | Test Recall: {m_freq['recall']:.4f} ({int(m_freq['recall']*12)}/12)")

# Detailed raw frequency distribution stats
cy, cx = 16, 16
y_grid, x_grid = np.ogrid[:32, :32]
dist_from_center = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
high_mask = dist_from_center > 8.0
rgb_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

high_ratios = []
for i in range(len(threat.images)):
    gray = threat.images[i].astype(np.float32) @ rgb_weights
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(f_shift) ** 2
    r = float(power[high_mask].sum()) / (float(power.sum()) + 1e-12)
    high_ratios.append(r)
high_ratios = np.array(high_ratios)

clean_r = high_ratios[y == 0]
poison_r = high_ratios[y == 1]
t_stat, p_val = stats.ttest_ind(clean_r, poison_r)
print(f"\nRaw High-Frequency Energy Ratio Statistics:")
print(f"  Clean:   mean={clean_r.mean():.4f}, std={clean_r.std():.4f}, median={np.median(clean_r):.4f}")
print(f"  Poison:  mean={poison_r.mean():.4f}, std={poison_r.std():.4f}, median={np.median(poison_r):.4f}")
print(f"  T-statistic: {t_stat:.4f}, p-value: {p_val:.4e}")
print(f"  Cohen's d: {(clean_r.mean() - poison_r.mean()) / np.sqrt((clean_r.var() + poison_r.var()) / 2):.4f}")
