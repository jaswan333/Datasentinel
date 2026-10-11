"""Rank individual clean-label evidence signals on the held-out test partition."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="experiments/image/clean_label/sample_evidence.csv")
    parser.add_argument("--out", default="experiments/image/clean_label_hardening/clean_label_signal_diagnostics.csv")
    args = parser.parse_args()
    path = Path(args.evidence)
    if not path.is_absolute(): path = ROOT / path
    if not path.exists(): raise SystemExit(f"Evidence not found: {path}")
    df = pd.read_csv(path)
    for c in ["partition", "is_poisoned"]:
        if c not in df: raise SystemExit(f"Required column missing: {c}")
    test = df[df["partition"].astype(str).str.lower().eq("test")].copy()
    if test.empty: raise SystemExit("No held-out rows with partition='test'.")
    y = test["is_poisoned"].astype(int).to_numpy()
    excluded = {"sample_id", "attack", "label", "is_poisoned", "group_id", "partition", "risk_probability", "risk_level", "quarantine", "original_label", "final_label"}
    signals = [c for c in test.columns if c not in excluded and pd.api.types.is_numeric_dtype(test[c])]
    rows=[]
    prevalence=float(y.mean())
    for col in signals:
        x=test[col].astype(float).to_numpy()
        finite=np.isfinite(x)
        if finite.sum()<2 or len(np.unique(y[finite]))<2: continue
        clean_mean=float(np.mean(x[finite][y[finite]==0])) if np.any(y[finite]==0) else None
        poison_mean=float(np.mean(x[finite][y[finite]==1])) if np.any(y[finite]==1) else None
        auc=float(roc_auc_score(y[finite],x[finite]))
        rows.append({"signal":col,"test_roc_auc":auc,"test_pr_auc":float(average_precision_score(y[finite],x[finite])),"clean_mean":clean_mean,"poisoned_mean":poison_mean,"mean_difference_poison_minus_clean":(poison_mean-clean_mean) if poison_mean is not None and clean_mean is not None else None,"poison_prevalence":prevalence,"higher_score_rankings_better":auc>=0.5})
    out=Path(args.out)
    if not out.is_absolute(): out=ROOT/out
    out.parent.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows).sort_values("test_roc_auc",ascending=False)
    result.to_csv(out,index=False)
    print(f"Held-out samples: {len(test)}; poison prevalence: {prevalence:.3f}")
    print(result.to_string(index=False))
    print("Saved:",out)

if __name__=="__main__": main()
