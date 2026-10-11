# DataSentinel M6.1 — Validation and False-Positive Hardening

## Why this phase exists

The current M6 report has strong results on the calibrated patch backdoor and duplicate injection, but the clean-label-centroid case is not reliable yet: the reported test ROC-AUC is 0.519, FPR is 89.04%, and screened clean accuracy drops from 35.00% to 11.33%. High recall does not make that a successful defense when many clean samples are quarantined.

M6.1 focuses on **ranking quality, false-positive control, and retained model utility**—not adding a large number of unrelated detectors.

## Two targeted changes

### 1. Additional signal: `cross_class_prototype_affinity_risk`

For each label, the code estimates class prototypes from the **fusion-training partition only**. For each image, it compares the distance to the assigned-class prototype with the closest non-assigned class prototype. A sample that is unusually closer to another class prototype receives a higher risk percentile within its assigned class.

Why it may help: your simulated clean-label attack keeps the original label while blending an image toward a target-class prototype. This signal tests whether that movement is visible in pixel space. It is a heuristic and may fail on semantically complex images; use held-out results to decide whether to retain it.

### 2. Threshold policy: maximum recall under an FPR ceiling

The old policy selected the validation threshold that maximized F1 without an explicit limit on false positives. The new policy chooses the threshold that **maximizes validation recall subject to validation FPR ≤ 5%**. Ties are broken by precision, F1, then higher threshold.

This does not guarantee 5% FPR on the test set. Test FPR must be computed and reported separately. If test ROC-AUC remains near 0.5, a threshold change alone cannot make the scores discriminate poisoned and clean samples.

## Files in this overlay

Copy these files into your existing DataSentinel project; do not overwrite the existing M6 runner:

- `detectors/image/clean_label_signals.py`
- `detectors/image/threshold_policy.py`
- `scripts/run_m6_clean_label_hardening.py`
- `scripts/analyze_clean_label_signals.py`
- `scripts/audit_m6_results.py`

The focused runner imports your current `scripts/run_m6_strong_benchmark.py` and applies the new signal/threshold in memory for this run. The original runner source is not edited.

## Exact execution order

Run from the **DataSentinel project root**:

```bash
python scripts/run_m6_clean_label_hardening.py --epochs 5 --train-limit 1200 --test-limit 300 --max-validation-fpr 0.05
```

The 1,200/300 size is selected to match the scale implied by your current six-family report (60 poisoned samples for a 5% attack). This is for apples-to-apples validation. After you inspect the result, repeat with a larger subset, but rerun the baseline at that same subset size for a fair comparison.

Then inspect which features separate clean and poisoned samples on the held-out test partition:

```bash
python scripts/analyze_clean_label_signals.py
```

Audit the six-family summary:

```bash
python scripts/audit_m6_results.py
```

Outputs:

- `experiments/image/clean_label_hardening/baseline_clean_label_summary.json`
- `experiments/image/clean_label_hardening/m6_clean_label_hardening_comparison.csv`
- `experiments/image/clean_label_hardening/m6_clean_label_hardening_comparison.json`
- `experiments/image/clean_label_hardening/clean_label_signal_diagnostics.csv`
- `experiments/image/m6_quality_audit.json`

## Decision rules

Do not decide the change worked based on recall alone. Compare all of these on the held-out test partition:

1. ROC-AUC and PR-AUC (ranking quality).
2. Precision, recall, F1 and FPR at the validation-selected threshold.
3. Quarantine rate and number of legitimate samples quarantined.
4. Victim versus screened clean accuracy and macro-F1.
5. For backdoors, victim ASR and screened ASR.

If the new signal increases test ROC-AUC while the constrained threshold reduces FPR without destroying recall/utility, integrate it into the common signal table and rerun all six attacks. Only after that should you rerun cross-attack zero-shot evaluation with the same signal schema.

If AUC stays near 0.5 or recall falls to a very low value under the FPR cap, report clean-label manipulation as an unresolved threat. Do not tune the threshold on the test partition or claim universal defense.
