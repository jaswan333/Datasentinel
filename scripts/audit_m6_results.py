"""Flag M6 threat families that fail utility, discrimination, or false-positive guardrails."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def value(row, keys):
    for k in keys:
        if k in row and pd.notna(row[k]):
            try: return float(row[k])
            except (TypeError,ValueError): pass
    return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--summary',default='experiments/image/m6_strong_benchmark_summary.csv')
    p.add_argument('--out',default='experiments/image/m6_quality_audit.json')
    a=p.parse_args(); src=Path(a.summary); src=src if src.is_absolute() else ROOT/src
    if not src.exists(): raise SystemExit(f'Summary not found: {src}')
    df=pd.read_csv(src); results=[]
    for _,row in df.iterrows():
        r=row.to_dict(); name=str(r.get('attack','unknown'))
        auc=value(r,['detector_roc_auc','detector_test_roc_auc','test_roc_auc','roc_auc'])
        pr=value(r,['detector_pr_auc','detector_test_pr_auc','test_pr_auc','pr_auc'])
        fpr=value(r,['detector_false_positive_rate','detector_fpr','detector_test_fpr','test_fpr','fpr'])
        recall=value(r,['detector_recall','detector_test_recall','quarantine_recall','test_recall','recall'])
        precision=value(r,['detector_precision','detector_test_precision','test_precision','precision'])
        va=value(r,['victim_clean_accuracy','victim_acc']); sa=value(r,['screened_clean_accuracy','screened_acc'])
        delta=value(r,['clean_accuracy_delta_pp'])
        if delta is None and va is not None and sa is not None:
            delta = (sa - va) if (sa > 1.0 or va > 1.0) else 100 * (sa - va)
        vasr=value(r,['victim_asr']); sasr=value(r,['screened_asr'])
        asr_drop=value(r,['asr_reduction_pp'])
        if asr_drop is None and vasr is not None and sasr is not None:
            asr_drop = (vasr - sasr) if (vasr > 1.0 or sasr > 1.0) else 100 * (vasr - sasr)
        warnings=[]
        if auc is not None and auc<0.60: warnings.append('ROC-AUC < 0.60: weak ranking; threshold changes alone cannot fix this.')
        if pr is not None and pr<0.10: warnings.append('PR-AUC < 0.10: compare with actual poison prevalence.')
        # test_fpr and detector_false_positive_rate in m6 summaries are percentage points (0-100)
        fpr_ratio = (fpr / 100.0) if fpr is not None else None
        if fpr_ratio is not None and fpr_ratio>0.10: warnings.append('Test FPR > 10%: inspect signal distributions and calibration.')
        if delta is not None and delta < -5: warnings.append('Screened clean accuracy dropped by >5 percentage points.')
        if asr_drop is not None and asr_drop < 20: warnings.append('Backdoor ASR reduction <20 percentage points.')
        if not warnings: warnings=['No configured warning fired; this is not proof of production readiness.']
        results.append({'attack':name,'test_roc_auc':auc,'test_pr_auc':pr,'test_precision':precision,'test_recall':recall,'test_fpr':fpr,'victim_clean_accuracy':va,'screened_clean_accuracy':sa,'clean_accuracy_delta_pp':delta,'victim_asr':vasr,'screened_asr':sasr,'asr_reduction_pp':asr_drop,'warnings':warnings})
    out=Path(a.out); out=out if out.is_absolute() else ROOT/out; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'summary_source':str(src),'guardrails_are_project_diagnostics_not_universal_standards':True,'results':results},indent=2),encoding='utf-8')
    for r in results:
        print(f"\n{r['attack']}: ROC-AUC={r['test_roc_auc']}, PR-AUC={r['test_pr_auc']}, FPR={r['test_fpr']}, clean-accuracy delta={r['clean_accuracy_delta_pp']}, ASR reduction={r['asr_reduction_pp']}")
        for w in r['warnings']: print(' -',w)
    print('\nSaved:',out)
if __name__=='__main__': main()
