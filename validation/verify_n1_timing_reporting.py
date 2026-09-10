"""Verify explicit timing exports against every frozen held-out N1 prediction.

This replays saved models without fitting or changing historical feature tables.
"""
from pathlib import Path
import hashlib,json,sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ERPy.n1 import N1MorphologyModel
from score_erdetect_validation import canonicalize
from validation.n1_frozen_inputs import FrozenN1Inputs

KEY=['subject','stimpair','channel']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    checked_inputs=FrozenN1Inputs()
    base=ROOT/'validation/frozen_results'
    paths=[base/'n1/features/n1_features.csv.gz',base/'n1/nested/out_of_fold_predictions.csv.gz',
           base/'erdetect/record_predictions_and_labels.csv.gz']
    features=canonicalize(pd.read_csv(paths[0],low_memory=False))
    saved=pd.read_csv(paths[1],low_memory=False)
    records=pd.read_csv(paths[2],low_memory=False)
    fields=['p_reproducibility','p_energy','rms_ratio_db']
    inference=records.loc[records.configuration.eq('early_10_90ms'),KEY+fields]
    frame=saved.merge(features.drop(columns=['feature_available','negative_peak_present']),on=KEY,
                      validate='one_to_one',how='left').merge(inference,on=KEY,validate='one_to_one',how='left')
    assert len(frame)==32048 and frame.feature_available.all()
    assert frame[fields+['peak_latency_ms','feature_version']].notna().all().all()
    max_error=0.;checks=0;imputed=0;missing_timing=0;model_paths=[]
    for subject,part in frame.groupby('subject'):
        for method in ['N1_logistic_morphology','N1_logistic_hybrid']:
            path=base/'n1/nested'/f'{method}_heldout_{subject}.json';model_paths.append(path)
            model=N1MorphologyModel.load(path)
            for row in part.to_dict('records'):
                if method.endswith('hybrid'):
                    key=tuple(row[k] for k in KEY)
                    bound=checked_inputs.prepare(checked_inputs.features.loc[key],
                                                 checked_inputs.inference.loc[key])
                    result=model.predict(*bound)
                else:
                    result=model.predict(row)
                error=abs(result['score']-row[method+'_score']);max_error=max(max_error,error)
                assert error<1e-12
                assert result['detected']==row[method+'_detected']
                assert result['peak_latency_imputed']==(not row['negative_peak_present'])
                if row['negative_peak_present']:
                    assert result['measured_peak_latency_ms']==row['peak_latency_ms']
                else:
                    assert result['measured_peak_latency_ms'] is None
                    assert np.isclose(row['peak_latency_ms'],50)
                    imputed+=1;missing_timing+=1
                checks+=1
    no_peak=frame.loc[~frame.negative_peak_present]
    calls=no_peak.loc[no_peak.N1_logistic_hybrid_detected]
    out=ROOT/'validation/n1_timing_reporting';out.mkdir(exist_ok=True)
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'records':len(frame),'saved_models':len(model_paths),
        'individual_prediction_checks':checks,'maximum_saved_score_absolute_difference':max_error,'changed_calls':0,
        'imputed_model_inputs_checked':imputed,'unavailable_measured_timings_checked':missing_timing,
        'no_peak_records':len(no_peak),'no_peak_hybrid_calls':len(calls),
        'no_peak_hybrid_positive_reference_weight':float(calls.reference_positive.sum()),
        'no_peak_hybrid_ppv':float(calls.reference_positive.sum()/len(calls)),
        'interpretation':'Reporting-only additions; original feature schema, midpoint imputation, models, scores and decisions retained. Measured latency is unavailable when no interior negative peak exists.',
        'artifacts':{p.relative_to(ROOT).as_posix():sha(p) for p in [*paths,*model_paths,ROOT/'ERPy/n1.py',Path(__file__),ROOT/'tests/test_n1_features.py']}}
    (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='artifacts'},indent=2),flush=True)

if __name__=='__main__':main()
