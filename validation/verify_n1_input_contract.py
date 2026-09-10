"""Replay immutable held-out N1 models through the checked input interface.

No fitting, threshold selection, source-table update or raw reprocessing occurs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ERPy.n1 import N1MorphologyModel
from validation.n1_frozen_inputs import FrozenN1Inputs, KEY


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(output):
    receipt=json.loads((output/'verification.json').read_text())
    for name, expected in receipt['artifacts'].items():
        if sha(ROOT/name)!=expected:
            raise ValueError(f'N1 input-contract artifact identity mismatch: {name}')
    if receipt['changed_calls'] or receipt['maximum_saved_score_absolute_difference']>1e-12:
        raise ValueError('Saved prediction comparison did not pass')
    return {'status':'verified','artifacts':len(receipt['artifacts']),
            'prediction_checks':receipt['individual_prediction_checks']}


def run(output):
    if output.exists():
        raise ValueError('Use a new output directory; prior receipts are preserved')
    base=ROOT/'validation/frozen_results/n1'
    frozen=FrozenN1Inputs(base)
    artifacts=[ROOT/'ERPy/n1.py',ROOT/'validation/n1_frozen_inputs.py',Path(__file__),
               ROOT/'tests/test_n1_input_contract.py',base/'artifact_manifest.json',
               base/'features/feature_manifest.json',base/'features/n1_features.csv.gz',
               base.parent/'erdetect/record_predictions_and_labels.csv.gz',
               base.parent/'erdetect/cache_safeguard_verification.json']
    frames={'participant_held_out':pd.read_csv(base/'nested/out_of_fold_predictions.csv.gz')}
    for site in ['Mayo','UMCU']:
        frames['transport_test_'+site]=pd.read_csv(base/f'nested/transport_test_{site}/predictions.csv.gz')
    prepared={};checks=0;max_error=0.;models={};by_frame={}
    for label,frame in frames.items():
        path=base/('nested/out_of_fold_predictions.csv.gz' if label=='participant_held_out' else f'nested/{label}/predictions.csv.gz')
        artifacts.append(path);frame_checks=0
        for ix,row in enumerate(frame.to_dict('records'),1):
            key=tuple(row[k] for k in KEY)
            if key not in prepared:
                prepared[key]=frozen.prepare(frozen.features.loc[key],frozen.inference.loc[key])
            feature,evidence=prepared[key]
            for method in ['N1_logistic_morphology','N1_logistic_hybrid']:
                suffix='heldout_'+row['subject'] if label=='participant_held_out' else label
                model_path=base/f'nested/{method}_{suffix}.json'
                if model_path not in models:
                    models[model_path]=N1MorphologyModel.load(model_path);artifacts.append(model_path)
                model=models[model_path]
                result=model.predict(feature,evidence if method.endswith('hybrid') else None)
                error=abs(result['score']-row[method+'_score']) if label=='participant_held_out' else 0.
                max_error=max(max_error,error)
                call=row[method+'_detected'] if label=='participant_held_out' else row[method]
                if error>1e-12 or result['detected']!=bool(call):
                    raise AssertionError(f'Saved prediction changed: {label}, {key}, {method}')
                if result['peak_latency_imputed'] != (not bool(feature['negative_peak_present'])):
                    raise AssertionError('Timing interpretation changed')
                checks+=1;frame_checks+=1
            if ix%5000==0:print(f'{label}: {ix}/{len(frame)} records verified',flush=True)
        by_frame[label]={'records':len(frame),'individual_prediction_checks':frame_checks,
                         'stored_scores_checked':frame_checks if label=='participant_held_out' else 0,
                         'stored_calls_checked':frame_checks}
    # Demonstrate the precise rejected substitution without new outcome tuning.
    key=('MAYO02','LG1-LG2','LG4');feature,evidence=prepared[key]
    plain=evidence.validated_values(feature)
    model=models[base/'nested/N1_logistic_hybrid_heldout_MAYO02.json']
    try:model.predict(feature,plain)
    except ValueError:pass
    else:raise AssertionError('Bare inference dictionary was accepted')
    records=pd.read_csv(base.parent/'erdetect/record_predictions_and_labels.csv.gz',
                        dtype={'random_seed':'string'},low_memory=False)
    broad=records.loc[(records.subject==key[0])&(records.stimpair==key[1])&
                      (records.channel==key[2])&(records.configuration=='broad_15_300ms')].iloc[0].to_dict()
    for relabel in [False,True]:
        candidate=dict(broad)
        if relabel:candidate['configuration']='early_10_90ms'
        try:frozen.prepare(frozen.features.loc[key],candidate)
        except ValueError:pass
        else:raise AssertionError('Broad values accepted, with relabel='+str(relabel))
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
             'records_with_verified_source_binding':len(prepared),'saved_models_replayed':len(models),
             'individual_prediction_checks':checks,'by_frame':by_frame,'changed_calls':0,
             'stored_scores_checked':2*len(frames['participant_held_out']),
             'stored_calls_checked':checks,
             'maximum_saved_score_absolute_difference':max_error,
             'bare_inference_dictionary_rejected':True,'broad_window_and_relabelled_broad_values_rejected':True,
             'exact_63bit_random_seed_text_preserved':True,
             'trial_provenance':'Frozen adapter verifies source epoch/record identity, complete configuration and recorded trial counts. Historical clean indices were not archived and remain None; new-array wrapper retains each method-specific mask.',
             'interpretation':'Input-contract correction only; no retraining or alteration of numerical feature transforms, saved model parameters, archived scores/calls or measured timing.',
             'artifacts':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(artifacts))}}
    output.mkdir(parents=True)
    (output/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='artifacts'},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'validation/n1_input_contract')
    p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if args.verify:print(json.dumps(verify(args.output),indent=2))
    else:run(args.output)


if __name__=='__main__':main()
