#!/usr/bin/env python3
"""Score completed fixed-rule predictions against human votes, separately."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation.predict_n1_label_free import (ROOT, KEY, PROTOCOL_SHA256,
                                            RECORDS_SHA256, canonical, sha)
from validation.score_erdetect_validation import metrics_from_counts

RATERS_SHA256='916c76fc6935913a2eb6a589797a2745092488d1ba5893da7f6bc17ef19dc8fc'
MODELS_SHA256='7fd37ffc162e1eab62ebbc83563806a2ef42f350edcc7806949b18ecbff8d56a'
METRICS=['sensitivity','specificity','ppv','npv','balanced_accuracy','kappa']


def reference_from_raters(path):
    """Reconstruct record-weighted human N1 reference; never use algorithm calls."""
    votes=pd.read_csv(path,usecols=KEY+['rater','configuration','raw_annotation'])
    if not votes.raw_annotation.dropna().isin([-1,0,1,2]).all():
        raise ValueError('Unknown human annotation code')
    votes=votes.loc[votes.configuration.eq('early_10_90ms') & votes.raw_annotation.isin([0,1,2])].copy()
    if votes.duplicated(KEY+['rater']).any():
        raise ValueError('Duplicate human vote for the same record and rater')
    summary=votes.groupby(KEY,sort=True).agg(n_raters=('rater','size'),
        positive_votes=('raw_annotation',lambda x:int(x.eq(1).sum())))
    summary['reference_positive']=summary.positive_votes/summary.n_raters
    return summary.reset_index()


def counts(y,call):
    y=np.asarray(y,float);call=np.asarray(call,bool)
    if y.shape!=call.shape or not np.isfinite(y).all() or np.any((y<0)|(y>1)):
        raise ValueError('Human reference fractions must match calls and lie in [0,1]')
    return np.array([np.sum(y*call),np.sum(y*~call),np.sum((1-y)*~call),
                     np.sum((1-y)*call),len(y),len(y)],float)


def cluster_draws(subjects,site):
    rng=np.random.default_rng(20260908)
    draws=np.zeros((2000,len(subjects)),int)
    for label in sorted(set(site.values())):
        indices=[i for i,s in enumerate(subjects) if site[s]==label]
        for b,row in enumerate(rng.choice(indices,size=(2000,len(indices)),replace=True)):
            draws[b]+=np.bincount(row,minlength=len(subjects))
    return draws


def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,np.ndarray) and value.ndim==0:return clean_json(value.item())
    if isinstance(value,(list,tuple,np.ndarray)):return [clean_json(x) for x in value]
    if isinstance(value,(bool,np.bool_)):return bool(value)
    if isinstance(value,(int,np.integer)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def execute(predictions,output,protocol):
    predictions,output,protocol=Path(predictions),Path(output),Path(protocol)
    if output.exists():raise ValueError('Use a new output directory')
    if sha(protocol)!=PROTOCOL_SHA256:raise ValueError('Fixed protocol mismatch')
    manifest_path=predictions/'prediction_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['protocol_sha256']!=PROTOCOL_SHA256 or manifest['human_reference_columns_read'] or manifest['archived_or_model_call_columns_read']:
        raise ValueError('Prediction stage did not satisfy the label-blind contract')
    for name,value in manifest['outputs'].items():
        if sha(predictions/name)!=value:raise ValueError('Completed prediction bytes changed')
    records=ROOT/'validation/frozen_results/erdetect/record_predictions_and_labels.csv.gz'
    raters=records.with_name('rater_label_joins.csv.gz')
    models=ROOT/'validation/frozen_results/n1/nested/out_of_fold_predictions.csv.gz'
    for path,expected in [(records,RECORDS_SHA256),(raters,RATERS_SHA256),(models,MODELS_SHA256)]:
        if sha(path)!=expected:raise ValueError('Frozen scoring input changed')
    reference=reference_from_raters(raters)
    original=pd.read_csv(records,usecols=KEY+['configuration','site','scoring_eligible',
        'evaluable','archived_record_present','archived_pair_comparable','development_overlap',
        'archived_detected','reference_positive','prediction_present'])
    original=canonical(original.loc[original.configuration.eq('early_10_90ms')])
    original=original.merge(reference,on=KEY,how='left',validate='one_to_one',suffixes=('_saved',''))
    if not np.allclose(original.reference_positive_saved,original.reference_positive,
                       rtol=0,atol=1e-15,equal_nan=True):
        raise ValueError('Reconstructed human-vote reference differs from frozen reference')
    calls=canonical(pd.read_csv(predictions/'contact_predictions.csv.gz',dtype={'random_seed':'string'}))
    eligible=original.loc[original.scoring_eligible & ~original.development_overlap & original.reference_positive.notna()].copy()
    available=eligible.merge(calls,on=KEY,how='left',validate='one_to_one',suffixes=('','_prediction'))
    primary=available.loc[available.evaluable & available.archived_record_present & available.archived_pair_comparable].copy()
    if len(primary)!=32048 or primary.subject.nunique()!=13 or primary.detected.isna().any():
        raise ValueError('Existing paired available-case comparison frame changed')
    saved=canonical(pd.read_csv(models,usecols=KEY+['N1_logistic_morphology_detected','N1_logistic_hybrid_detected']))
    primary=primary.merge(saved,on=KEY,validate='one_to_one')
    if len(primary)!=32048:raise ValueError('Saved supervised comparison membership changed')
    method_calls={'label_free_N1':primary.detected.to_numpy(bool),
        'archived_ER_detect':primary.archived_detected.to_numpy(bool),
        'baseline_only_N1_screen':primary.ablation_detected.to_numpy(bool),
        'N1_logistic_morphology':primary.N1_logistic_morphology_detected.to_numpy(bool),
        'N1_logistic_hybrid':primary.N1_logistic_hybrid_detected.to_numpy(bool)}
    subjects=sorted(primary.subject.unique())
    site={s:primary.loc[primary.subject.eq(s),'site'].iloc[0] for s in subjects}
    draws=cluster_draws(subjects,site)
    vectors={};estimates={};bootstraps={};metrics=[];subject_rows=[]
    y=primary.reference_positive.to_numpy(float)
    for name,call in method_calls.items():
        vectors[name]=np.array([counts(y[primary.subject.eq(s)],call[primary.subject.eq(s)]) for s in subjects])
        estimates[name]=metrics_from_counts(vectors[name].sum(axis=0))
        bootstraps[name]=metrics_from_counts(draws@vectors[name])
        for metric in METRICS:
            lo,hi=np.nanquantile(bootstraps[name][metric],[.025,.975],method='linear')
            metrics.append(dict(method=name,metric=metric,estimate=estimates[name][metric],
                low=lo,high=hi,n_records=len(primary),n_subjects=len(subjects),calls=int(call.sum())))
        for s,v in zip(subjects,vectors[name]):
            subject_rows.append(dict(method=name,subject=s,site=site[s],
                **dict(zip(['tp','fn','tn','fp','available','eligible'],v)),**metrics_from_counts(v)))
    differences=[]
    for name in method_calls:
        if name=='archived_ER_detect':continue
        for metric in ['sensitivity','specificity','ppv']:
            d=bootstraps[name][metric]-bootstraps['archived_ER_detect'][metric]
            lo,hi=np.nanquantile(d,[.025,.975],method='linear')
            differences.append(dict(method=name,reference='archived_ER_detect',metric=metric,
                difference=estimates[name][metric]-estimates['archived_ER_detect'][metric],
                low=lo,high=hi,ci_excludes_zero=bool(lo>0 or hi<0),n_records=len(primary),n_subjects=len(subjects)))
    stages=[]
    for cohort,frame in [('paired32048',primary),('eligible_human_reference',available)]:
        frame=frame.copy();frame['decision_stage']=frame.decision_stage.fillna('inference_unavailable')
        for grouping in [['decision_stage'],['subject','decision_stage']]:
            for key,g in frame.groupby(grouping,sort=True):
                if not isinstance(key,tuple):key=(key,)
                stages.append(dict(cohort=cohort,**dict(zip(grouping,key)),records=len(g),
                    human_n1_positive_weight=float(g.reference_positive.sum()),
                    human_n1_negative_weight=float((1-g.reference_positive).sum())))
    output.mkdir(parents=True)
    all_metrics=pd.DataFrame(metrics);all_differences=pd.DataFrame(differences)
    all_metrics.loc[all_metrics.method.isin(['label_free_N1','archived_ER_detect']) & all_metrics.metric.isin(['sensitivity','specificity','ppv'])].to_csv(output/'operating_points.csv',index=False)
    all_differences.loc[all_differences.method.eq('label_free_N1')].to_csv(output/'paired_differences.csv',index=False)
    all_metrics.to_csv(output/'all_candidate_metrics.csv',index=False)
    all_differences.loc[~all_differences.method.eq('label_free_N1')].to_csv(output/'secondary_paired_differences.csv',index=False)
    pd.DataFrame(subject_rows).to_csv(output/'participant_metrics.csv',index=False)
    pd.DataFrame(stages).to_csv(output/'decision_stage_counts.csv',index=False)
    primary.to_csv(output/'paired_records.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    site_rows=[]
    for name,call in method_calls.items():
        for label in sorted(set(site.values())):
            mask=primary.site.eq(label).to_numpy()
            site_rows.append(dict(method=name,site=label,n_records=int(mask.sum()),
                                  **metrics_from_counts(counts(y[mask],call[mask]))))
    pd.DataFrame(site_rows).to_csv(output/'site_descriptives.csv',index=False)
    summary=dict(status='passed',protocol_sha256=PROTOCOL_SHA256,comparison='fixed label-free primary; archived ER-detect comparator; supervised saved models secondary',
        human_reference_source='raw_annotation in rater_label_joins only; record total weight1',
        rater_input_columns=KEY+['rater','configuration','raw_annotation'],
        algorithm_reference_columns_used=[],reconstructed_reference_matches_frozen=True,
        paired_records=len(primary),participants=subjects,
        coverage=dict(eligible_human_reference_records=len(available),
            original_inference_available=int(available.evaluable.sum()),
            proposed_available=int(available.available.eq(True).sum()),
            proposed_calls=int(available.detected.eq(True).sum())),
        weighted_reference_positive=float(y.sum()),weighted_reference_negative=float((1-y).sum()),
        estimates=estimates,bootstrap=dict(seed=20260908,resamples=2000,strata='site',unit='whole participant',quantile='linear .025/.975',conditional_fixed_predictions=True),
        posthoc_development=True,threshold_selection=False,models_retrained=False,
        hypothesis_limit='p/q concern original response conjunction; not N1-truth FDR',
        input_hashes=dict(prediction_manifest=sha(manifest_path),records=RECORDS_SHA256,raters=RATERS_SHA256,saved_models_predictions=MODELS_SHA256),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'validation/predict_n1_label_free.py',ROOT/'validation/score_erdetect_validation.py']})
    (output/'summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n')
    (output/'manifest.json').write_text(json.dumps({'outputs':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},'protocol_sha256':PROTOCOL_SHA256},indent=2)+'\n')
    print(json.dumps(clean_json({'paired_records':len(primary),'estimates':estimates}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args();execute(args.predictions,args.output,args.protocol)
