#!/usr/bin/env python3
"""Patient-separated development of negative-N1 morphology classifiers.

All outer-fold predictions are from participants excluded from model fitting,
feature standardization and operating-threshold selection. This is post hoc
development cross-validation, not a new independent external validation.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import datetime
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ERPy.n1 import (N1MorphologyModel, MORPHOLOGY_FEATURES, INFERENCE_FEATURES,
                     n1_feature_vector)
from score_erdetect_validation import metrics_from_counts, canonicalize, require_unique

KEY = ['subject','stimpair','channel']
SEED = 20260908
REGULARIZATION_C = 1.0
RULE_GRID = [dict(amplitude_z=z,prominence_z=p,min_width_ms=w,min_trial_fraction=t,
                  amplitude_uv=a,require_iut=i)
             for z,p,w,t,a,i in itertools.product([2.,3.4,5.,7.],[0.,1.,2.],[0.,4.],
                                                [0.,.8],[0.,30.,60.,100.],[False,True])]


def sha(path):
    # Stream files on all supported Python versions (file_digest needs 3.11).
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean_json(x) for x in value]
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(clean_json(value),indent=2,allow_nan=False)+'\n')


def confusion(y,calls):
    y=np.asarray(y,float);calls=np.asarray(calls,bool)
    return np.array([np.sum(y*calls),np.sum(y*~calls),np.sum((1-y)*~calls),
                     np.sum((1-y)*calls),len(y),len(y)],float)


def choose_candidate(y,calls,archived):
    """Maximize PPV subject to comparator sensitivity and specificity on training.

    If joint constraints are infeasible, preserve sensitivity first and record
    the specificity failure. If sensitivity itself is unreachable, maximize
    sensitivity then PPV. Remaining ties prefer specificity then sensitivity,
    then the earliest declared candidate. No test outcomes are accepted here.
    """
    y=np.asarray(y,float);calls=np.asarray(calls,bool)
    if calls.ndim==1:calls=calls[:,None]
    tp=y@calls;fp=(1-y)@calls
    with np.errstate(invalid='ignore',divide='ignore'):
        sensitivity=tp/y.sum();specificity=1-fp/(1-y).sum();ppv=tp/(tp+fp)
    base=metrics_from_counts(confusion(y,archived))
    sens_ok=sensitivity>=base['sensitivity']-1e-12
    joint=sens_ok & (specificity>=base['specificity']-1e-12)
    if joint.any():eligible=joint;policy='both_constraints_met'
    elif sens_ok.any():eligible=sens_ok;policy='specificity_constraint_infeasible'
    else:eligible=sensitivity>=np.nanmax(sensitivity)-1e-12;policy='sensitivity_constraint_infeasible'
    candidates=np.flatnonzero(eligible)
    best=max(candidates,key=lambda i:(np.nan_to_num(ppv[i],nan=-1),specificity[i],sensitivity[i],-int(i)))
    return int(best),dict(policy=policy,training_sensitivity=float(sensitivity[best]),
                          training_specificity=float(specificity[best]),training_ppv=float(ppv[best]),
                          comparator_sensitivity=float(base['sensitivity']),
                          comparator_specificity=float(base['specificity']),comparator_ppv=float(base['ppv']))


def choose_threshold(y,score,archived):
    # Cumulative weighted counts avoid a rows × unique-threshold matrix.
    y=np.asarray(y,float);score=np.asarray(score,float)
    available=np.isfinite(score)
    order=np.flatnonzero(available)[np.argsort(-score[available],kind='stable')]
    if not len(order):return 1.,{'policy':'no_available_training_predictions'}
    sorted_score=score[order]
    ends=np.r_[np.flatnonzero(sorted_score[1:]!=sorted_score[:-1]),len(order)-1]
    tp=np.cumsum(y[order])[ends];fp=np.cumsum(1-y[order])[ends]
    sens=tp/y.sum();spec=1-fp/(1-y).sum();ppv=tp/(tp+fp)
    base=metrics_from_counts(confusion(y,archived))
    sens_ok=sens>=base['sensitivity']-1e-12
    joint=sens_ok & (spec>=base['specificity']-1e-12)
    if joint.any():eligible=joint;policy='both_constraints_met'
    elif sens_ok.any():eligible=sens_ok;policy='specificity_constraint_infeasible'
    else:eligible=sens>=sens.max()-1e-12;policy='sensitivity_constraint_infeasible'
    best=max(np.flatnonzero(eligible),key=lambda i:(ppv[i],spec[i],sens[i],-int(i)))
    return float(sorted_score[ends[best]]),dict(policy=policy,
        training_sensitivity=float(sens[best]),training_specificity=float(spec[best]),training_ppv=float(ppv[best]),
        comparator_sensitivity=float(base['sensitivity']),comparator_specificity=float(base['specificity']),
        comparator_ppv=float(base['ppv']))


def build_design(frame,hybrid):
    n=len(MORPHOLOGY_FEATURES)+(len(INFERENCE_FEATURES) if hybrid else 0)
    x=np.zeros((len(frame),n),float);available=frame.feature_available.fillna(False).to_numpy(bool)
    for i,row in enumerate(frame.to_dict('records')):
        if available[i]:x[i]=n1_feature_vector(row,row if hybrid else None)
    return x,available


def fit_model(x,y,available,train,hybrid,provenance):
    indices=np.flatnonzero(train & available)
    if not len(indices):raise ValueError('No available training observations')
    xx=x[indices];yy=y[indices]
    scaler=StandardScaler().fit(xx)
    xx=scaler.transform(xx)
    # Two fractional observations implement cross entropy for soft rater votes;
    # their weights sum to one for each record, irrespective of rater count.
    duplicated=np.concatenate([xx,xx]);labels=np.r_[np.zeros(len(xx)),np.ones(len(xx))]
    weights=np.r_[1-yy,yy]
    keep=weights>0
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model=LogisticRegression(C=REGULARIZATION_C,solver='lbfgs',max_iter=1500,tol=1e-8,
                                 random_state=SEED).fit(duplicated[keep],labels[keep],sample_weight=weights[keep])
    return N1MorphologyModel(feature_names=MORPHOLOGY_FEATURES+(INFERENCE_FEATURES if hybrid else ()),
          mean=tuple(scaler.mean_),scale=tuple(scaler.scale_),coefficients=tuple(model.coef_[0]),
          intercept=float(model.intercept_[0]),threshold=.5,provenance=provenance)


def predict_matrix(model,x,available):
    z=((x-np.asarray(model.mean))/np.asarray(model.scale))@np.asarray(model.coefficients)+model.intercept
    score=np.exp(-np.logaddexp(0,-z));score[~available]=np.nan
    return score


def rule_calls(frame,rule):
    call=(frame.feature_available.fillna(False) & frame.negative_peak_present.fillna(False)
          & frame.negative_peak_z.ge(rule['amplitude_z'])
          & frame.negative_prominence_z.ge(rule['prominence_z'])
          & frame.peak_width_ms.ge(rule['min_width_ms'])
          & frame.trial_negative_fraction.ge(rule['min_trial_fraction'])
          & frame.negative_peak_uv.ge(rule['amplitude_uv'])).to_numpy(bool)
    if rule['require_iut']:call &= frame.detected.to_numpy(bool)
    return call


def join_features(records,features):
    """Use the same undirected stimulation-pair identity as frozen scoring."""
    features=canonicalize(features)
    require_unique(features,KEY,'N1 feature records')
    return records.merge(features,on=KEY,how='left',validate='one_to_one',suffixes=('','_feature')).reset_index(drop=True)


def evaluate_predictions(frame,methods,output):
    y=frame.reference_positive.to_numpy(float)
    subjects=sorted(frame.subject.unique());site={s:frame.loc[frame.subject.eq(s),'site'].iloc[0] for s in subjects}
    rng=np.random.default_rng(SEED);draws=np.zeros((2000,len(subjects)),int)
    for label in sorted(set(site.values())):
        ids=[i for i,s in enumerate(subjects) if site[s]==label]
        sampled=rng.choice(ids,size=(2000,len(ids)),replace=True)
        for b,row in enumerate(sampled):draws[b]=np.bincount(row,minlength=len(subjects))+draws[b]
    rows=[];vectors={}
    for name,calls in methods.items():
        counts=np.array([confusion(y[frame.subject.eq(s)],np.asarray(calls)[frame.subject.eq(s)]) for s in subjects])
        vectors[name]=counts;point=metrics_from_counts(counts.sum(axis=0));boot=metrics_from_counts(draws@counts)
        for metric in ['sensitivity','specificity','ppv','npv','balanced_accuracy','kappa']:
            lo,hi=np.nanquantile(boot[metric],[.025,.975])
            rows.append(dict(method=name,metric=metric,estimate=point[metric],low=lo,high=hi,
                             n_records=len(frame),n_subjects=len(subjects),calls=int(np.sum(calls))))
    differences=[]
    for name,counts in vectors.items():
        for reference in ['original_ERPy','archived_ER_detect']:
            if name==reference:continue
            a=metrics_from_counts(counts.sum(axis=0));b=metrics_from_counts(vectors[reference].sum(axis=0))
            aa=metrics_from_counts(draws@counts);bb=metrics_from_counts(draws@vectors[reference])
            for metric in ['sensitivity','specificity','ppv']:
                lo,hi=np.nanquantile(aa[metric]-bb[metric],[.025,.975])
                differences.append(dict(method=name,reference=reference,metric=metric,
                                        difference=a[metric]-b[metric],low=lo,high=hi))
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(differences).to_csv(output/'paired_differences.csv',index=False)
    write_json(output/'subject_confusion_counts.json',dict(subjects=subjects,site=site,methods=vectors))
    return pd.DataFrame(rows)


def curve_and_calibration(y,score,name,output):
    """Precision–recall and calibration against fractional expert votes."""
    y=np.asarray(y,float);score=np.asarray(score,float);available=np.isfinite(score)
    order=np.flatnonzero(available)[np.argsort(-score[available],kind='stable')]
    s=score[order];ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.cumsum(y[order])[ends];fp=np.cumsum(1-y[order])[ends]
    recall=tp/y.sum();precision=tp/(tp+fp)
    curve=pd.DataFrame(dict(threshold=s[ends],recall=recall,precision=precision,
                            specificity=1-fp/(1-y).sum()))
    curve.to_csv(output/f'{name}_precision_recall.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    summary=dict(method=name,average_precision=float(np.sum(np.diff(np.r_[0,recall])*precision)),
                 available=int(available.sum()),total=len(y),positive_weight=float(y.sum()))
    if name.startswith('N1_logistic'):
        bins=[]
        for lo in np.arange(0,1,.1):
            selected=available & (score>=lo) & (score<(lo+.1) if lo<.9 else score<=1)
            bins.append(dict(lower=lo,upper=lo+.1,records=int(selected.sum()),
                       mean_score=float(score[selected].mean()) if selected.any() else np.nan,
                       reference_positive_fraction=float(y[selected].mean()) if selected.any() else np.nan))
        pd.DataFrame(bins).to_csv(output/f'{name}_calibration.csv',index=False)
        # Expected squared error for a uniformly selected rater's binary vote.
        summary['rater_weighted_brier']=float(np.mean(y[available]*(1-score[available])**2+(1-y[available])*score[available]**2))
    return summary


def execute(args):
    if args.output.exists():raise ValueError('Use a new output directory; do not overwrite development outcomes')
    args.output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[1]
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'status':'post_hoc_development_not_independent_external_validation',
      'outer_folds':'leave_one_participant_out','inner_folds':'leave_one_training_participant_out',
      'fixed_regularization_C':REGULARIZATION_C,'rule_grid':RULE_GRID,'seed':SEED,
      'primary_frame':'original paired available-case; new feature unavailability yields no call',
      'threshold_objective':'maximize PPV meeting archived training sensitivity and specificity; declared fallback',
      'feature_manifest_sha256':sha(args.features/'feature_manifest.json'),
      'inputs':{'record_predictions_and_labels':sha(args.records),'n1_features':sha(args.features/'n1_features.csv.gz')},
      'code_sha256':{str(p.relative_to(root)):sha(p) for p in [Path(__file__),root/'ERPy/n1.py',root/'validation/N1_OPTIMIZATION_PROTOCOL.md']},
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}}
    write_json(args.output/'run_protocol.json',protocol)
    feature_manifest=json.loads((args.features/'feature_manifest.json').read_text())
    if feature_manifest['output_sha256']!=protocol['inputs']['n1_features']:
        raise ValueError('Feature output does not match its extraction manifest')
    if feature_manifest['code_sha256']['ERPy/n1.py']!=sha(root/'ERPy/n1.py'):
        raise ValueError('Feature implementation changed; regenerate features under the frozen version')
    records=pd.read_csv(args.records,low_memory=False)
    mask=(records.configuration.eq('early_10_90ms') & records.scoring_eligible & records.evaluable
          & records.archived_record_present & records.archived_pair_comparable
          & records.reference_positive.notna() & ~records.development_overlap)
    frame=records.loc[mask].copy()
    features=pd.read_csv(args.features/'n1_features.csv.gz',low_memory=False)
    frame=join_features(frame,features)
    if len(frame)!=32048 or frame.subject.nunique()!=13:raise ValueError('Primary frozen comparison frame changed')
    present=frame.epoch_sha256_feature.notna()
    if not frame.loc[present,'epoch_sha256'].eq(frame.loc[present,'epoch_sha256_feature']).all():
        raise ValueError('Feature epochs differ from frozen inference epochs')
    y=frame.reference_positive.to_numpy(float);subjects=sorted(frame.subject.unique())
    archived=frame.archived_detected.to_numpy(bool)
    methods={'original_ERPy':frame.detected.to_numpy(bool),'archived_ER_detect':archived}
    fixed=dict(amplitude_z=3.4,prominence_z=0,min_width_ms=0,min_trial_fraction=0,amplitude_uv=0,require_iut=False)
    methods['N1_fixed_amplitude']=rule_calls(frame,fixed)
    rules=np.column_stack([rule_calls(frame,r) for r in RULE_GRID])
    selections=[];final_models={};curves=[]
    predictions=frame[KEY+['site','reference_positive','feature_available','negative_peak_present']].copy()
    site_methods={site:{name:np.asarray(call)[frame.site.eq(site)] for name,call in methods.items()}
                  for site in sorted(frame.site.unique())}
    for require in [False,True]:
        name='N1_rule_with_IUT' if require else 'N1_rule'
        options=np.array([i for i,r in enumerate(RULE_GRID) if r['require_iut']==require])
        calls=np.zeros(len(frame),bool)
        for subject in subjects:
            test=frame.subject.eq(subject).to_numpy();train=~test
            idx,detail=choose_candidate(y[train],rules[train][:,options],archived[train]);chosen=options[idx]
            calls[test]=rules[test,chosen]
            selections.append(dict(method=name,held_out=subject,rule=RULE_GRID[chosen],**detail))
        methods[name]=calls
        i,detail=choose_candidate(y,rules[:,options],archived)
        final_models[name]=dict(rule=RULE_GRID[options[i]],selection=detail,training_subjects=subjects)
        for site in sorted(frame.site.unique()):
            test=frame.site.eq(site).to_numpy();train=~test
            idx,detail=choose_candidate(y[train],rules[train][:,options],archived[train]);chosen=options[idx]
            site_methods[site][name]=rules[test,chosen]
            selections.append(dict(method=name,transport_test_site=site,rule=RULE_GRID[chosen],**detail))
    for hybrid in [False,True]:
        name='N1_logistic_hybrid' if hybrid else 'N1_logistic_morphology'
        x,available=build_design(frame,hybrid);oof=np.full(len(frame),np.nan);calls=np.zeros(len(frame),bool)
        fit_cache={};thresholds=np.full(len(frame),np.nan)
        def fit_excluding(excluded):
            key=tuple(sorted(excluded))
            if key not in fit_cache:
                train=~frame.subject.isin(key).to_numpy()
                model=fit_model(x,y,available,train,hybrid,dict(training_subjects=[s for s in subjects if s not in key],
                     excluded_subjects=list(key),status=protocol['status'],regularization_C=REGULARIZATION_C,
                     feature_manifest_sha256=protocol['feature_manifest_sha256'],
                     run_protocol_sha256=sha(args.output/'run_protocol.json')))
                fit_cache[key]=(model,predict_matrix(model,x,available))
            return fit_cache[key]
        for count,subject in enumerate(subjects,1):
            test=frame.subject.eq(subject).to_numpy();train=~test
            inner=np.full(len(frame),np.nan)
            for inner_subject in subjects:
                if inner_subject==subject:continue
                _,score=fit_excluding([subject,inner_subject])
                inner_test=frame.subject.eq(inner_subject).to_numpy();inner[inner_test]=score[inner_test]
            threshold,detail=choose_threshold(y[train],inner[train],archived[train])
            model,score=fit_excluding([subject]);oof[test]=score[test]
            calls[test]=np.isfinite(score[test]) & (score[test]>=threshold);thresholds[test]=threshold
            model_data=asdict(model);model_data['threshold']=threshold
            write_json(args.output/f'{name}_heldout_{subject}.json',model_data)
            selections.append(dict(method=name,held_out=subject,threshold=threshold,**detail))
            print(f'{name}: held-out participant {count}/{len(subjects)} complete',flush=True)
        methods[name]=calls;predictions[name+'_score']=oof;predictions[name+'_threshold']=thresholds
        final_threshold,detail=choose_threshold(y,oof,archived)
        final_model,_=fit_excluding([]);model_data=asdict(final_model);model_data['threshold']=final_threshold
        model_data['provenance']['threshold_selection']='all-participant cross-validated training scores; not the reported nested outer decisions'
        write_json(args.output/f'{name}_development_model.json',model_data)
        final_models[name]=dict(threshold=final_threshold,selection=detail,training_subjects=subjects)
        curves.append(curve_and_calibration(y,oof,name,args.output))
        for site in sorted(frame.site.unique()):
            test=frame.site.eq(site).to_numpy();train=~test
            excluded=sorted(frame.loc[test,'subject'].unique());inner=np.full(len(frame),np.nan)
            for subject in sorted(frame.loc[train,'subject'].unique()):
                _,score=fit_excluding(excluded+[subject]);mask=frame.subject.eq(subject).to_numpy()
                inner[mask]=score[mask]
            threshold,detail=choose_threshold(y[train],inner[train],archived[train])
            model,score=fit_excluding(excluded)
            site_methods[site][name]=np.isfinite(score[test]) & (score[test]>=threshold)
            selections.append(dict(method=name,transport_test_site=site,threshold=threshold,**detail))
            model_data=asdict(model);model_data['threshold']=threshold
            write_json(args.output/f'{name}_transport_test_{site}.json',model_data)
    for name,call in methods.items():predictions[name+'_detected']=call
    predictions.to_csv(args.output/'out_of_fold_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    metrics=evaluate_predictions(frame,methods,args.output)
    common=frame.feature_available.fillna(False).to_numpy(bool)
    secondary=args.output/'common_feature_intersection';secondary.mkdir()
    evaluate_predictions(frame.loc[common],{name:np.asarray(call)[common] for name,call in methods.items()},secondary)
    for site in sorted(frame.site.unique()):
        target=args.output/f'transport_test_{site}';target.mkdir()
        evaluate_predictions(frame.loc[frame.site.eq(site)],site_methods[site],target)
        p=frame.loc[frame.site.eq(site),KEY+['reference_positive']].copy()
        for name,call in site_methods[site].items():p[name]=call
        p.to_csv(target/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    curves.append(curve_and_calibration(y,-np.log10(np.maximum(frame.q_joint.to_numpy(float),1e-12)),
                                        'original_ERPy',args.output))
    write_json(args.output/'curve_summaries.json',curves)
    write_json(args.output/'fold_selections.json',selections)
    write_json(args.output/'development_model_selection.json',final_models)
    frame.groupby(['site','subject'],dropna=False).agg(records=('channel','size'),
        features_available=('feature_available','sum'),positive_weight=('reference_positive','sum')).to_csv(args.output/'coverage.csv')
    if protocol['inputs']!={'record_predictions_and_labels':sha(args.records),'n1_features':sha(args.features/'n1_features.csv.gz')}:
        raise ValueError('Input changed during development execution')
    for path,expected in protocol['code_sha256'].items():
        if sha(root/path)!=expected:raise ValueError('Implementation or protocol changed during execution')
    write_json(args.output/'completion.json',dict(status='complete',n_records=len(frame),n_subjects=len(subjects),
             feature_unavailable=int((~frame.feature_available.fillna(False)).sum()),
             metrics_sha256=sha(args.output/'metrics.csv'),predictions_sha256=sha(args.output/'out_of_fold_predictions.csv.gz')))
    print(metrics.pivot(index='method',columns='metric',values='estimate').round(5).to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,required=True);p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    execute(p.parse_args())
