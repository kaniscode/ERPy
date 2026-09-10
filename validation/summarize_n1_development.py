#!/usr/bin/env python3
"""Secondary summaries of frozen patient-held-out N1 models; no refitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ERPy.n1 import N1MorphologyModel
from optimize_n1 import (KEY,SEED,build_design,confusion,join_features,predict_matrix,
                         rule_calls,sha,write_json)
from score_erdetect_validation import metrics_from_counts


def main(args):
    if args.output.exists():raise ValueError('Use a new secondary-summary directory')
    args.output.mkdir(parents=True)
    completed=json.loads((args.run/'completion.json').read_text())
    if sha(args.run/'out_of_fold_predictions.csv.gz')!=completed['predictions_sha256']:
        raise ValueError('Cross-validation predictions changed')
    records=pd.read_csv(args.records,low_memory=False)
    mask=(records.configuration.eq('early_10_90ms') & records.scoring_eligible
          & records.reference_positive.notna() & ~records.development_overlap)
    frame=join_features(records.loc[mask].copy(),pd.read_csv(args.features/'n1_features.csv.gz',low_memory=False))
    if len(frame)!=33694 or int(frame.evaluable.sum())!=32121:raise ValueError('Source-frame denominator changed')
    y=frame.reference_positive.to_numpy(float)
    # Secondary comparison uses the original detector-evaluable frame and a
    # separately labeled policy counting unavailable processing as no call.
    available=frame.evaluable.to_numpy(bool) & frame.feature_available.fillna(False).to_numpy(bool)
    methods={'original_ERPy':frame.detected.to_numpy(bool)}
    choices=pd.DataFrame(json.loads((args.run/'fold_selections.json').read_text()))
    for name in ['N1_rule','N1_rule_with_IUT']:
        call=np.zeros(len(frame),bool)
        for subject in sorted(frame.subject.unique()):
            chosen=choices[choices.method.eq(name)&choices.held_out.eq(subject)].iloc[0]['rule']
            selected=frame.subject.eq(subject).to_numpy();call[selected]=rule_calls(frame.loc[selected],chosen)
        methods[name]=call & available
    for hybrid in [False,True]:
        name='N1_logistic_hybrid' if hybrid else 'N1_logistic_morphology'
        # Avoid coercing nonfinite general inference into useful evidence.
        work=frame.copy();work['feature_available']=available
        x,valid=build_design(work,hybrid);call=np.zeros(len(frame),bool)
        for subject in sorted(frame.subject.unique()):
            model=N1MorphologyModel.load(args.run/f'{name}_heldout_{subject}.json')
            if subject in model.provenance['training_subjects']:raise ValueError('Participant leakage')
            selected=frame.subject.eq(subject).to_numpy();scores=predict_matrix(model,x,valid)
            call[selected]=np.isfinite(scores[selected])&(scores[selected]>=model.threshold)
        methods[name]=call
    out=frame[KEY+['site','reference_positive','evaluable','scoring_eligible']].copy()
    rows=[]
    for name,call in methods.items():
        out[name]=call
        for policy,use in [('original_evaluable_frame',frame.evaluable.to_numpy(bool)),
                           ('processing_unavailable_as_no_call',np.ones(len(frame),bool))]:
            m=metrics_from_counts(confusion(y[use],call[use]))
            for metric in ['sensitivity','specificity','ppv','npv','balanced_accuracy','kappa']:
                rows.append(dict(method=name,policy=policy,metric=metric,estimate=float(m[metric]),
                                 records=int(use.sum()),feature_available=int(available.sum())))
    out.to_csv(args.output/'all_source_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(rows).to_csv(args.output/'all_source_metrics.csv',index=False)
    # Verify the 32,048 paired records exactly retain their frozen predictions.
    prior=pd.read_csv(args.run/'out_of_fold_predictions.csv.gz',low_memory=False)
    joined=prior.merge(out,on=KEY,validate='one_to_one',suffixes=('','_allsource'))
    if len(joined)!=len(prior):raise ValueError('Secondary application lost paired records')
    for name in methods:
        if not joined[name].eq(joined[name+'_detected']).all():raise ValueError('Saved model/rule application changed paired decisions')
    # Paired ablation: added inference features versus morphology alone.
    names=['N1_logistic_hybrid','N1_logistic_morphology'];subjects=sorted(prior.subject.unique())
    counts={name:np.array([confusion(prior.loc[prior.subject.eq(s),'reference_positive'],
                 prior.loc[prior.subject.eq(s),name+'_detected']) for s in subjects]) for name in names}
    rng=np.random.default_rng(SEED);draws=np.zeros((2000,len(subjects)),int)
    for site in sorted(prior.site.unique()):
        ids=[i for i,s in enumerate(subjects) if prior.loc[prior.subject.eq(s),'site'].iloc[0]==site]
        sample=rng.choice(ids,size=(2000,len(ids)),replace=True)
        for i,row in enumerate(sample):draws[i]+=np.bincount(row,minlength=len(subjects))
    a,b=[metrics_from_counts(counts[n].sum(axis=0)) for n in names]
    aa,bb=[metrics_from_counts(draws@counts[n]) for n in names]
    ablation=[]
    for metric in ['sensitivity','specificity','ppv']:
        lo,hi=np.nanquantile(aa[metric]-bb[metric],[.025,.975])
        ablation.append(dict(metric=metric,difference=float(a[metric]-b[metric]),low=lo,high=hi))
    pd.DataFrame(ablation).to_csv(args.output/'hybrid_vs_morphology.csv',index=False)
    write_json(args.output/'verification.json',dict(status='complete',paired_decisions_matched=len(prior),
       source_eligible=len(frame),original_evaluable=int(frame.evaluable.sum()),features_and_inference_available=int(available.sum()),
       original_run_protocol_sha256=sha(args.run/'run_protocol.json'),script_sha256=sha(Path(__file__)),
       note='Frozen held-out models/rules applied without refitting or threshold changes. Processing no-call policy is not an assertion that absent inputs are physiologically negative.'))
    print(pd.DataFrame(ablation).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','records','features','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
