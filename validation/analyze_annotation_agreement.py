#!/usr/bin/env python3
"""Descriptive reanalysis of saved public broad-window calls; no detector rerun.

Run with numpy and pandas; validated on Python 3.12.5. Recompute aggregates
without raw data, or pass --epochs to reconstruct the selected public traces.
--verify checks pinned files and replays every derived table in a temporary folder.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, tempfile
import numpy as np
import pandas as pd

KEY=['subject','stimpair','channel']
CONFIG='broad_15_300ms'
SEED=20260907
RESAMPLES=2000
COUNTS=['tp','fn','tn','fp','available','eligible']

def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def js(p,value): p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def scalar(x): return float(x) if np.isfinite(x) else None
def measures(c):
    tp,fn,tn,fp,n,eligible=np.moveaxis(np.asarray(c,float),-1,0)
    with np.errstate(divide='ignore',invalid='ignore'):
        expected=((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp))/n**2
        return {'agreement':(tp+tn)/n,'disagreement':(fp+fn)/n,
          'kappa':((tp+tn)/n-expected)/(1-expected),
          'n1_positive_call_rate':tp/(tp+fn),
          'n1_negative_call_rate':fp/(tn+fp),
          'n1_call_rate_difference':tp/(tp+fn)-fp/(tn+fp),
          'positive_agreement':2*tp/(2*tp+fp+fn),
          'negative_agreement':2*tn/(2*tn+fp+fn),
          'n1_fraction_among_calls':tp/(tp+fp),'coverage':n/eligible}
def interval(v):
    a=np.asarray(v);a=a[np.isfinite(a)]
    q=np.quantile(a,[.025,.975]) if len(a) else [np.nan,np.nan]
    return {'low':scalar(q[0]),'high':scalar(q[1]),'valid_resamples':len(a)}
def boot_counts(subject_vectors):
    rng=np.random.default_rng(SEED);draws=np.zeros((RESAMPLES,subject_vectors.shape[1]))
    for _,g in subject_vectors.groupby(level='site',sort=True):
        v=g.to_numpy();ix=rng.integers(0,len(v),(RESAMPLES,len(v)))
        draws+=v[ix].sum(axis=1)
    return draws

def execute(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    frozen=a.repo/'validation/frozen_results';root=frozen/'erdetect'
    input_paths=[root/'record_predictions_and_labels.csv.gz',root/'rater_label_joins.csv.gz',root/'metrics.csv',root/'summary.json',a.repo/'validation/predict_erdetect_epochs.py',a.repo/'validation/score_erdetect_validation.py',a.repo/'ERPy/crp_energy.py',root/'coverage_audit/annotation_source_flow.json',frozen/'n1/nested/paired_differences.csv',frozen/'n1/secondary/hybrid_vs_morphology.csv']
    inputs={str(x.relative_to(a.repo)):sha(x) for x in input_paths}
    coverage=next(r for r in json.loads((root/'coverage_audit/annotation_source_flow.json').read_text())['overall'] if r['cohort']=='independent13')
    all_records=pd.read_csv(input_paths[0],dtype={'random_seed':str},low_memory=False)
    f=all_records[all_records.configuration.eq(CONFIG)&all_records.independent_external_subject&all_records.scoring_eligible&all_records.reference_positive.notna()].copy()
    assert not f.duplicated(KEY).any()
    assert f.subject.nunique()==13
    assert (f.detected==(f.q_joint<=.05)).all()
    assert np.array_equal(f.evaluable,np.isfinite(f.q_joint))
    y=f.reference_positive.to_numpy();d=f.detected.to_numpy();v=f.evaluable.to_numpy()
    assert np.isfinite(y).all() and ((y>=0)&(y<=1)).all(), 'Invalid fractional reference'
    for name,vals in {'tp':y*d*v,'fn':y*(~d)*v,'tn':(1-y)*(~d)*v,'fp':(1-y)*d*v,'available':v.astype(float),'eligible':np.ones(len(f))}.items(): f[name]=vals
    sv=f.groupby(['site','subject'],sort=True)[COUNTS].sum();draw=boot_counts(sv);points=measures(sv.sum().to_numpy());ci={k:interval(v) for k,v in measures(draw).items()}
    summary={'analysis':'Broad/default response-window detector versus released negative N1 annotations','configuration':CONFIG,'configuration_settings':{'response_window_s':[.015,.300],'artifact_interval_s':[0,.015],'candidate_baseline_s':[-1,-.1],'baseline_matching':'last baseline samples, same count as response; separately demeaned segments','normalization':'per-trial nanmedian over rounded half-open [-.5,-.02) slice','minimum_clean_trials':8,'family':'saved BH across all retained good nonstimulation ECoG contacts per pair; finite joint p','call':'saved q_joint <=0.05; no additional acquisition-level contact QC gate was run in this external benchmark'},'cohort':{'participants':sorted(f.subject.unique()),'available_records':int(f.evaluable.sum()),'source_and_site_eligible_labeled_records':len(f),'MAYO01_excluded':'previous development/worked example overlap','expanded_otherwise_eligible_records':coverage['expanded_coverage_frame'],'expanded_coverage':coverage['expanded_evaluable_fraction']},'counts':dict(zip(COUNTS,sv.sum().to_numpy().tolist())),'metrics':{k:{'value':float(x),**ci[k]} for k,x in points.items()},'interval_method':{'resamples':RESAMPLES,'seed':SEED,'unit':'whole participant','strata':'site (4 Mayo, 9 UMCU)','quantile':'numpy linear 2.5th/97.5th percentile','interpretation':'nominal conditional interval for fixed saved predictions in this convenience cohort; no training uncertainty or guaranteed population coverage'},'tests':'No independent-record significance tests or bootstrap-tail p values. Exemplar q values are original detector values, not tests of annotator correctness.'}
    stored=pd.read_csv(root/'metrics.csv');s=stored[(stored.reference_view=='equal_record_available_rater')&(stored.cohort_scope=='independent_external_excluding_development_overlap')&(stored.configuration==CONFIG)&(stored.comparison=='erpy_all_source_eligible')&(stored.missingness_policy=='available_case')&(stored.method=='ERPy')&(stored.level=='overall')].iloc[0]
    for x in COUNTS: assert np.isclose(summary['counts'][x],s[x],rtol=0,atol=1e-8)
    for ours,theirs in [('agreement','accuracy'),('kappa','kappa'),('n1_positive_call_rate','sensitivity'),('n1_fraction_among_calls','ppv'),('coverage','coverage')]:
        assert np.isclose(points[ours],s[theirs],rtol=0,atol=1e-12)
        for bound in ['low','high']: assert np.isclose(ci[ours][bound],s[theirs+'_'+bound],rtol=0,atol=1e-12)
    summary['saved_metric_reproduction']='exact within 1e-12 for verified points and intervals'
    sub=sv.reset_index()
    for k,x in measures(sv.to_numpy()).items():sub[k]=x
    sub.to_csv(out/'subject_metrics.csv',index=False)
    matrix=[]
    for i,(name,ref,call) in enumerate([('tp','N1-positive vote',True),('fn','N1-positive vote',False),('tn','N1-negative vote',False),('fp','N1-negative vote',True)]):
        matrix.append({'cell':name,'reference':ref,'detected':call,'weighted_record_equivalents':summary['counts'][name],'fraction_of_available_records':summary['counts'][name]/summary['counts']['available'],**interval(draw[:,i]/draw[:,4])})
    pd.DataFrame(matrix).to_csv(out/'crossclassification.csv',index=False)
    f=f[f.evaluable].copy()
    labels=pd.read_csv(input_paths[1],dtype={'random_seed':str},low_memory=False)
    labels=labels[labels.configuration.eq(CONFIG)&labels.independent_external_subject&labels.scoring_eligible&labels.evaluable&labels.raw_annotation.isin([0,1,2])].copy()
    assert not labels.duplicated(KEY+['rater']).any()
    agg=labels.groupby(KEY).agg(n1_votes=('raw_annotation',lambda x:int((x==1).sum())),p1_votes=('raw_annotation',lambda x:int((x==2).sum())),zero_votes=('raw_annotation',lambda x:int((x==0).sum())),n_raters_raw=('rater','size')).reset_index()
    f=f.merge(agg,on=KEY,validate='one_to_one');assert np.allclose(f.n1_votes/f.n_raters_raw,f.reference_positive);assert (f.n_raters==f.n_raters_raw).all()
    f['annotation_group']=np.select([f.n_raters.eq(1)&f.reference_positive.eq(1),f.n_raters.eq(1)&f.reference_positive.eq(0),f.n_raters.ge(2)&f.reference_positive.eq(1),f.n_raters.ge(2)&f.reference_positive.eq(0)],['Single-rater N1-positive','Single-rater N1-negative','Unanimous N1-positive (>=2 raters)','Unanimous N1-negative (>=2 raters)'],default='Mixed N1 ratings (>=2 raters)')
    groups=list(f.annotation_group.unique());groups.sort();rates=[];rate_vectors={}
    subjects=sv.index
    for group in groups:
        z=f[f.annotation_group==group];v=z.groupby(['site','subject']).agg(calls=('detected','sum'),n=('detected','size')).reindex(subjects,fill_value=0)
        bd=boot_counts(v);with_rate=np.divide(bd[:,0],bd[:,1],out=np.full(RESAMPLES,np.nan),where=bd[:,1]>0)
        rate_vectors[group]=with_rate
        rates.append({'annotation_group':group,'n_records':len(z),'n_calls':int(z.detected.sum()),'n_participants':z.subject.nunique(),'call_rate':float(z.detected.mean()),**interval(with_rate)})
    pd.DataFrame(rates).to_csv(out/'annotation_group_rates.csv',index=False)
    f['three_group']=np.select([f.reference_positive.eq(1),f.reference_positive.eq(0)],['All available raters N1-positive','All available raters N1-negative'],default='Mixed N1 ratings')
    three=[]
    for group,z in f.groupby('three_group',sort=True):
        v=z.groupby(['site','subject']).agg(calls=('detected','sum'),n=('detected','size')).reindex(subjects,fill_value=0);bd=boot_counts(v)
        br=np.divide(bd[:,0],bd[:,1],out=np.full(RESAMPLES,np.nan),where=bd[:,1]>0)
        three.append({'annotation_group':group,'n_records':len(z),'n_calls':int(z.detected.sum()),'n_single_rater_records':int(z.n_raters.eq(1).sum()),'call_rate':float(z.detected.mean()),**interval(br)})
    pd.DataFrame(three).to_csv(out/'three_annotation_group_rates.csv',index=False)
    sg=f.groupby(['site','subject','three_group']).agg(n_records=('detected','size'),n_calls=('detected','sum')).reset_index();sg['call_rate']=sg.n_calls/sg.n_records;sg.to_csv(out/'subject_three_annotation_group_rates.csv',index=False)

    subjectgroups=f.groupby(['site','subject','annotation_group']).agg(n_records=('detected','size'),n_calls=('detected','sum')).reset_index();subjectgroups['call_rate']=subjectgroups.n_calls/subjectgroups.n_records;subjectgroups.to_csv(out/'subject_annotation_group_rates.csv',index=False)
    rp={r['annotation_group']:r['call_rate'] for r in rates};contrasts=[]
    for g1,g0 in [('Unanimous N1-positive (>=2 raters)','Unanimous N1-negative (>=2 raters)'),('Mixed N1 ratings (>=2 raters)','Unanimous N1-negative (>=2 raters)'),('Unanimous N1-positive (>=2 raters)','Mixed N1 ratings (>=2 raters)')]:
        contrasts.append({'group_1':g1,'group_0':g0,'call_rate_difference':rp[g1]-rp[g0],**interval(rate_vectors[g1]-rate_vectors[g0])})
    pd.DataFrame(contrasts).to_csv(out/'annotation_group_contrasts.csv',index=False)
    rc=labels.groupby(['site','rater','raw_annotation']).size().rename('n_annotations').reset_index();rc.to_csv(out/'observed_annotation_codes_by_rater.csv',index=False)
    z=f[f.p1_votes>0];summary['observed_P1_subset']={'records_with_at_least_one_P1_vote':len(z),'broad_calls':int(z.detected.sum()),'also_N1_positive_vote':int((z.n1_votes>0).sum()),'all_available_raters_P1':int((z.p1_votes==z.n_raters).sum()),'interpretation':'Observed P1 annotations only. No assumption that code0 rules out P1 or later effects; categorical coverage differs by rater. Not a validated any-response reference.'}
    keep=KEY+['site','annotation_group','n_raters','n1_votes','p1_votes','zero_votes','reference_positive','detected','q_joint','p_joint','p_reproducibility','p_energy','family_size','n_trials_total','n_trials_clean','rms_ratio_db','epoch_sha256','random_seed']
    f[keep].to_csv(out/'record_classification.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    examples=[]
    for name,ref,call in [('both_N1_and_default',1,True),('neither_N1_nor_default',0,False),('default_without_N1',0,True),('N1_without_default',1,False)]:
        z=f[f.n_raters.ge(2)&f.reference_positive.eq(ref)&f.detected.eq(call)].copy();target=float(np.log10(z.q_joint).median());z['median_logq_distance']=abs(np.log10(z.q_joint)-target);z=z.sort_values(['median_logq_distance']+KEY);row=z.iloc[0]
        e={k:row[k].item() if isinstance(row[k],np.generic) else row[k] for k in keep};e.update(example=name,selection_pool_records=len(z),selection_target_log10_q=target,selection_rule='Within strict binary agreement >=2-rater stratum and saved detector call, closest to median log10(q); ties lexicographic subject/pair/channel. No waveform inspection before selection.')
        examples.append(e)
    pd.DataFrame(examples).to_csv(out/'exemplar_selection.csv',index=False)
    js(out/'summary.json',summary)
    traces_source=a.repo/'validation/annotation_agreement/source_traces'
    trace_manifest=json.loads((traces_source/'manifest.json').read_text())
    for name,item in trace_manifest['files'].items():
        assert sha(traces_source/name)==item['sha256'] and (traces_source/name).stat().st_size==item['bytes'], name
    if a.epochs is None:
        assert (out/'exemplar_selection.csv').read_bytes()==(traces_source/'exemplar_selection.csv').read_bytes(), 'Exemplar selection changed'
        for name in ['exemplar_traces.csv.gz','exemplar_source_provenance.json']:
            shutil.copy2(traces_source/name,out/name)
        exmeta=json.loads((out/'exemplar_source_provenance.json').read_text())
        for e,m in zip(examples,exmeta):
            assert all(e[k]==m[k] for k in KEY+['example','epoch_sha256']), 'Trace identity mismatch'
            assert len(m['clean_trial_indices'])==e['n_trials_clean']
    else:
        # A source NPZ is loaded only after selecting examples from tabular q/labels.
        trace_rows=[];exmeta=[]
        for e in examples:
            subject=e['subject'];pair=e['stimpair'];channel=e['channel'];paths=sorted((a.epochs/subject).glob('*.npz'))
            paths=[p for p in paths if pair in p.stem or '-'.join(reversed(pair.split('-'))) in p.stem]
            assert len(paths)==1,(e,paths);path=paths[0];actual=sha(path);assert actual==e['epoch_sha256'],(path,actual,e['epoch_sha256'])
            with np.load(path,allow_pickle=False) as saved:
                t=np.asarray(saved['times'],float);names=np.char.upper(saved['channels'].astype(str));ix=np.flatnonzero(names==channel);assert len(ix)==1
                arr=np.asarray(saved['values'] if 'values' in saved else saved['trials'],float)[:,:,ix[0]]
                md=json.loads(str(saved['metadata'] if 'metadata' in saved else saved['metadata_json']))
                assert str(saved['subject'])==subject
                if 'channel_status' in saved:assert saved['channel_status'][ix[0]]=='good'
                if 'channel_types' in saved:assert saved['channel_types'][ix[0]].upper()=='ECOG'
            assert arr.shape[0]==e['n_trials_total'];fs=1/np.median(np.diff(t));b0=round((-.5-t[0])*fs);b1=round((-.02-t[0])*fs);arr-=np.nanmedian(arr[:,b0:b1],axis=1,keepdims=True)
            ri=np.flatnonzero((t>=.015)&(t<=.3));bi=np.flatnonzero((t>=-1)&(t<=-.1))[-len(ri):];clean=np.flatnonzero(np.isfinite(arr[:,np.r_[bi,ri]]).all(axis=1));assert len(clean)==e['n_trials_clean']
            # Mean/SEM: exact detector-available trials, normalized as original predictor.
            arr=arr[clean];display=(t>=-.1)&(t<=.35);mean=np.nanmean(arr,axis=0);sem=np.nanstd(arr,axis=0,ddof=1)/np.sqrt(np.isfinite(arr).sum(axis=0))
            for k in np.flatnonzero(display):trace_rows.append({'example':e['example'],'time_s':t[k],'mean_uv':mean[k],'sem_uv':sem[k],'n_finite':int(np.isfinite(arr[:,k]).sum())})
            metadata=dict(e,epoch_path='epochs/'+path.relative_to(a.epochs).as_posix(),epoch_source_sha256=actual,clean_trial_indices=clean.tolist(),response_first_s=float(t[ri[0]]),response_last_s=float(t[ri[-1]]),n_response_samples=len(ri),baseline_first_s=float(t[bi[0]]),baseline_last_s=float(t[bi[-1]]),sfreq_hz=float(fs),source_metadata=md,display_interval_s=[float(t[display][0]),float(t[display][-1])])
            exmeta.append(metadata)
        pd.DataFrame(trace_rows).to_csv(out/'exemplar_traces.csv.gz',index=False,compression={'method':'gzip','mtime':0});js(out/'exemplar_source_provenance.json',exmeta)
    # Six figure7 paired intervals: literal interval status, not p-value stars.
    h=pd.read_csv(input_paths[-2]);h=h[(h.method=='N1_logistic_hybrid')&(h.reference=='archived_ER_detect')].copy();m=pd.read_csv(input_paths[-1]);m['method']='N1_logistic_hybrid';m['reference']='N1_logistic_morphology';h=pd.concat([h,m]);h['interval_status']=np.where((h.low>0)|(h.high<0),'nominal 95% CI excludes zero','nominal 95% CI includes zero');h.to_csv(out/'figure7_paired_intervals.csv',index=False)
    generated=['subject_metrics.csv','crossclassification.csv','annotation_group_rates.csv','three_annotation_group_rates.csv','subject_three_annotation_group_rates.csv','subject_annotation_group_rates.csv','annotation_group_contrasts.csv','observed_annotation_codes_by_rater.csv','record_classification.csv.gz','exemplar_selection.csv','summary.json','exemplar_traces.csv.gz','exemplar_source_provenance.json','figure7_paired_intervals.csv']
    js(out/'manifest.json',{'schema_version':1,'script_sha256':sha(__file__),'input_sha256':inputs,'source_trace_manifest_sha256':sha(traces_source/'manifest.json'),'outputs':{name:{'bytes':(out/name).stat().st_size,'sha256':sha(out/name)} for name in sorted(generated)},'source_epoch_sha256':{e['epoch_path']:e['epoch_source_sha256'] for e in exmeta},'no_input_mutations':all(sha(a.repo/k)==v for k,v in inputs.items())})
    return summary
def verify(repo):
    folder=repo/'validation/annotation_agreement'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['script_sha256']==sha(__file__), 'Analysis source changed'
    for name,digest in manifest['input_sha256'].items():
        assert sha(repo/name)==digest, 'Analysis input changed: '+name
    for name,item in manifest['outputs'].items():
        assert sha(folder/name)==item['sha256'] and (folder/name).stat().st_size==item['bytes'], 'Output changed: '+name
    assert sha(folder/'source_traces/manifest.json')==manifest['source_trace_manifest_sha256']
    with tempfile.TemporaryDirectory(prefix='erpy-annotation-agreement-') as tmp:
        execute(argparse.Namespace(repo=repo,epochs=None,output=Path(tmp)))
        replay=json.loads((Path(tmp)/'manifest.json').read_text())
        assert replay==manifest, 'Recomputed analysis differs from retained outputs'
    return {'status':'passed','outputs_verified':len(manifest['outputs']),'aggregate_replay_exact':True,'source_inputs_unchanged':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--epochs',type=Path,help='Optional recovered ds004774 epoch directory for raw-trace reconstruction')
    parser.add_argument('--output',type=Path,help='New output directory for a reproduction; defaults to retained evidence directory')
    parser.add_argument('--verify',action='store_true',help='Verify input/output hashes and reproduce all tables without changing retained evidence')
    a=parser.parse_args()
    if a.verify:
        if a.output or a.epochs:parser.error('--verify cannot be combined with --output or --epochs')
        print(json.dumps(verify(a.repo),indent=2));return
    if a.output is None:a.output=a.repo/'validation/annotation_agreement'
    result=execute(a)
    print(json.dumps({'status':'completed','cohort':result['cohort'],'output':str(a.output)},indent=2))

if __name__=='__main__':main()
