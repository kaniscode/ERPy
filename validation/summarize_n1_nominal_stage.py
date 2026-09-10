#!/usr/bin/env python3
"""Post-result description of the fixed detector's nominal screening stage.

This does not replace its prespecified BH primary decision or change a gate,
p value, family, threshold or model. The screen is not multiplicity adjusted.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation.predict_n1_label_free import ROOT, KEY, PROTOCOL_SHA256, sha
from validation.score_n1_label_free import counts, cluster_draws, clean_json
from validation.score_erdetect_validation import metrics_from_counts


def nominal_screen(frame):
    """The fixed gate and unadjusted joint p<=.05, with availability explicit."""
    return (frame.available.eq(True) & frame.morphology_gate.eq(True)
            & np.isfinite(frame.p_joint) & frame.p_joint.le(.05)).to_numpy(bool)


def execute(scoring,output):
    scoring,output=Path(scoring),Path(output)
    if output.exists():raise ValueError('Use a new output directory')
    manifest=json.loads((scoring/'manifest.json').read_text())
    summary=json.loads((scoring/'summary.json').read_text())
    if manifest['protocol_sha256']!=PROTOCOL_SHA256 or not summary['reconstructed_reference_matches_frozen']:
        raise ValueError('Completed fixed-rule human scoring contract required')
    for name,value in manifest['outputs'].items():
        if sha(scoring/name)!=value:raise ValueError('Completed scoring bytes changed')
    frame=pd.read_csv(scoring/'paired_records.csv.gz')
    if len(frame)!=32048 or frame.subject.nunique()!=13:raise ValueError('Fixed paired frame changed')
    y=frame.reference_positive.to_numpy(float)
    call=nominal_screen(frame)
    if not np.array_equal(call,frame.decision_stage.isin([
            'final_pass','gate_pass_joint_p_at_most_0.05_but_screened_q_above_0.05']).to_numpy()):
        raise ValueError('Nominal screen differs from the already recorded decision stages')
    subjects=sorted(frame.subject.unique())
    site={s:frame.loc[frame.subject.eq(s),'site'].iloc[0] for s in subjects}
    draws=cluster_draws(subjects,site)
    vectors={name:np.array([counts(y[frame.subject.eq(s)],d[frame.subject.eq(s)]) for s in subjects])
             for name,d in {'N1_nominal_screen':call,'archived_ER_detect':frame.archived_detected.to_numpy(bool)}.items()}
    point={k:metrics_from_counts(v.sum(axis=0)) for k,v in vectors.items()}
    boot={k:metrics_from_counts(draws@v) for k,v in vectors.items()}
    metrics=[];differences=[]
    for metric in ['sensitivity','specificity','ppv']:
        lo,hi=np.nanquantile(boot['N1_nominal_screen'][metric],[.025,.975],method='linear')
        metrics.append(dict(method='N1_nominal_screen',metric=metric,estimate=point['N1_nominal_screen'][metric],
            low=lo,high=hi,n_records=len(frame),n_subjects=len(subjects),calls=int(call.sum()),
            role='post_result_nominal_screen_diagnostic'))
        low,high=np.nanquantile(boot['N1_nominal_screen'][metric]-boot['archived_ER_detect'][metric],[.025,.975],method='linear')
        differences.append(dict(method='N1_nominal_screen',reference='archived_ER_detect',metric=metric,
            difference=point['N1_nominal_screen'][metric]-point['archived_ER_detect'][metric],
            low=low,high=high,ci_excludes_zero=bool(low>0 or high<0),n_records=len(frame),n_subjects=len(subjects),
            role='post_result_nominal_screen_diagnostic'))
    base_metrics=pd.read_csv(scoring/'operating_points.csv')
    base_metrics['role']=base_metrics.method.map({'label_free_N1':'prespecified_family_adjusted_primary',
                                                'archived_ER_detect':'archived_per_record_comparator'})
    base_diff=pd.read_csv(scoring/'paired_differences.csv')
    base_diff['role']='prespecified_family_adjusted_primary'
    output.mkdir(parents=True)
    pd.concat([base_metrics,pd.DataFrame(metrics)],ignore_index=True).to_csv(output/'operating_points.csv',index=False)
    pd.concat([base_diff,pd.DataFrame(differences)],ignore_index=True).to_csv(output/'paired_differences.csv',index=False)
    individual=frame[KEY+['reference_positive','morphology_gate','p_joint','q_screened','detected','archived_detected']].copy()
    individual['screen_positive']=call
    individual.to_csv(output/'nominal_stage_records.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    participant=[]
    for s,v in zip(subjects,vectors['N1_nominal_screen']):
        participant.append(dict(subject=s,site=site[s],**dict(zip(['tp','fn','tn','fp','available','eligible'],v)),
                                **metrics_from_counts(v)))
    pd.DataFrame(participant).to_csv(output/'participant_metrics.csv',index=False)
    report=dict(status='passed',status_of_analysis='post-result diagnostic of an already recorded stage; no new primary/threshold selection',
        primary_protocol_sha256=PROTOCOL_SHA256,primary_method_unchanged=True,
        screen_definition='available & fixed 50uV-floor morphology gate & p_joint<=0.05; no BH selection',
        screen_is_fdr_adjusted=False,reference='same raw-human-vote reconstructed reference as completed scoring',
        records=len(frame),participants=subjects,calls=int(call.sum()),
        confusion=dict(zip(['tp','fn','tn','fp','available','eligible'],vectors['N1_nominal_screen'].sum(axis=0))),
        estimates=point['N1_nominal_screen'],paired_differences=differences,
        uncertainty='same2000 whole-participant site-stratified draws,seed20260908; nominal conditional intervals',
        input_scoring_manifest_sha256=sha(scoring/'manifest.json'),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'validation/score_n1_label_free.py']})
    (output/'summary.json').write_text(json.dumps(clean_json(report),indent=2,allow_nan=False)+'\n')
    (output/'manifest.json').write_text(json.dumps({'outputs':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        'primary_protocol_sha256':PROTOCOL_SHA256,'role':'post_result_nominal_screen_diagnostic'},indent=2)+'\n')
    print(json.dumps(clean_json(report),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scoring',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();execute(args.scoring,args.output)
