"""Independent raw-cache versus frozen-mean check; no response label values used."""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat

import argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("root",type=Path)
p.add_argument("--output",type=Path,required=True)
args=p.parse_args()
root=args.root
rows=[]
for folder in sorted((root/'epochs').glob('UMCU*')):
    if not folder.is_dir():
        continue
    mats=list((root/'data/ds004774/derivatives/app_detect_output').glob('sub-'+folder.name+'_*/erdetect_data.mat'))
    if not mats:
        continue
    m=loadmat(mats[0],simplify_cells=True)
    channels_source=list(np.atleast_1d(m['channel_labels']))
    pairs_source=list(np.atleast_1d(m['stimpair_labels']))
    avg=m['ccep_average']
    onset=int(m['onset_sample'])
    for path in sorted(folder.glob('*.npz')):
        with np.load(path,allow_pickle=False) as a:
            pair=str(a['stimpair'])
            metadata=json.loads(str(a['metadata_json']))
            if metadata.get('cache_schema_version') != 2:
                rows.append({'subject':folder.name,'stimpair':pair,'status':'old_alignment_cache'})
                continue
            pair_index=next((i for i,x in enumerate(pairs_source) if sorted(x.split('-'))==sorted(pair.split('-'))),None)
            if pair_index is None:
                rows.append({'subject':folder.name,'stimpair':pair,'status':'pair_absent_from_archived_output','n_trials':int(len(a['trials']))})
                continue
            x=a['trials'].astype(float)
            fs=float(a['sfreq'])
            cache_onset=int(np.argmin(np.abs(a['times'])))
            base_start=cache_onset+round(-.5*fs)
            base_stop=cache_onset+round(-.02*fs)
            centered=x-np.median(x[:,base_start:base_stop,:],axis=1)[:,None,:]
            ours=centered.mean(axis=0)
            offsets=np.arange(int(np.ceil(.010*fs)),int(np.floor(.300*fs))+1)
            ours_time=a['times'][cache_onset+offsets]
            source_time=np.asarray(m['epoch_time_s'])[onset+offsets]
            channels=list(a['channels'])
            shared=[c for c in channels_source if c in channels and c not in pair.split('-')]
            ours_matrix=ours[cache_onset+offsets][:,[channels.index(c) for c in shared]]
            source_matrix=avg[[channels_source.index(c) for c in shared],pair_index,:][:,onset+offsets].T
            finite_pattern_mismatches=int(np.sum(np.isfinite(ours_matrix)!=np.isfinite(source_matrix)))
            diff=ours_matrix-source_matrix
            finite=np.isfinite(diff)
            values=diff[finite]
            row={'subject':folder.name,'stimpair':pair,'status':'compared','n_trials':len(x),
                 'n_shared_channels':len(shared),'n_time_samples':len(offsets),'finite_compared_values':len(values),
                 'max_abs_difference_uv':float(np.max(np.abs(values))) if len(values) else None,
                 'rms_difference_uv':float(np.sqrt(np.mean(values**2))) if len(values) else None,
                 'max_timestamp_difference_s':float(np.max(np.abs(ours_time-source_time))),
                 'finite_pattern_mismatches':finite_pattern_mismatches,
                 'mean_exact':bool(len(values)>0 and finite_pattern_mismatches==0 and np.all(values==0)),
                 'source_sampling_rate':float(m['sampling_rate']),'loader_sampling_rate':fs}
            rows.append(row)
report={'baseline':'half-open integer-rounded median baseline -0.5 to -0.02 s',
        'time_window_s':[.010,.300], 'rows':rows}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report,indent=2))
for subject in sorted(set(r['subject'] for r in rows)):
    subset=[r for r in rows if r['subject']==subject]
    compared=[r for r in subset if r['status']=='compared']
    print(json.dumps({'subject':subject,'cached_pairs':len(subset),'compared_pairs':len(compared),
                      'exact_pairs':sum(r['mean_exact'] for r in compared),
                      'max_rms_difference_uv':max((r['rms_difference_uv'] or 0 for r in compared),default=0)}),flush=True)
