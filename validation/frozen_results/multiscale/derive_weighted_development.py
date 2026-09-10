"""Recombine preserved window p-values after the explicit fixed-weight amendment."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np,pandas as pd
# Import the staged, checksum-verified sources supplied through PYTHONPATH.
from ERPy.inference import fdr_bh
from ERPy.multiscale import conservative_weighted_bonferroni
from validation.benchmark_multiscale import summarize
ROOT=Path(__file__).parent
WEIGHTS=(.1,.1,.8)
WINDOWS=['early','late','matched_broad']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for source,dest in [(ROOT,ROOT/'weighted_development'),(ROOT/'real_baseline',ROOT/'weighted_real_baseline')]:
 dest.mkdir(exist_ok=True)
 if (dest/'derivation_manifest.json').exists():raise FileExistsError(dest)
 windows=pd.read_csv(source/'window_results.csv.gz',float_precision='round_trip')
 methods=pd.read_csv(source/'method_results.csv.gz',float_precision='round_trip')
 families=pd.read_csv(source/'family_results.csv.gz',float_precision='round_trip')
 method_rows=[];family_rows=[]
 wp=windows[windows.window.isin(WINDOWS)].pivot(index=['family_id','channel'],columns='window',values='p_for_combination')[WINDOWS]
 for fid,group in methods[methods.method.eq('multiscale')].groupby('family_id',sort=True):
  new=group.copy();new['method']='weighted_multiscale'
  ps=np.array([conservative_weighted_bonferroni(wp.loc[(fid,c)].to_numpy(),WEIGHTS) for c in new.channel])
  q,calls=fdr_bh(ps,alpha=.05);calls &= new.available.to_numpy()
  new['p_value']=ps;new['q_value']=q;new['call']=calls;method_rows.append(new)
  old=families[(families.family_id.eq(fid))&families.method.eq('multiscale')].iloc[0].to_dict()
  false=calls&~new.is_injected_target.to_numpy()
  target=new.channel.eq('target').to_numpy()
  old.update(method='weighted_multiscale',target_call=bool(calls[target][0]),any_null_call=bool(false.any()),null_calls=int(false.sum()))
  # Recombination is not new detector timing. Keep source timing only in the
  # original rows and mark the derived row runtime unavailable.
  old['elapsed_seconds']=np.nan
  family_rows.append(old)
 allmethods=pd.concat([methods,*method_rows],ignore_index=True)
 allfamilies=pd.concat([families,pd.DataFrame(family_rows)],ignore_index=True)
 for name,df in [('method_results',allmethods),('family_results',allfamilies)]:
  df.to_csv(dest/(name+'.csv.gz'),index=False,compression={'method':'gzip','mtime':0})
 summarize(allfamilies,dest)
 inputs={str(source/name):sha(source/name) for name in ['window_results.csv.gz','method_results.csv.gz','family_results.csv.gz','run_manifest.json']}
 outputs={p.name:sha(p) for p in dest.iterdir() if p.is_file()}
 (dest/'derivation_manifest.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'stage':'Development recombination after observing unweighted outcomes; not independent confirmation','weights':WEIGHTS,'new_core_computations':0,'new_source_trials':0,'amendment_sha256':sha(ROOT/'weighted_amendment/analysis_plan.json'),'inputs':inputs,'outputs':outputs},indent=2))
 print(dest,len(allfamilies),flush=True)
