"""One predeclared optional-bank comparison; generated/source-injected data only.

Run with BLAS threads set to one. The analysis plan must already exist in the
output directory. Results never overwrite an existing completed run. Existing
frozen benchmarks and the default detector are untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import norm

from ERPy.crp_energy import run_crp_energy_array
from ERPy.inference import fdr_bh
from ERPy.multiscale import (MultiScaleConfig, run_multiscale_array, window_seed,
                             conservative_weighted_bonferroni)
from validation.benchmark_crp_energy import _ar1_noise


MORPHS = ("early", "narrow", "late", "biphasic")
METHODS = ("early", "broad", "late", "matched_broad", "multiscale", "weighted_multiscale")
FIXED_WEIGHTS = (.1, .1, .8)
WINDOW_METHODS = ("early", "late", "matched_broad")
CHANNELS = ("target", "reference_01", "reference_02", "reference_03")
TIMES = np.arange(-300, 161, dtype=float) / 500.0
RESPONSE = (TIMES >= .01) & (TIMES <= .3)
MASTER_SEED = 20260908


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(spec):
    payload = json.dumps({k:v for k,v in spec.items() if k != "fixture_file"}, sort_keys=True)
    return int.from_bytes(hashlib.sha256((str(MASTER_SEED)+payload).encode()).digest()[:4], "little")


def waveform(times, morphology, jitter=0.0):
    t = np.asarray(times)-jitter
    if morphology == "early":
        shape = np.exp(-.5*((t-.040)/.012)**2)
    elif morphology == "narrow":
        shape = np.exp(-.5*((t-.025)/.004)**2)
    elif morphology == "late":
        shape = np.exp(-.5*((t-.180)/.025)**2)
    elif morphology == "biphasic":
        shape = np.exp(-.5*((t-.045)/.012)**2)-.8*np.exp(-.5*((t-.125)/.025)**2)
    else:
        raise ValueError(morphology)
    rms = np.sqrt(np.mean(shape**2))
    return shape/rms


def specs_primary(namespace=None):
    specs=[]
    for n, morph, polarity, snr, rep in itertools.product([8,12,24],MORPHS,[-1,1],[.5,1.,2.],range(20)):
        specs.append(dict(scenario="power",n_trials=n,morphology=morph,polarity=polarity,snr=snr,replicate=rep))
    for n, rep in itertools.product([8,12,24],range(80)):
        specs.append(dict(scenario="noise_only",n_trials=n,morphology="none",polarity=0,snr=0.,replicate=rep))
    for arm, rep in itertools.product(["projection_alt_energy_null","energy_alt_projection_null"],range(160)):
        specs.append(dict(scenario=arm,n_trials=12,morphology="stress",polarity=0,snr=0.,replicate=rep))
    if namespace is not None:
        for spec in specs:
            spec["confirmation_namespace"] = str(namespace)
    return specs


def specs_real(fixture_manifest):
    specs=[]
    for item in fixture_manifest["fixtures"]:
        for n, morph, polarity, snr, rep in itertools.product([8,10],MORPHS,[-1,1],[1.,2.],range(10)):
            specs.append(dict(scenario="real_baseline_power",site=item["site"],fixture_file=item["fixture_file"],
                              n_trials=n,morphology=morph,polarity=polarity,snr=snr,replicate=rep))
        # Declared before source-injection outcomes; controlled polarity-null
        # realizations provide a separate check, not a natural-noise FPR.
        for n, rep in itertools.product([8,10],range(80)):
            specs.append(dict(scenario="real_baseline_null",site=item["site"],fixture_file=item["fixture_file"],
                              n_trials=n,morphology="none",polarity=0,snr=0.,replicate=rep))
    return specs


def _real_baseline_family(spec, rng):
    with np.load(spec["fixture_file"],allow_pickle=False) as data:
        source=np.asarray(data["values"],float)
        native_times=np.asarray(data["times"],float)
    # Both source blocks are wholly prestimulation. Resampling uses only each
    # cropped baseline block, with a polyphase low-pass at the target rate.
    native_rate=1/np.median(np.diff(native_times))
    from fractions import Fraction
    fraction=Fraction(500/native_rate).limit_denominator(100000)
    resampled=signal.resample_poly(source,fraction.numerator,fraction.denominator,axis=1)
    length=int(RESPONSE.sum())
    if resampled.shape[1] < 2*length:
        raise ValueError("prestimulation fixture does not contain two disjoint blocks")
    values=np.zeros((spec["n_trials"],len(TIMES),4),float)
    b_indices=np.flatnonzero((TIMES>=-.5)&(TIMES<=-.02))[-length:]
    for c in range(4):
        trials=rng.choice(len(source),size=spec["n_trials"],replace=False)
        first=resampled[trials,:length].copy()
        second=resampled[trials,-length:].copy()
        swap=rng.integers(0,2,size=len(trials)).astype(bool)
        baseline=np.where(swap[:,None],second,first)
        response=np.where(swap[:,None],first,second)
        signs=rng.choice([-1.,1.],size=len(trials))
        values[:,b_indices,c]=baseline*signs[:,None]
        values[:,RESPONSE,c]=response*signs[:,None]
    scale=float(np.sqrt(np.mean((values[:,b_indices,0]-values[:,b_indices,0].mean(axis=1,keepdims=True))**2)))
    return values,scale


def simulate(spec):
    seed=seed_for(spec)
    simulation,detector=np.random.SeedSequence(seed).spawn(2)
    rng=np.random.default_rng(simulation)
    detector_seed=int(detector.generate_state(1,dtype=np.uint32)[0])
    n=spec["n_trials"]
    if spec["scenario"].startswith("real_baseline"):
        values,scale=_real_baseline_family(spec,rng)
    else:
        values=_ar1_noise(rng,(n,len(TIMES),4),phi=.85,scale=10.)
        values*=rng.lognormal(mean=-.5*.18**2,sigma=.18,size=(n,1,4))
        scale=10.
    if spec["scenario"] in ["power","real_baseline_power"]:
        amps=np.clip(rng.normal(1.,.15,size=n),.4,1.6)
        shifts=rng.normal(0.,.003,size=n)
        for i in range(n):
            values[i,RESPONSE,0]+=spec["polarity"]*scale*spec["snr"]*amps[i]*waveform(TIMES[RESPONSE],spec["morphology"],shifts[i])
    elif spec["scenario"]=="projection_alt_energy_null":
        # Exchangeable positive slopes give exact joint energy-null symmetry
        # at every contiguous matched length, including the RMS stabilizer.
        slopes=rng.lognormal(mean=np.log(.2)-.5*.3**2,sigma=.3,size=(n,2))
        ramp=TIMES*500.
        values[:,:,0]=np.where(TIMES[None,:]<0,slopes[:,0,None]*ramp,slopes[:,1,None]*ramp)
    elif spec["scenario"]=="energy_alt_projection_null":
        signs=rng.choice([-1.,1.],size=n)
        values[:,RESPONSE,0]+=20.*signs[:,None]*waveform(TIMES[RESPONSE],"biphasic")[None,:]
    return values,detector_seed,scale


def evaluate(spec):
    started=time.perf_counter()
    seed=seed_for(spec)
    family_id=f"{spec['scenario']}_{seed:010d}"
    common={k:v for k,v in spec.items() if k!='fixture_file'}
    common.update(family_id=family_id,family_seed=seed)
    try:
        values,detector_seed,scale=simulate(spec)
        base=MultiScaleConfig()
        config=replace(base,component_config=replace(base.component_config,random_state=detector_seed))
        method_rows=[];window_rows=[];by_method={name:[] for name in METHODS};available={name:[] for name in METHODS}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for c,channel in enumerate(CHANNELS):
                multi=run_multiscale_array(values[:,:,c],TIMES,channel=channel,config=config)
                for name,w in zip(WINDOW_METHODS,multi.windows):
                    result=w.result
                    by_method[name].append(w.p_for_combination)
                    available[name].append(w.available)
                    window_rows.append(dict(common,channel=channel,window=name,window_seed=w.seed,p_projection=result.p_crp if result else np.nan,
                        p_energy=result.p_energy if result else np.nan,p_joint=w.p_joint,p_for_combination=w.p_for_combination,
                        available=w.available,status=w.status,n_clean=result.n_trials_clean if result else 0,
                        response_start=result.response_window[0] if result else np.nan,response_stop=result.response_window[1] if result else np.nan,
                        baseline_start=result.baseline_window[0] if result else np.nan,baseline_stop=result.baseline_window[1] if result else np.nan,
                        projection_exact=result.reproducibility_test_exact if result else False,energy_exact=result.energy_test_exact if result else False,
                        projection_randomizations=result.reproducibility_n_randomizations if result else 0,energy_randomizations=result.energy_n_permutations if result else 0))
                # Original broad response setting, same family/trial set.
                cleaned=values[:,:,c].copy()
                cleaned[np.setdiff1d(np.arange(len(values)),multi.common_clean_trial_indices),:]=np.nan
                legacy=(.015,.3)
                old=run_crp_energy_array(cleaned,TIMES,channel=channel,config=replace(config.component_config,response_window=legacy,
                     artifact_interval=(0.,.015),random_state=window_seed(detector_seed,channel,legacy)))
                ok=bool(np.isfinite(old.p_joint) and old.qc_status=="pass")
                by_method["broad"].append(float(old.p_joint) if ok else 1.)
                available["broad"].append(ok)
                window_rows.append(dict(common,channel=channel,window="broad",window_seed=window_seed(detector_seed,channel,legacy),
                    p_projection=old.p_crp,p_energy=old.p_energy,p_joint=old.p_joint,p_for_combination=old.p_joint if ok else 1.,
                    available=ok,status=old.qc_status,n_clean=old.n_trials_clean,response_start=old.response_window[0],response_stop=old.response_window[1],
                    baseline_start=old.baseline_window[0],baseline_stop=old.baseline_window[1],projection_exact=old.reproducibility_test_exact,
                    energy_exact=old.energy_test_exact,projection_randomizations=old.reproducibility_n_randomizations,energy_randomizations=old.energy_n_permutations))
                by_method["multiscale"].append(multi.p_multiscale)
                available["multiscale"].append(multi.available)
                by_method["weighted_multiscale"].append(conservative_weighted_bonferroni(
                    [w.p_for_combination for w in multi.windows], FIXED_WEIGHTS))
                available["weighted_multiscale"].append(multi.available)
            family_rows=[]
            power=spec["scenario"] in ["power","real_baseline_power"]
            for method in METHODS:
                p=np.asarray(by_method[method]);q,calls=fdr_bh(p,alpha=.05)
                ok=np.asarray(available[method],bool);calls &= ok
                truth=np.array([power,False,False,False])
                for c,channel in enumerate(CHANNELS):
                    method_rows.append(dict(common,method=method,channel=channel,p_value=p[c],q_value=q[c],call=bool(calls[c]),
                                            available=bool(ok[c]),is_injected_target=bool(truth[c]),noise_scale_uv=scale))
                false=calls&~truth
                family_rows.append(dict(common,method=method,target_call=bool(calls[0]),target_available=bool(ok[0]),
                    any_null_call=bool(false.any()),null_calls=int(false.sum()),null_contacts=int((~truth).sum()),
                    available_contacts=int(ok.sum()),noise_scale_uv=scale,elapsed_seconds=time.perf_counter()-started))
        warning_text=sorted(set(str(w.message) for w in caught))
        return dict(families=family_rows,methods=method_rows,windows=window_rows,warnings=warning_text,error=None)
    except Exception:
        return dict(families=[],methods=[],windows=[],warnings=[],error=dict(common,traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started))


def wilson(k,n):
    if not n:return np.nan,np.nan
    z=float(norm.ppf(.975));rate=k/n;den=1+z*z/n
    center=(rate+z*z/(2*n))/den;half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
    return center-half,center+half


def summarize(families, output):
    fields=["scenario","n_trials","morphology","polarity","snr","method"]
    if "site" in families:
        families["site"]=families["site"].fillna("generated")
        fields.insert(1,"site")
    rows=[]
    for keys,group in families.groupby(fields,dropna=False,sort=True):
        row=dict(zip(fields,keys));n=len(group);k=int(group.target_call.sum());null=int(group.any_null_call.sum())
        lo,hi=wilson(k,n);flo,fhi=wilson(null,n)
        row.update(families=n,target_available=int(group.target_available.sum()),target_calls=k,target_call_rate=k/n,
                   target_rate_ci_low=lo,target_rate_ci_high=hi,any_null_families=null,any_null_family_rate=null/n,
                   any_null_ci_low=flo,any_null_ci_high=fhi,null_contact_calls=int(group.null_calls.sum()),
                   null_contact_count=int(group.null_contacts.sum()),null_contact_rate=group.null_calls.sum()/group.null_contacts.sum())
        rows.append(row)
    cells=pd.DataFrame(rows);cells.to_csv(output/'all_cells.csv',index=False)
    paired=[]
    power=families[families.scenario.eq("power")]
    rng=np.random.default_rng(20260909)
    for candidate in ["multiscale", "weighted_multiscale"]:
        for n in ["all",8,12,24]:
            subset=power if n=="all" else power[power.n_trials.eq(n)]
            if subset.empty:
                continue
            wide=subset.pivot(index=["family_id","n_trials","morphology","polarity","snr"],columns="method",values="target_call").astype(int)
            for comparison in ["early","broad","matched_broad"]:
                diff=wide[candidate]-wide[comparison];samples=np.zeros(2000)
                for _,indices in wide.groupby(level=["n_trials","morphology","polarity","snr"]).indices.items():
                    values=diff.iloc[indices].to_numpy();samples+=values[rng.integers(0,len(values),size=(2000,len(values)))].sum(axis=1)
                samples/=len(wide)
                lo,hi=np.quantile(samples,[.025,.975])
                paired.append(dict(candidate=candidate,n_trials=n,comparison=comparison,families=len(wide),candidate_recovery=wide[candidate].mean(),
                    reference_recovery=wide[comparison].mean(),paired_difference=diff.mean(),ci_low=lo,ci_high=hi,
                    multiscale_only=int(((wide[candidate]==1)&(wide[comparison]==0)).sum()),reference_only=int(((wide[candidate]==0)&(wide[comparison]==1)).sum())))
    if paired:pd.DataFrame(paired).to_csv(output/'paired_recovery.csv',index=False)
    pooled=families.groupby(["scenario","n_trials","method"],dropna=False).agg(families=("family_id","size"),target_calls=("target_call","sum"),
        target_available=("target_available","sum"),any_null_families=("any_null_call","sum"),null_contact_calls=("null_calls","sum"),null_contacts=("null_contacts","sum")).reset_index()
    pooled["target_rate"]=pooled.target_calls/pooled.families
    pooled["any_null_family_rate"]=pooled.any_null_families/pooled.families
    pooled.to_csv(output/'pooled_by_trials.csv',index=False)
    return cells,pooled,pd.DataFrame(paired)


def run(output, workers, real=False, weighted_confirmation=False):
    plan=output.parent/'analysis_plan.json' if real else output/'analysis_plan.json'
    if not plan.exists():raise FileNotFoundError("Write the analysis plan before outcomes")
    if (output/'run_manifest.json').exists():raise FileExistsError("Completed runs are immutable; do not overwrite")
    output.mkdir(parents=True,exist_ok=True)
    if real:
        fixture_path=output.parent/'fixtures/fixture_manifest.json'
        fixtures=json.loads(fixture_path.read_text())
        for item in fixtures['fixtures']:
            if sha(item['fixture_file']) != item['fixture_sha256']:
                raise ValueError('Fixture checksum changed after selection: '+item['fixture_file'])
        specs=specs_real(fixtures)
    else:
        namespace = None
        if weighted_confirmation:
            namespace = json.loads(plan.read_text())["confirmation_namespace"]
        specs=specs_primary(namespace)
        if weighted_confirmation:
            development = json.loads((output.parent/'declared_scenarios.json').read_text())
            if set(map(seed_for, specs)) & set(map(seed_for, development)):
                raise ValueError('Confirmation seed overlaps development')
    if len({seed_for(s) for s in specs}) != len(specs):
        raise ValueError('Scenario seed collision; preserve the declared plan and report before execution')
    (output/'declared_scenarios.json').write_text(json.dumps(specs,indent=2))
    start=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,result in enumerate(pool.map(evaluate,specs,chunksize=2),1):
            results.append(result)
            if i%50==0:print(f"Completed {i}/{len(specs)} predeclared families; failures={sum(r['error'] is not None for r in results)}",flush=True)
    elapsed=time.perf_counter()-start
    failures=[r['error'] for r in results if r['error'] is not None]
    (output/'failures.json').write_text(json.dumps(failures,indent=2))
    (output/'warnings.json').write_text(json.dumps(sorted(set(w for r in results for w in r['warnings'])),indent=2))
    families=pd.DataFrame([x for r in results for x in r['families']]);methods=pd.DataFrame([x for r in results for x in r['methods']]);windows=pd.DataFrame([x for r in results for x in r['windows']])
    for name,frame in [("family_results",families),("method_results",methods),("window_results",windows)]:frame.to_csv(output/(name+'.csv.gz'),index=False,compression={"method":"gzip","mtime":0})
    cells,pooled,paired=summarize(families,output)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'analysis_plan_sha256':sha(plan),'predeclared_families':len(specs),'completed_families':len(specs)-len(failures),
        'failed_families':len(failures),'methods':METHODS,'fixed_weights':FIXED_WEIGHTS,'weighted_confirmation':weighted_confirmation,'elapsed_wall_seconds':elapsed,'worker_processes':workers,'python':platform.python_version(),'numpy':np.__version__,
        'platform':platform.platform(),'thread_settings':{key:os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        'code_hashes':{str(p):sha(p) for p in [Path(__file__),Path(__file__).parents[1]/'ERPy/multiscale.py',Path(__file__).parents[1]/'ERPy/crp_energy.py']},
        'independence_note':'Generated primary families independent; all methods paired within family. Source-injection scenario reuses ten native baseline trials per site and is conditional on those sources, not new independent clinical observations.',
        'null_note':'Global multiscale null requires at least one component null in EVERY bank window. Both declared stresses meet this; no mixed-window unsupported null is claimed.'}
    # Freeze analysis artifacts only. Redirected progress logs may still be
    # growing while this manifest is written; unrelated directory files are
    # not outputs of this analysis.
    artifact_names = ['failures.json', 'warnings.json', 'declared_scenarios.json',
                      'family_results.csv.gz', 'method_results.csv.gz',
                      'window_results.csv.gz', 'all_cells.csv', 'pooled_by_trials.csv']
    if not real:
        artifact_names.append('paired_recovery.csv')
    manifest['outputs'] = {name:sha(output/name) for name in artifact_names}
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--real-baseline',action='store_true');ap.add_argument('--weighted-confirmation',action='store_true');args=ap.parse_args()
    run(args.output_dir,args.workers,args.real_baseline,args.weighted_confirmation)
