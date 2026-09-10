"""Post-run preservation, arithmetic and bounded raw-replay audit; no tuning."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np,pandas as pd
# Import the staged, checksum-verified sources supplied through PYTHONPATH.
from validation.benchmark_multiscale import evaluate,wilson,seed_for
ROOT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:pd.read_csv(p,float_precision='round_trip')
checks=[]
def check(name,ok,details=None):
 checks.append(dict(check=name,pass_=bool(ok),details=details))
 if not ok:raise AssertionError(name)
# Preserve the original self-including log metadata. Only scientific artifacts
# are asserted immutable; redirected logs were still being appended at freeze.
for sub in [ROOT,ROOT/'real_baseline',ROOT/'weighted_confirmation']:
 m=json.loads((sub/'run_manifest.json').read_text())
 artifacts={name:h for name,h in m['outputs'].items() if name.endswith(('.csv','.csv.gz','.json'))}
 check(str(sub.name)+': scientific artifact hashes',all(sha(sub/n)==h for n,h in artifacts.items()),len(artifacts))
 check(str(sub.name)+': completed declared grid',m['predeclared_families']==m['completed_families'] and m['failed_families']==0,m['completed_families'])
 check(str(sub.name)+': no recorded warnings',json.loads((sub/'warnings.json').read_text())==[])
 windows=read(sub/'window_results.csv.gz');methods=read(sub/'method_results.csv.gz')
 check(str(sub.name)+': component conjunction',np.allclose(windows.p_joint,np.maximum(windows.p_projection,windows.p_energy),rtol=0,atol=0))
 check(str(sub.name)+': complete detector availability',methods.available.all() and windows.available.all())
 for (fid,method),g in methods.groupby(['family_id','method']):
  order=np.argsort(g.p_value.to_numpy(),kind='stable');ps=g.p_value.to_numpy()[order]
  qs=np.minimum(1,np.minimum.accumulate((ps*len(ps)/np.arange(1,len(ps)+1))[::-1])[::-1])
  actual=g.q_value.to_numpy()[order]
  if not np.allclose(qs,actual,rtol=0,atol=2e-16):raise AssertionError((fid,method,'BH'))
  if not np.array_equal(actual<=.05,g.call.to_numpy()[order]):raise AssertionError((fid,method,'calls'))
 check(str(sub.name)+': independent BH and call arithmetic',True,len(methods))
 # Replay one representative per trial count and each null arm, fixed by
 # lexicographic declared order; comparisons do not select successful cells.
 specs=json.loads((sub/'declared_scenarios.json').read_text());selected=[]
 for scenario in sorted({s['scenario'] for s in specs}):
  for n in sorted({s['n_trials'] for s in specs if s['scenario']==scenario}):
   selected.append(next(s for s in specs if s['scenario']==scenario and s['n_trials']==n))
 for spec in selected:
  result=evaluate(spec);check(str(sub.name)+': replay executes '+str(seed_for(spec)),result['error'] is None)
  new=pd.DataFrame(result['methods']);new=new[new.method.isin(methods.method.unique())]
  old=methods[methods.family_seed.eq(seed_for(spec))]
  columns=['method','channel','p_value','q_value','call','available','is_injected_target','noise_scale_uv']
  a=new[columns].sort_values(['method','channel']).reset_index(drop=True)
  b=old[columns].sort_values(['method','channel']).reset_index(drop=True)
  pd.testing.assert_frame_equal(a,b,check_exact=True)
 check(str(sub.name)+': exact raw-data method replay',True,len(selected))
old=json.loads((ROOT/'declared_scenarios.json').read_text());new=json.loads((ROOT/'weighted_confirmation/declared_scenarios.json').read_text())
check('development and confirmation seeds disjoint',not (set(map(seed_for,old)) & set(map(seed_for,new))))
for sub in [ROOT/'weighted_development',ROOT/'weighted_real_baseline']:
 m=json.loads((sub/'derivation_manifest.json').read_text())
 check(sub.name+': derivation hashes',all(sha(sub/n)==h for n,h in m['outputs'].items()))
check('pre-weighting amendment hash retained',sha(ROOT/'weighted_amendment/analysis_plan.json')=='c674080dbc5bd33de91e4c9242882293c2b0e7ce13d0b67d98bd03a781ece9ab')
# Retain complete marginal summaries; cell-level outputs already hold full grid.
for tag,sub in [('development',ROOT),('weighted_confirmation',ROOT/'weighted_confirmation'),('source_conditional',ROOT/'weighted_real_baseline')]:
 f=read(sub/'family_results.csv.gz')
 groups=['scenario','n_trials','morphology','polarity','method']
 if 'site' in f:groups.insert(1,'site')
 f.groupby(groups).agg(families=('family_id','size'),target_calls=('target_call','sum'),any_null_families=('any_null_call','sum')).reset_index().to_csv(ROOT/(tag+'_shape_polarity_summary.csv'),index=False)
report=['# Optional multiscale comparison: final bounded evaluation','',
'**Decision: retain the existing single-window default.** Neither the original equal-weight bank nor the single amended fixed-weight bank establishes a general recovery improvement over the matched broad window. Both remain optional experimental APIs. No further window or weight search was performed.','',
'The original plan preceded all unweighted outcomes. The explicit weighted amendment (early 0.1, late 0.1, broad 0.8) was written after seeing the unweighted outcome, before any weighted outcomes. Development recombination is separated from one fresh 2,000-family confirmation. Its independent family seeds have no overlap with development. This is an internal prespecified analysis and an openly declared later amendment, not a registered protocol.','',
'## Fresh confirmation: recovery of the injected target','',
'Each row pools the fixed four-shape, two-polarity, three-signal-level grid, 20 independent families per cell. Values are percentages; intervals are stratified paired-family bootstrap 95% intervals for this grid. They do not describe clinical-population uncertainty.','',
'| Trials | Families | Early | Original broad 15–300 ms | Matched broad 10–300 ms | Equal-weight bank | Weighted bank | Weighted minus matched broad (pp, 95% CI) |','|---|---:|---:|---:|---:|---:|---:|---:|']
pool=read(ROOT/'weighted_confirmation/pooled_by_trials.csv');pair=read(ROOT/'weighted_confirmation/paired_recovery.csv')
for n in [8,12,24]:
 d=pool[(pool.scenario=='power')&(pool.n_trials==n)].set_index('method');p=pair[(pair.candidate=='weighted_multiscale')&(pair.n_trials.astype(str)==str(n))&(pair.comparison=='matched_broad')].iloc[0]
 row=[str(n),'480']+[f'{100*d.loc[m,"target_rate"]:.2f}' for m in ['early','broad','matched_broad','multiscale','weighted_multiscale']]+[f'{100*p.paired_difference:+.2f} ({100*p.ci_low:+.2f}, {100*p.ci_high:+.2f})']
 report.append('| '+' | '.join(row)+' |')
p=pair[(pair.candidate=='weighted_multiscale')&(pair.n_trials.astype(str)=='all')&(pair.comparison=='matched_broad')].iloc[0]
report+=['',f'Across all 1,440 confirmation target families, weighted recovery was {100*p.candidate_recovery:.2f}% versus {100*p.reference_recovery:.2f}% for matched broad: {100*p.paired_difference:+.2f} percentage points (95% CI {100*p.ci_low:+.2f} to {100*p.ci_high:+.2f}). The equal-weight development bank recovered 0/480 eight-trial targets, 365/480 twelve-trial targets and 421/480 twenty-four-trial targets; those results remain intact. Weighted development showed a positive 24-trial point difference with a conditional interval above zero, but its fresh confirmation interval includes zero. Do not select the development finding as evidence of general improvement.','',
'At eight trials, an isolated four-contact discovery has equal-weight adjusted resolution 12/128 = 0.09375. The amended broad weight permits 4/(128×0.8) = 0.0390625, while retaining penalties for the other windows. This restores the possibility of broad-window detection but still reduces recovery versus a single broad test. Unavailable windows contribute p=1 and never redistribute or reduce the declared multiplicity.','',
'## Global-null checks in fresh confirmation','',
'Rates below count families with at least one false call; Wilson intervals use independent generated families. They are empirical checks under the stated generators, not proof of calibration under arbitrary physiology or response-dependent selection.','',
'| Construction | Families | Matched broad any false call | Equal-weight any false call | Weighted any false call (95% CI) |','|---|---:|---:|---:|---:|']
for scenario in ['noise_only','projection_alt_energy_null','energy_alt_projection_null']:
 d=pool[pool.scenario.eq(scenario)].groupby('method')[['families','any_null_families']].sum();n=int(d.loc['weighted_multiscale','families']);k=int(d.loc['weighted_multiscale','any_null_families']);lo,hi=wilson(k,n)
 report.append(f'| {scenario} | {n} | {int(d.loc["matched_broad","any_null_families"])} | {int(d.loc["multiscale","any_null_families"])} | {k}/{n} ({100*k/n:.2f}%; {100*lo:.2f}–{100*hi:.2f}%) |')
report+=['',
'The ramp stress uses independent exchangeable positive slopes for baseline and response, giving joint sign symmetry of all matched-window log-RMS differences after separate demeaning. The converse uses independently signed response templates with centrally symmetric noise, giving the whole-trial projection null jointly over windows. The global multiscale null is the intersection of window-specific component-union nulls. Bonferroni requires valid component/window p-values but no cross-window independence; across-contact BH retains its own conditions. No mixed-window construction is mislabeled as globally null.','',
'## Public baseline-injection characterization','',
'The 960 source-based families use two annotation-blind prestimulus fixtures: MAYO02/LG1-LG2/LAS1 and UMCU20/HF14-HF15/F01, ten native trials each. The initial ≥24-trial criterion proved infeasible from source metadata and was amended before outcomes to ten native trials, with n=8/10 selected without replacement. Disjoint prestimulus blocks were resampled to 500 Hz, swapped and whole-trial-sign randomized before injection. The same finite source recordings are reused across contacts and scenarios; these are conditional semisynthetic experiments, not 960 independent clinical recordings. Their imposed projection null is not a natural-noise false-positive estimate.','',
'| Native trials used | Families with injection | Matched broad recovery | Equal-weight recovery | Weighted recovery |','|---|---:|---:|---:|---:|']
r=read(ROOT/'weighted_real_baseline/pooled_by_trials.csv')
for n in [8,10]:
 d=r[(r.scenario=='real_baseline_power')&(r.n_trials==n)].set_index('method')
 report.append('| '+ ' | '.join([str(n),'320']+[f'{100*d.loc[m,"target_rate"]:.2f}%' for m in ['matched_broad','multiscale','weighted_multiscale']])+' |')
report+=['',
'Complete positive/negative and morphology-specific cells, including zero rates, are retained in each `all_cells.csv`; all contact/window p-values, seeds, availability and realized intervals remain in compressed row-level results. No family failed, and no production warning was recorded. All tested contact/window evaluations were available.','',
'## Verification, timing and attribution','',
'31 focused tests passed. They cover exact core concordance, whole-record polarity reflection including Monte Carlo paths, common clean-trial masks, missing-window conservatism, fixed weights, bank/contact ordering, malformed inputs and both eight-trial resolution bounds. Two deliberate identical-trial tests emit the existing descriptive t-statistic precision-loss warning; the randomization calculations and all production runs are unaffected. This count is a targeted subset, not a claimed total project suite.','',
'The audit independently recomputes BH arithmetic from saved contact p-values, verifies all scientific output hashes, checks complete grids and disjoint confirmation seeds, and exactly replays deterministic raw-data scenarios from every scenario/trial-count combination. The original run manifests accidentally included progress-log hashes while logs were still growing; they remain preserved, and the amended runner freezes only explicit scientific artifacts. This bookkeeping repair did not alter prior numerical results. Exact execution code snapshots are retained.','',
'CRP supplies the semi-normalized projection representation (Miller et al., 2023, https://doi.org/10.1371/journal.pcbi.1011105). ER-detect also includes fixed-window CRP projection t-threshold detection, Python/BIDS workflow and selectable polarity (van Boom et al., 2025, https://doi.org/10.1016/j.jneumeth.2025.110389). ERPy combines its existing whole-trial sign-randomized projection and separately demeaned log-RMS conjunction with standard Bonferroni weighting. This is not a new statistical theorem, a first projection detector, or evidence of clinical utility.','']
for sub,name in [(ROOT,'Original 2,000 generated families'),(ROOT/'real_baseline','960 source-based families'),(ROOT/'weighted_confirmation','Fresh 2,000-family confirmation')]:
 m=json.loads((sub/'run_manifest.json').read_text());report.append(f'- {name}: {m["elapsed_wall_seconds"]:.2f} s for family evaluation, {m["worker_processes"]} workers, Python {m["python"]}, NumPy {m["numpy"]}, {m["platform"]}, single BLAS thread per worker. Measured workload timing excludes summary/report generation and is not a scalability benchmark.')
(ROOT/'RESULTS_AND_DECISION.md').write_text('\n'.join(report)+'\n')
(ROOT/'verification.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'all_pass':all(c['pass_'] for c in checks),'focused_tests_passed':31,'original_log_hash_caveat':'Original manifests are preserved; growing progress logs were excluded from scientific artifact verification. Future runner uses explicit result artifacts.'},indent=2))
print('ALL CHECKS PASSED',len(checks))
