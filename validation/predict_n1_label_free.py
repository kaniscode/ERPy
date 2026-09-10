#!/usr/bin/env python3
"""Generate fixed N1 calls from pinned full-family inputs without reading labels."""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ERPy.n1 import N1FeatureConfig, n1_feature_vector, n1_inference_config
from ERPy.n1_detection import _apply_complete_family

ROOT = Path(__file__).resolve().parents[1]
KEY = ['subject', 'stimpair', 'channel']
PROTOCOL_SHA256 = '7b8170a51b118078c5567592b3cd1be12ef70db008cd0b02d422a629df201d21'
RECORDS_SHA256 = '773927fb3bf76f3a1a249e5092f077ba25a3046e4c0385deb442849bc771b48a'
FEATURES_SHA256 = '18177ea77d9dbd44afa7fa146aaef99b2fab31ef5a6c088aea9df1bcba2cbcaf'
FEATURE_MANIFEST_SHA256 = '420e2b826da3699d059e9ae5de7bb4fda6134d61715b5529c2d3187bbf7db63d'
INFERENCE_COLUMNS = KEY+['configuration','prediction_present','epoch_sha256','random_seed',
    'n_trials_total','n_trials_clean','p_reproducibility','p_energy','p_joint','q_joint','family_size']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(frame):
    frame = frame.copy()
    for column in ['subject', 'channel']:
        frame[column] = frame[column].str.strip().str.upper()
    frame['stimpair'] = frame.stimpair.map(lambda value: '-'.join(sorted(
        part.strip().upper() for part in value.split('-'))))
    if frame.duplicated(KEY).any():
        raise ValueError('Duplicate canonical source identity')
    return frame


def load_checked_inputs(protocol):
    if sha(protocol) != PROTOCOL_SHA256:
        raise ValueError('Fixed N1 protocol identity mismatch')
    records = ROOT/'validation/frozen_results/erdetect/record_predictions_and_labels.csv.gz'
    features = ROOT/'validation/frozen_results/n1/features/n1_features.csv.gz'
    fm = features.with_name('feature_manifest.json')
    for path, expected in [(records, RECORDS_SHA256),(features, FEATURES_SHA256),(fm, FEATURE_MANIFEST_SHA256)]:
        if sha(path) != expected:
            raise ValueError(f'Frozen input identity mismatch: {path.name}')
    manifest = json.loads(fm.read_text())
    if (manifest['output_sha256'] != FEATURES_SHA256 or manifest['annotation_inputs']
            or manifest['archived_call_inputs']
            or manifest['config'] != json.loads(json.dumps(asdict(N1FeatureConfig())))):
        raise ValueError('Feature extraction contract mismatch')
    cp = ROOT/'validation/frozen_results/erdetect/cache_safeguard_verification.json'
    artifacts = json.loads((ROOT/'validation/erdetect_reference/artifact_manifest.json').read_text())
    entry = next(x for x in artifacts if x['path']=='frozen_results/erdetect/cache_safeguard_verification.json')
    if sha(cp) != entry['sha256'] or cp.stat().st_size != entry['bytes']:
        raise ValueError('Frozen configuration receipt identity mismatch')
    fingerprint = json.loads(cp.read_text())['prediction_fingerprint']
    if (fingerprint['configurations']['early_10_90ms'] != json.loads(json.dumps(asdict(n1_inference_config())))
            or fingerprint['normalization']['baseline_s'] != [-.5,-.02]):
        raise ValueError('Early inference configuration mismatch')
    # Explicit column whitelist: human votes, eligibility, archived/model calls
    # never enter this phase, even though they coexist in the source archive.
    inference = pd.read_csv(records, usecols=INFERENCE_COLUMNS, dtype={'random_seed':'string'})
    inference = canonical(inference.loc[inference.configuration.eq('early_10_90ms') & inference.prediction_present])
    morphology = canonical(pd.read_csv(features))
    if len(inference)!=36016 or len(morphology)!=36016:
        raise ValueError('Full retained contact universe changed')
    combined = inference.merge(morphology,on=KEY,validate='one_to_one',how='outer',
                               suffixes=('','_feature'),indicator=True)
    if not combined._merge.eq('both').all() or not combined.epoch_sha256.eq(combined.epoch_sha256_feature).all():
        raise ValueError('Feature/inference identity or source epoch mismatch')
    return combined, dict(records=RECORDS_SHA256,features=FEATURES_SHA256,
                         feature_manifest=FEATURE_MANIFEST_SHA256,configuration_receipt=entry['sha256'])


def execute(output, protocol):
    output, protocol = Path(output), Path(protocol)
    if output.exists():
        raise ValueError('Use a new output directory; preserve all prior results')
    frame, inputs = load_checked_inputs(protocol)
    sources = [Path(__file__),ROOT/'ERPy/n1_detection.py',ROOT/'ERPy/n1.py',ROOT/'ERPy/inference.py']
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    rows, families = [], []
    for (subject,pair), family in frame.groupby(['subject','stimpair'],sort=True):
        finite = np.isfinite(family.p_joint)
        if family.family_size.nunique()!=1 or family.family_size.iloc[0]!=finite.sum():
            raise ValueError('Incomplete original finite inference family')
        if not np.allclose(family.loc[finite,'p_joint'],np.maximum(
                family.loc[finite,'p_reproducibility'],family.loc[finite,'p_energy']),rtol=0,atol=1e-15):
            raise ValueError('Joint p does not match the original component conjunction')
        prepared = []
        for item in family.to_dict('records'):
            if item['feature_available']:
                n1_feature_vector(item)  # Current contract, units and numerical domains.
            if not isinstance(item['random_seed'],str) or not item['random_seed'].isdecimal():
                raise ValueError('Exact seed text unavailable')
            if item['n_feature_trials'] > item['n_trials_total']:
                raise ValueError('Invalid source trial counts')
            prepared.append(dict(subject=subject,site='Mayo' if subject.startswith('MAYO') else 'UMCU',
                stimpair=pair,channel=item['channel'],configuration='early_10_90ms',
                epoch_sha256=item['epoch_sha256'],random_seed=item['random_seed'],
                n_trials_total=item['n_trials_total'],n_trials_clean=item['n_trials_clean'],
                features=item,p_reproducibility=item['p_reproducibility'],p_energy=item['p_energy'],p_joint=item['p_joint']))
        results = _apply_complete_family(prepared)
        for result, old_q in zip(results,family.q_joint):
            if not np.isclose(result['original_q_joint'],old_q,rtol=0,atol=1e-14,equal_nan=True):
                raise ValueError('Reconstructed original BH differs from the saved complete family')
            features = result.pop('features')
            result.update({key:features.get(key) for key in [
                'feature_available','feature_reason','n_feature_trials','negative_peak_present',
                'negative_peak_uv','negative_peak_z','baseline_std_uv','peak_latency_ms',
                'realized_response_start_ms','realized_response_end_ms']})
            result['measured_peak_latency_ms'] = features.get('peak_latency_ms') if features['negative_peak_present'] else None
            result['trial_provenance_scope'] = 'matching hashed source epoch and declared policies; historical clean indices unavailable'
            rows.append(result)
        families.append(dict(subject=subject,stimpair=pair,retained_contacts=len(family),
            original_finite_family=int(finite.sum()),primary_calls=sum(x['detected'] for x in results),
            ablation_calls=sum(x['ablation_detected'] for x in results)))
    if len(families)!=577:
        raise ValueError('Original family universe changed')
    if source_hashes != {str(p.relative_to(ROOT)):sha(p) for p in sources}:
        raise ValueError('Rule sources changed during generation')
    output.mkdir(parents=True)
    table = pd.DataFrame(rows).sort_values(KEY)
    table.to_csv(output/'contact_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(families).to_csv(output/'families.csv',index=False)
    receipt = dict(status='passed',protocol_sha256=PROTOCOL_SHA256,input_hashes=inputs,
        source_hashes=source_hashes,records=len(table),families=len(families),
        finite_inference=int(np.isfinite(table.p_joint).sum()),inference_columns_read=INFERENCE_COLUMNS,
        human_reference_columns_read=[],archived_or_model_call_columns_read=[],
        outputs={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'prediction_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','records','families','finite_inference']}))
    return table


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args()
    execute(args.output,args.protocol)
