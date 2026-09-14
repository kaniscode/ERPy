"""Checked adapter for the released ds004774 N1 example records.

This is a limited-provenance archival path. It verifies table bytes, complete
trained settings, separate row identities, epoch hashes and recorded trial
counts. Historical tables do not retain clean-trial indices: none are invented.
For new recordings use ERPy.n1.prepare_n1_hybrid_inputs, which retains both masks.
"""
from dataclasses import asdict
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ERPy.n1 import N1InferenceEvidence, n1_feature_vector, n1_inference_config
KEY = ['subject', 'stimpair', 'channel']
FEATURE_MANIFEST_SHA256 = '420e2b826da3699d059e9ae5de7bb4fda6134d61715b5529c2d3187bbf7db63d'
RECORDS_SHA256 = '773927fb3bf76f3a1a249e5092f077ba25a3046e4c0385deb442849bc771b48a'
CONFIGURATION = 'early_10_90ms'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_channel(value: str) -> str:
    return str(value).strip().upper()


def canonical_pair(value: str) -> str:
    parts = [canonical_channel(part) for part in str(value).split("-")]
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        raise ValueError(f"Invalid stimulation pair: {value!r}")
    return "-".join(sorted(parts))


def subject_of(value: str) -> str:
    match = re.search(r"(?:sub-)?((?:MAYO|UMCU)\d+)", str(value), re.I)
    if not match:
        raise ValueError(f"Unrecognized subject: {value!r}")
    return match[1].upper()


def site_of(subject: str) -> str:
    return "Mayo" if subject.startswith("MAYO") else "UMCU"


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["subject"] = frame.subject.map(subject_of)
    frame["channel"] = frame.channel.map(canonical_channel)
    frame["stimpair"] = frame.stimpair.map(canonical_pair)
    return frame


def require_unique(frame: pd.DataFrame, keys: list[str], description: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate canonical {description}; refusing a many-to-many join")



def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    paths = set()
    for group in ("files", "external_inputs"):
        for item in manifest[group]:
            relative = item["path"]
            if relative in paths:
                raise ValueError(f"Duplicate artifact identity: {relative}")
            paths.add(relative)
            path = root / relative
            # External input is the adjacent, already public frozen evaluation.
            if not path.resolve().is_relative_to(root.parent):
                raise ValueError(f"Artifact escapes the frozen-results folder: {relative}")
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing or linked artifact: {relative}")
            with path.open("rb") as stream:
                hasher = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
                digest = hasher.hexdigest()
            if path.stat().st_size != item["bytes"] or digest != item["sha256"]:
                raise ValueError(f"Artifact identity mismatch: {relative}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name != "artifact_manifest.json"}
    expected = {item["path"] for item in manifest["files"]}
    if actual != expected:
        raise ValueError("N1 evidence contains missing or unlisted files")
    return {"status": "verified", "files": len(manifest["files"]),
            "external_inputs": len(manifest["external_inputs"])}



class FrozenN1Inputs:
    """Load and validate the released tables once, then bind selected records.

    ``prepare(feature_row, inference_row)`` verifies both rows against the
    immutable inputs; it cannot relabel broad-window values as early evidence.
    The returned pair can be passed directly to ``model.predict(*inputs)``.
    """
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[2]/'validation/frozen_results/n1').resolve()
        verify(self.root)
        fm = self.root/'features/feature_manifest.json'
        records = self.root/'../erdetect/record_predictions_and_labels.csv.gz'
        if _sha(fm) != FEATURE_MANIFEST_SHA256 or _sha(records) != RECORDS_SHA256:
            raise ValueError('Unsupported frozen N1 input identity')
        manifest = json.loads(fm.read_text())
        feature_path = self.root/'features/n1_features.csv.gz'
        if _sha(feature_path) != manifest['output_sha256']:
            raise ValueError('Frozen feature output identity mismatch')
        self.feature_sha256 = manifest['output_sha256']
        # The independently retained original predictor receipt states every
        # resolved configuration value; verify its immutable public checksum.
        artifact_manifest = json.loads((self.root.parents[1]/'erdetect_reference/artifact_manifest.json').read_text())
        relative = 'frozen_results/erdetect/cache_safeguard_verification.json'
        entry = next(item for item in artifact_manifest if item['path'] == relative)
        cp = self.root.parent/'erdetect/cache_safeguard_verification.json'
        if _sha(cp) != entry['sha256'] or cp.stat().st_size != entry['bytes']:
            raise ValueError('Frozen inference configuration receipt identity mismatch')
        saved = json.loads(cp.read_text())['prediction_fingerprint']
        expected = json.loads(json.dumps(asdict(n1_inference_config())))
        if saved['configurations'][CONFIGURATION] != expected:
            raise ValueError('Frozen early inference settings differ from the trained contract')
        if saved['normalization']['baseline_s'] != [-.5, -.02]:
            raise ValueError('Frozen trial normalization differs from the trained contract')
        self.config_receipt_sha256 = entry['sha256']
        features = canonicalize(pd.read_csv(feature_path, low_memory=False))
        inference = canonicalize(pd.read_csv(records, low_memory=False,
                                             dtype={'random_seed': 'string'}))
        inference = inference.loc[inference.configuration.eq(CONFIGURATION)]
        require_unique(features, KEY, 'frozen N1 features')
        require_unique(inference, KEY, 'frozen early inference')
        self.features = features.set_index(KEY, drop=False)
        self.inference = inference.set_index(KEY, drop=False)

    def prepare(self, feature_row, inference_row):
        feature_row, inference_row = dict(feature_row), dict(inference_row)
        if inference_row.get('configuration') != CONFIGURATION:
            raise ValueError('Hybrid inference requires early_10_90ms, not another window')
        def identity(row):
            return (subject_of(row['subject']), canonical_pair(row['stimpair']),
                    canonical_channel(row['channel']))
        key = identity(feature_row)
        if key != identity(inference_row):
            raise ValueError('Feature and inference participant/pair/channel identity mismatch')
        try:
            feature = self.features.loc[key].to_dict()
            inference = self.inference.loc[key].to_dict()
        except KeyError as exc:
            raise ValueError('Record absent from the verified frozen inputs') from exc
        epochs = [feature_row.get('epoch_sha256'), inference_row.get('epoch_sha256'),
                  feature['epoch_sha256'], inference['epoch_sha256']]
        if any(not isinstance(x, str) or len(x) != 64 for x in epochs) or len(set(epochs)) != 1:
            raise ValueError('Feature and inference source epoch identity mismatch')
        if not np.array_equal(n1_feature_vector(feature_row), n1_feature_vector(feature)):
            raise ValueError('Feature values differ from the verified frozen record')
        if bool(feature_row.get('negative_peak_present')) != bool(feature['negative_peak_present']):
            raise ValueError('Peak status differs from the verified frozen record')
        for name in ['p_reproducibility', 'p_energy', 'rms_ratio_db',
                     'n_trials_total', 'n_trials_clean']:
            if name not in inference_row or inference_row[name] != inference[name]:
                raise ValueError(f'{name} differs from the verified early inference record')
        # The outer annotation table has missing records; default pandas float
        # inference would round these 63-bit seeds. Accept only the exact text.
        seed = inference_row.get('random_seed')
        if not isinstance(seed, str) or not seed.isdecimal() or seed != inference['random_seed']:
            raise ValueError('random_seed differs from the exact frozen seed text')
        total = int(inference['n_trials_total'])
        if not 8 <= feature['n_feature_trials'] <= total or not 8 <= inference['n_trials_clean'] <= total:
            raise ValueError('Recorded finite-trial counts violate the N1 contract')
        provenance = dict(
            kind='verified_frozen_source_records', subject=key[0], stimpair=key[1], channel=key[2],
            epoch_sha256=epochs[0], n_trials_total=total,
            n_feature_trials=int(feature['n_feature_trials']),
            n_inference_trials=int(inference['n_trials_clean']),
            feature_clean_trial_indices=None, inference_clean_trial_indices=None,
            trial_provenance_scope='shared hashed source epoch and original declared finite-selection policies; historical clean indices were not archived',
            feature_selection='finite across full rounded half-open N1 feature/normalization windows',
            inference_selection='finite across effective response and latest sample-matched baseline',
            normalization_window_s=[-.5, -.02],
            normalization='per-trial nanmedian; half-open rounded sample slice',
            feature_table_sha256=self.feature_sha256, inference_table_sha256=RECORDS_SHA256,
            configuration_receipt_sha256=self.config_receipt_sha256,
        )
        return N1InferenceEvidence._bind(feature, inference,
                                        n1_inference_config(int(inference['random_seed'])), provenance)
