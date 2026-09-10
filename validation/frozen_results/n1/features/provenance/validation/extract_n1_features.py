#!/usr/bin/env python3
"""Extract annotation-blind N1 morphology from public epoch files."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ERPy.n1 import N1FeatureConfig, extract_n1_features
from predict_erdetect_epochs import load_epochs


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def extract_file(path: Path) -> tuple[list[dict], dict]:
    before = sha(path)
    values, times, names, subject, pair, metadata = load_epochs(path)
    stim = set(pair.split("-"))
    record = {"subject": subject, "stimpair": pair, "epoch_file": path.name,
              "epoch_sha256": before}
    rows = []
    if len(stim) == 2 and stim.issubset(names):
        for index, channel in enumerate(names):
            if channel in stim:
                continue
            features = extract_n1_features(values[:, :, index], times)
            rows.append(dict(subject=subject, stimpair=pair, channel=channel,
                             epoch_sha256=before, **features))
    else:
        record["exclusion"] = "bad_or_missing_stimulation_contact"
    if sha(path) != before:
        raise ValueError("Source epoch changed during feature extraction")
    record["records"] = len(rows)
    return rows, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epochs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory; feature outputs are immutable")
    paths = sorted(args.epochs.glob("*/*.npz"))
    if not paths or args.workers < 1:
        parser.error("Epoch inputs and positive worker count required")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/'ERPy/n1.py', root/'validation/predict_erdetect_epochs.py']
    hashes = {str(p.relative_to(root)): sha(p) for p in sources}
    rows, inputs = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_file, path) for path in paths]
        for count, future in enumerate(as_completed(futures), 1):
            data, record = future.result()
            rows.extend(data); inputs.append(record)
            if count % 25 == 0 or count == len(paths):
                print(f"Extracted {count}/{len(paths)} epoch files", flush=True)
    if hashes != {str(p.relative_to(root)): sha(p) for p in sources}:
        raise ValueError("Feature code changed during extraction")
    frame = pd.DataFrame(rows).sort_values(['subject','stimpair','channel'])
    if frame.duplicated(['subject','stimpair','channel']).any():
        raise ValueError("Duplicate source identities")
    target = args.output/'n1_features.csv.gz'
    frame.to_csv(target,index=False,compression={'method':'gzip','mtime':0})
    manifest = {'protocol_sha256':sha(root/'validation/N1_OPTIMIZATION_PROTOCOL.md'),
                'config':N1FeatureConfig().to_dict(),'code_sha256':hashes,
                'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                'pandas':pd.__version__,'annotation_inputs':[], 'archived_call_inputs':[],
                'records':len(frame),'feature_available':int(frame.feature_available.sum()),
                'inputs':sorted(inputs,key=lambda x:(x['subject'],x['stimpair'])),
                'output_sha256':sha(target)}
    (args.output/'feature_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['records','feature_available','output_sha256']}))


if __name__ == '__main__':
    main()
