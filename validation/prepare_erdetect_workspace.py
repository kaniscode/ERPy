#!/usr/bin/env python3
"""Verify bundled public evidence and initialize an independent rerun directory."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent


def _atomic_write(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=target.parent, prefix='.' + target.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def initialize(destination: Path, frozen_predictions: bool = False) -> dict:
    manifest = json.loads((HERE / 'erdetect_reference/artifact_manifest.json').read_text())
    for entry in manifest:
        source = HERE / entry['path']
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f"Bundled artifact is missing or changed: {entry['path']}")
    # Preflight all destinations before making changes. A late conflict must
    # not leave a workspace partly restored from a different evidence bundle.
    writes = {}
    entries = {entry['path']: entry for entry in manifest}
    for source in (HERE / 'erdetect_reference/source_metadata').rglob('*'):
        if not source.is_file():
            continue
        target = destination / 'source_metadata' / source.relative_to(HERE / 'erdetect_reference/source_metadata')
        relative = str(source.relative_to(HERE))
        payload = source.read_bytes()
        if relative not in entries or hashlib.sha256(payload).hexdigest() != entries[relative]['sha256']:
            raise ValueError(f"Metadata is unmanifested or changed: {relative}")
        writes[target] = payload
    # The scorer checks Mayo coordinate units against the data-side metadata.
    for source in (HERE / 'erdetect_reference/source_metadata/dataset').glob('sub-MAYO*/ses-*/ieeg/*'):
        if source.suffix not in {'.tsv', '.json'}:
            continue
        target = destination / 'data/ds004774' / source.relative_to(HERE / 'erdetect_reference/source_metadata/dataset')
        writes[target] = writes[destination / 'source_metadata' / source.relative_to(HERE / 'erdetect_reference/source_metadata')]
    if frozen_predictions:
        with tarfile.open(HERE / 'frozen_results/erdetect/prediction_files.tar.gz', 'r:gz') as archive:
            for member in archive.getmembers():
                relative = Path(member.name)
                if not member.isfile() or relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'predictions':
                    raise ValueError('Unexpected prediction archive member')
                payload = archive.extractfile(member).read()
                target = destination / relative
                if target in writes:
                    raise ValueError(f"Duplicate prediction archive member: {member.name}")
                writes[target] = payload
    for target, payload in writes.items():
        if target.exists() and (not target.is_file() or target.read_bytes() != payload):
            raise ValueError(f"Existing workspace content differs; use a fresh workspace: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    for target, payload in writes.items():
        if not target.exists():
            _atomic_write(target, payload)
    result = {'verified_bundled_files': len(manifest), 'frozen_predictions_restored': frozen_predictions}
    print(json.dumps(result, indent=2))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--frozen-predictions', action='store_true', help='Restore exact prediction CSVs for a scoring-only reproduction; omit for a detector rerun')
    args = parser.parse_args()
    initialize(args.destination, args.frozen_predictions)
