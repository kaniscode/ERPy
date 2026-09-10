#!/usr/bin/env python3
"""Run declared ds004774 reproduction stages without changing detector settings."""
from __future__ import annotations
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MAYO = ['MAYO01', 'MAYO02', 'MAYO03', 'MAYO04', 'MAYO05']
UMCU = ['UMCU20', 'UMCU21', 'UMCU22', 'UMCU23', 'UMCU25', 'UMCU26', 'UMCU59', 'UMCU62', 'UMCU67']
STAGES = ['prepare', 'download', 'extract', 'predict', 'reconstruction', 'score', 'coverage', 'figure']

def commands(stage: str, root: Path, workers: int, frozen_predictions: bool):
    root = root.resolve()
    metadata = root / 'source_metadata'
    inventory = metadata / 's3_object_inventory.json'
    subject_flags = lambda subjects: [arg for subject in subjects for arg in ['--subject', subject]]
    prefix = lambda script: [sys.executable, str(HERE / script)]
    plans = {
        'prepare': [prefix('prepare_erdetect_workspace.py') + [str(root)] + (['--frozen-predictions'] if frozen_predictions else [])],
        'download': [prefix('download_erdetect_public.py') + [str(inventory), str(root / 'data')] + subject_flags(MAYO) + ['--workers', str(workers)],
                     prefix('download_erdetect_public.py') + [str(inventory), str(root / 'data')] + subject_flags(MAYO + UMCU) + ['--archived-calls', '--workers', str(workers)]],
        'extract': [prefix('erdetect_mayo_data.py') + [str(root / 'data/ds004774'), str(root / 'epochs')] + subject_flags(MAYO),
                    prefix('erdetect_umcu_data.py') + ['--metadata-root', str(metadata), '--output-root', str(root / 'epochs'), '--workers', str(workers)]],
        'predict': [prefix('predict_erdetect_epochs.py') + [str(root / 'epochs'), str(root / 'predictions'), '--workers', str(workers)]],
        'score': [prefix('score_erdetect_validation.py') + ['--root', str(root), '--output', str(root / 'scoring'), '--pair-manifest', str(metadata / 'archived_pair_exclusions.csv')]],
        'coverage': [prefix('audit_annotation_coverage.py') + ['--records', str(root / 'scoring/record_predictions_and_labels.csv'), '--output', str(root / 'coverage_audit')]],
        'reconstruction': [prefix('check_erdetect_reconstruction.py') + [str(root), str(root / 'reconstruction/mayo')] + subject_flags(MAYO),
                           prefix('check_erdetect_umcu_reconstruction.py') + [str(root), '--output', str(root / 'reconstruction/umcu.json')]],
        'figure': [prefix('plot_erdetect_validation.py') + ['--metrics', str(root / 'scoring/metrics.csv'), '--output', str(root / 'figures/fig06_external_validation')]],
    }
    return plans[stage]

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=STAGES + ['all'], action='append', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--frozen-predictions', action='store_true')
    p.add_argument('--extraction-python', type=Path, help='Python from the separate extraction environment (used for extract only)')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if a.workers < 1:p.error('workers must be positive')
    if 'all' in a.stage and len(a.stage) != 1:
        p.error('--stage all cannot be combined with individual stages')
    stages = STAGES if a.stage == ['all'] else a.stage
    if a.frozen_predictions and 'predict' in stages:
        p.error('--frozen-predictions is for scoring-only reproduction; omit predict or use a fresh detector workspace')
    for stage in stages:
        for command in commands(stage, a.root, a.workers, a.frozen_predictions):
            if stage == 'extract' and a.extraction_python:
                command[0] = str(a.extraction_python.resolve())
            print(shlex.join(command), flush=True)
            if not a.dry_run:subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
