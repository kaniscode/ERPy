#!/usr/bin/env python3
"""Plot saved default-detector calls, N1 annotations and selected public traces."""
from pathlib import Path
import argparse
import hashlib
import json
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image

TEAL = '#007C70'
AMBER = '#A56418'
INK = '#23343F'
GRAY = '#D8DEE3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probability(value):
    return f'{value:.3g}'


def plot(analysis, output, journal):
    if output.exists():
        raise ValueError('Use a new output directory; earlier artwork is preserved.')
    output.mkdir(parents=True)
    width_mm = 174 if journal == 'neuroinformatics' else 180
    height_mm = width_mm
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
                         'ytick.color': INK, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 2, 'lines.linewidth': 2, 'savefig.facecolor': 'white'})
    summary = json.loads((analysis / 'summary.json').read_text())
    cells = pd.read_csv(analysis / 'crossclassification.csv').set_index('cell')
    rates = pd.read_csv(analysis / 'annotation_group_rates.csv').set_index('annotation_group')
    examples = pd.read_csv(analysis / 'exemplar_selection.csv').set_index('example')
    traces = pd.read_csv(analysis / 'exemplar_traces.csv.gz')
    assert summary['configuration'] == 'broad_15_300ms'
    assert cells.weighted_record_equivalents.sum() == summary['cohort']['available_records']
    assert int(rates.n_records.sum()) == summary['cohort']['available_records']
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    letter = lambda value: value.lower() if journal == 'neuroinformatics' else value
    fig.text(.025, .967, f'{letter("A")}   Default calls and N1 votes', fontsize=10, weight='bold')
    fig.text(.025, .939, '32,121 records · 13 participants', fontsize=8)
    matrix = fig.add_axes([.125, .682, .305, .205])
    matrix.set_xlim(-.5, 1.5); matrix.set_ylim(1.5, -.5)
    for (row, col), cell in { (0,0): 'tp', (0,1): 'fp', (1,0): 'fn', (1,1): 'tn' }.items():
        value = cells.loc[cell]
        same = row == col
        matrix.add_patch(Rectangle((col-.5, row-.5), 1, 1,
                                   facecolor='#E7F2EF' if same else '#FCF0DD',
                                   edgecolor='white', linewidth=2))
        matrix.text(col, row-.10, f'{value.weighted_record_equivalents:,.1f}',
                    ha='center', va='center', fontsize=10, weight='bold')
        matrix.text(col, row+.22, f'{100*value.fraction_of_available_records:.2f}%',
                    ha='center', va='center', fontsize=8)
    matrix.set_xticks([0, 1], ['N1+', 'N1−'])
    matrix.set_yticks([0, 1], ['Call +', 'Call −'])
    matrix.xaxis.tick_top(); matrix.xaxis.set_label_position('top')
    matrix.set_xlabel('')
    matrix.set_ylabel('Default detector', labelpad=5)
    matrix.tick_params(axis='both', length=0, pad=6)
    for spine in matrix.spines.values(): spine.set_visible(False)
    agreement = summary['metrics']['agreement']
    fig.text(.025, .655, 'Vote-weighted units; percentages of all records.', fontsize=8)
    fig.text(.025, .631,
             f'Agreement {100*agreement["value"]:.2f}% (95% CI {100*agreement["low"]:.2f}–{100*agreement["high"]:.2f}).', fontsize=8)

    fig.text(.50, .967, f'{letter("B")}   Calls by annotation pattern', fontsize=10, weight='bold')
    fig.text(.50, .939, 'Points and bars: detection rate and 95% CI', fontsize=8)
    groups = [('Unanimous N1-positive (>=2 raters)', 'Unanimous N1+'),
              ('Mixed N1 ratings (>=2 raters)', 'Mixed N1 votes'),
              ('Unanimous N1-negative (>=2 raters)', 'Unanimous N1−'),
              ('Single-rater N1-positive', 'One rater, N1+'),
              ('Single-rater N1-negative', 'One rater, N1−')]
    rates.index = rates.index.str.strip()
    rate_ax = fig.add_axes([.724, .682, .249, .219])
    positions = np.arange(4, -1, -1)
    for y, (name, _) in zip(positions, groups):
        row = rates.loc[name]
        point, low, high = row[['call_rate', 'low', 'high']].to_numpy(float) * 100
        rate_ax.errorbar(point, y, xerr=[[point-low], [high-point]], fmt='o', color=TEAL,
                         markersize=4, markeredgewidth=2, elinewidth=2, capsize=3, capthick=2)
    rate_ax.set_yticks(positions, [f'{label}\n{int(rates.loc[name,"n_records"]):,} records' for name, label in groups])
    rate_ax.set_ylim(-.55, 4.55); rate_ax.set_xlim(-1, 101)
    rate_ax.set_xticks([0, 50, 100]); rate_ax.set_xlabel('Default detections (%)', labelpad=6)
    rate_ax.tick_params(axis='y', length=0, pad=8)
    rate_ax.tick_params(axis='x', width=2, length=3)
    rate_ax.spines['left'].set_visible(False)
    rate_ax.grid(axis='x', linewidth=2, color='#EEF0F2'); rate_ax.set_axisbelow(True)
    fig.text(.50, .607, 'Unanimous/mixed groups have ≥2 raters.', fontsize=8)
    fig.text(.025, .607, 'N1− does not mean no evoked response.', fontsize=8)
    contrast = pd.read_csv(analysis / 'annotation_group_contrasts.csv').iloc[0]
    assert contrast.group_1.strip() == groups[0][0] and contrast.group_0.strip() == groups[2][0]
    fig.text(.025, .582, f'Unanimous N1+ minus N1− call rate: {100*contrast.call_rate_difference:.2f} pp (95% CI {100*contrast.low:.2f}–{100*contrast.high:.2f}; excludes 0).', fontsize=8)

    # Each trace was selected before loading its waveform, by median log10(q)
    # within its predefined agreement/disagreement stratum.
    panel_specs = [('both_N1_and_default', 'Both N1+ and default+', .08, .335, 'C', TEAL),
                   ('neither_N1_nor_default', 'Both N1− and default−', .58, .335, 'D', '#566B78'),
                   ('default_without_N1', 'Default+ with N1− votes', .08, .065, 'E', AMBER),
                   ('N1_without_default', 'N1+ votes with default−', .58, .065, 'F', AMBER)]
    for key, title, x, bottom, label, color in panel_specs:
        row = examples.loc[key]
        vals = traces.loc[traces.example.eq(key)].copy()
        assert not vals.empty and int(row.n_raters) >= 2
        metadata_x = x - .055
        fig.text(metadata_x, bottom+.226, f'{letter(label)}   {title}', fontsize=9, weight='bold')
        fig.text(metadata_x, bottom+.202,
                 f'{row.subject} · stim {row.stimpair} → rec {row.channel}', fontsize=8)
        fig.text(metadata_x, bottom+.178,
                 f'N1 votes {int(row.n1_votes)}/{int(row.n_raters)} · n={int(row.n_trials_clean)} · q={probability(row.q_joint)}', fontsize=8)
        ax = fig.add_axes([x, bottom, .382, .150])
        t = vals.time_s.to_numpy(float) * 1000
        mean = vals.mean_uv.to_numpy(float); sem = vals.sem_uv.to_numpy(float)
        visible = (t < 0) | (t >= 15)
        mean = np.where(visible, mean, np.nan); sem = np.where(visible, sem, np.nan)
        ax.axvspan(0, 15, color='#DFE3E6', linewidth=0, zorder=0)
        ax.axvspan(15, 300, color='#EEF6F3', linewidth=0, zorder=0)
        ax.axhline(0, color='#A9B3BB', linewidth=2, zorder=1)
        ax.fill_between(t, mean-sem, mean+sem, color=color, alpha=.18, linewidth=0, zorder=2)
        ax.plot(t, mean, color=color, linewidth=2, zorder=3)
        ax.set_xlim(-100, 350); ax.set_xticks([-100, 0, 150, 300])
        if bottom < .1:
            ax.set_xlabel('Time after stimulation (ms)', labelpad=4)
        ax.set_ylabel('µV', labelpad=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.tick_params(width=2, length=3)
    stem = output / 'fig06_external_validation'
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); frame = fig.bbox
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text(): continue
        # Some automatic ticks are generated outside limits and are not drawn.
        box = artist.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > frame.x1+.5 or box.y1 > frame.y1+.5:
            outside.append(artist.get_text())
    for ext in ['svg', 'pdf', 'png']:
        fig.savefig(stem.with_suffix('.'+ext), dpi=600)
    with Image.open(stem.with_suffix('.png')) as image:
        image.convert('RGB').save(stem.with_suffix('.tiff'), dpi=(600,600), compression='tiff_lzw')
    fig.savefig(output/'publication_scale.png', dpi=180)
    inputs = ['summary.json', 'crossclassification.csv', 'annotation_group_rates.csv',
              'exemplar_selection.csv', 'exemplar_traces.csv.gz', 'exemplar_source_provenance.json', 'annotation_group_contrasts.csv']
    manifest = {'journal': journal, 'width_mm': width_mm, 'height_mm': height_mm,
                'font_min_pt': 8, 'visible_stroke_min_pt': 2, 'default_configuration': 'broad_15_300ms',
                'record_count': 32121, 'participant_count': 13, 'outside_text': outside,
                'generator_sha256': sha(__file__), 'inputs': {name: sha(analysis/name) for name in inputs},
                'files': {p.name: sha(p) for p in output.iterdir() if p.is_file()},
                'visual_inspection_required': True}
    (output/'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    plt.close(fig)
    print(json.dumps({'journal':journal,'outside_text':outside,'output':str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--journal', choices=['jnm','frontiers','neuroinformatics'], required=True)
    args = parser.parse_args()
    plot(args.analysis, args.output, args.journal)
