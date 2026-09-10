#!/usr/bin/env python3
"""Publication figure for the frozen, nested N1 development comparison."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
from PIL import Image


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    if args.output.exists():raise ValueError('Choose a new figure-output directory')
    args.output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'svg.fonttype':'none',
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':2,'xtick.major.width':2,'ytick.major.width':2,
        'lines.linewidth':2,'lines.markeredgewidth':2,'savefig.facecolor':'white'})
    metrics=pd.read_csv(args.run/'metrics.csv')
    methods=['original_ERPy','archived_ER_detect','N1_logistic_morphology','N1_logistic_hybrid']
    labels=['ERPy, 10–90 ms','ER-detect archive','N1 morphology','N1 hybrid']
    colors={'original_ERPy':'#386b8c','archived_ER_detect':'#50545b',
            'N1_logistic_hybrid':'#007c70','N1_logistic_morphology':'#bc7b27'}
    width_mm=getattr(args,'width_mm',180.)
    height_mm=150*width_mm/180
    fig=plt.figure(figsize=(width_mm/25.4,height_mm/25.4))
    positions=np.arange(len(methods)-1,-1,-1)
    axes=[]
    for col,(metric,title,limits,ticks) in enumerate([
        ('sensitivity','Sensitivity (%)',(45,101),[50,75,100]),
        ('specificity','Specificity (%)',(79,101),[80,90,100]),
        ('ppv','Positive predictive value (%)',(35,101),[40,70,100])]):
        ax=fig.add_axes([.18+col*.275,.665,.235,.255]);axes.append(ax)
        for row,method in enumerate(methods):
            r=metrics[(metrics.method==method)&(metrics.metric==metric)].iloc[0]
            estimate,lo,hi=100*r[['estimate','low','high']].to_numpy(float)
            ax.errorbar(estimate,positions[row],xerr=np.array([[estimate-lo],[hi-estimate]]),
                        fmt='o',color=colors[method],markersize=5,capsize=3,capthick=2,elinewidth=2)
        ax.set_xlim(limits);ax.set_ylim(-.55,len(methods)-.45);ax.set_xticks(ticks)
        ax.set_yticks(positions);ax.set_yticklabels(labels if col==0 else [])
        ax.tick_params(axis='y',length=0,pad=7)
        ax.set_xlabel(title,labelpad=8)
        ax.text(-.10,1.13,chr(65+col).lower() if args.lowercase else chr(65+col),
                transform=ax.transAxes,fontsize=10,fontweight='bold',ha='left')
        ax.grid(axis='x',color='#e8eaec',linewidth=2,zorder=0)
        ax.set_axisbelow(True)
    ax=fig.add_axes([.18,.15,.785,.355]);axes.append(ax)
    for method,label in [('original_ERPy','ERPy, 10–90 ms'),('N1_logistic_morphology','N1 morphology'),
                         ('N1_logistic_hybrid','N1 hybrid')]:
        curve=pd.read_csv(args.run/f'{method}_precision_recall.csv.gz')
        ax.step(curve.recall*100,curve.precision*100,where='post',color=colors[method],label=label,linewidth=2)
    ax.set_xlim(0,100);ax.set_ylim(0,103);ax.set_xticks([0,25,50,75,100]);ax.set_yticks([0,25,50,75,100])
    ax.set_xlabel('Sensitivity (recall, %)',labelpad=5);ax.set_ylabel('Positive predictive value (%)',labelpad=7)
    ax.text(-.10,1.12,'d' if args.lowercase else 'D',transform=ax.transAxes,fontsize=10,fontweight='bold')
    ax.text(0,1.12,'Precision–recall comparison',transform=ax.transAxes,fontsize=9)
    ax.grid(color='#e8eaec',linewidth=2);ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=3,frameon=False,
              columnspacing=1.4,handlelength=2.2,borderaxespad=0)
    stem=args.output/'fig07_n1_development'
    plotted=metrics[metrics.method.isin(methods)&metrics.metric.isin(['sensitivity','specificity','ppv'])]
    if len(plotted)!=12:raise ValueError('Expected twelve operating-point estimates')
    plotted.to_csv(args.output/'fig07_n1_development_data.csv',index=False)
    for ext in ['pdf','svg','png']:
        fig.savefig(stem.with_suffix('.'+ext),dpi=600)
    with Image.open(stem.with_suffix('.png')) as img:
        img.convert('RGB').save(stem.with_suffix('.tiff'),compression='tiff_lzw',dpi=(600,600))
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bbox=fig.bbox
    outside=[]
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():continue
        b=artist.get_window_extent(renderer)
        if b.x0<-.5 or b.y0<-.5 or b.x1>bbox.x1+.5 or b.y1>bbox.y1+.5:outside.append(artist.get_text())
    manifest={'width_mm':width_mm,'height_mm':height_mm,'font_min_pt':8,'visible_stroke_min_pt':2,
        'lowercase_panels':args.lowercase,'input_metrics_sha256':digest(args.run/'metrics.csv'),
        'run_protocol_sha256':digest(args.run/'run_protocol.json'),'generator_sha256':digest(__file__),
        'operating_point_methods':methods,
        'input_curve_sha256':{method:digest(args.run/f'{method}_precision_recall.csv.gz') for method in ['original_ERPy','N1_logistic_morphology','N1_logistic_hybrid']},
        'outside_text':outside,'files':{p.name:digest(p) for p in args.output.iterdir() if p.is_file()},
        'visual_inspection_required':True}
    (args.output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    plt.close(fig)
    if outside:raise ValueError(f'Text outside figure: {outside}')
    print(json.dumps({'files':len(manifest['files']),'outside_text':outside}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width-mm',type=float,choices=[174.,180.],default=180.)
    p.add_argument('--lowercase',action='store_true');main(p.parse_args())
