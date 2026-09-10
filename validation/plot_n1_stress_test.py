#!/usr/bin/env python3
"""Plot the complete fixed N1 injection calibration; no new detector analyses.

Uses recorded independent-family counts for every fixed injection condition.
Target-contact proportions are descriptive because targets share family noise.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from PIL import Image

VARIANTS = {'jnm': (180., False), 'frontiers': (180., False),
            'neuroinformatics': (174., True)}
SCENARIOS = ['early_negative', 'early_positive', 'late_negative', 'residual_artifact']
NAMES = ['Early negative\nN1 target', 'Early positive\nchallenge',
         'Late negative\nchallenge', 'Residual artifact\nchallenge']
COLS = [(100,25), (100,100), (250,25), (250,100)]
TRIALS = [8,12,24]
TEAL = '#007d73'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_data(path):
    summary = json.loads((path/'summary.json').read_text())
    families = pd.read_csv(path/'family_results.csv')
    assert len(families)==480 and families.family_id.nunique()==480
    assert (families.family_size==8).all()
    cells = summary['phenotype_challenge_cells']
    assert len(cells)==48
    rows=[]
    arrays = {key:np.zeros((12,4)) for key in ['primary','ablation']}
    for si,scenario in enumerate(SCENARIOS):
        for ti,n in enumerate(TRIALS):
            for ci,(amp,noise) in enumerate(COLS):
                cell = next(r for r in cells if (r['scenario'],r['n_trials'],r['amplitude_uv'],r['noise_sd_uv'])==(scenario,n,amp,noise))
                frame = families.loc[(families.scenario==scenario)&(families.n_trials==n)&(families.amplitude_uv==amp)&(families.noise_sd_uv==noise)]
                assert len(frame)==5 and frame.target_contacts.sum()==10
                for method in arrays:
                    count = int(frame[method+'_target_calls'].sum())
                    assert count==cell[method]['target_calls']
                    assert cell[method]['target_contacts']==10
                    assert abs(cell[method]['target_call_rate']-count/10)<1e-12
                    assert frame[method+'_null_contact_calls'].sum()==0
                    arrays[method][si*3+ti,ci]=count/10
                    rows.append({'method':method, 'scenario':scenario, 'n_trials':n,
                                 'amplitude_uv':amp,'noise_sd_uv':noise,
                                 'families':5,'target_contacts':10,'target_calls':count})
    null = families.loc[families.scenario=='pure_null']
    assert len(null)==240
    for method in arrays:
        val=summary['pure_null_family_any_call']['overall'][method]
        assert int(null[method+'_any_call'].sum())==val['successes']==0
        assert val['total']==240 and val['ci_low']==0
        # Recheck the already recorded two-sided95% Wilson zero-event bound.
        z=1.959963984540054
        assert abs(val['ci_high']-z*z/(240+z*z))<1e-12
    return summary,arrays,rows

def text_color(rgba):
    rgb=np.array(rgba[:3])
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    lum=float(linear @ np.array([.2126,.7152,.0722]))
    black=(lum+.05)/.05
    white=1.05/(lum+.05)
    return ('black' if black>=white else 'white'),max(black,white)

def render(journal,out,data_dir):
    width,lower=VARIANTS[journal]
    height=172*.76*width/180
    y=lambda value:(value-.24)/.76
    out.mkdir(parents=True,exist_ok=True)
    summary,arrays,rows=read_data(data_dir)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
        'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.linewidth':2,'xtick.major.width':2,'ytick.major.width':2,
        'lines.linewidth':2,'lines.markeredgewidth':2,
        'svg.fonttype':'none','pdf.fonttype':42,
        'svg.hashsalt':'ERPy-fixed-N1-injection-calibration',
        'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(width/25.4,height/25.4))
    fig.text(.035,y(.982),'Fixed synthetic injection stress test',fontsize=10,fontweight='bold',va='top')
    fig.text(.035,y(.947),'240 injection families · 8 contacts per family',fontsize=8,va='top')
    fig.text(.965,y(.947),'Cells: detected targets / 10',fontsize=8,ha='right',va='top')
    cmap=LinearSegmentedColormap.from_list('ERPy_detection',['#f1f5f4',TEAL])
    heat_bounds=[]
    all_annotations=[]
    for pi,(method,left,title) in enumerate([('primary',.25,'Primary N1 detector'),('ablation',.665,'No-floor ablation')]):
        ax=fig.add_axes([left,y(.36),.30,.48/.76])
        ax.imshow(arrays[method],vmin=0,vmax=1,cmap=cmap,aspect='auto',interpolation='nearest')
        ax.set(xticks=[],yticks=np.arange(12),yticklabels=[str(n) for _ in SCENARIOS for n in TRIALS])
        ax.tick_params(axis='y',length=0,pad=6)
        for side in ax.spines.values(): side.set_visible(False)
        for edge in np.arange(.5,11.5,1):
            ax.axhline(edge,color='white',lw=2)
        for edge in [.5,1.5,2.5]: ax.axvline(edge,color='white',lw=2)
        for edge in [2.5,5.5,8.5]: ax.axhline(edge,color='#879791',lw=2)
        for ri in range(12):
            for ci in range(4):
                rate=arrays[method][ri,ci]
                color,contrast=text_color(cmap(rate))
                assert contrast>=4.5
                obj=ax.text(ci,ri,f'{round(10*rate)}/10',ha='center',va='center',fontsize=8,color=color)
                all_annotations.append((obj,ax,ci,ri,contrast))
        fig.text(left-.036,y(.903),'AB'[pi].lower() if lower else 'AB'[pi],fontsize=10,fontweight='bold',va='bottom')
        fig.text(left,y(.903),title,fontsize=9,fontweight='bold',va='bottom')
        fig.text(left+.15,y(.875),'Amplitude / noise SD (µV)',fontsize=8,ha='center',va='bottom')
        for ci,(amp,noise) in enumerate(COLS):
            fig.text(left+(ci+.5)*.30/4,y(.846),f'{amp}/{noise}',fontsize=8,ha='center',va='bottom')
        fig.text(left-.027,y(.846),'Trials',fontsize=8,ha='right',va='bottom')
        if pi==0:
            for si,name in enumerate(NAMES):
                ypos=.36+.48*(1-(3*si+1.5)/12)
                fig.text(.035,y(ypos),name,fontsize=8,ha='left',va='center')
        heat_bounds.append(ax)
    cax=fig.add_axes([.335,y(.308),.545,.015/.76])
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,100),cmap=cmap),cax=cax,orientation='horizontal',ticks=[0,25,50,75,100])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=3,width=2,pad=3)
    cb.set_label('Target-contact call proportion (%)',labelpad=4)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    annotations=[]
    for obj,axis,x,y,contrast in all_annotations:
        box=obj.get_window_extent(renderer)
        lower=axis.transData.transform((x-.5,y+.5)); upper=axis.transData.transform((x+.5,y-.5))
        inside=box.x0>=lower[0] and box.x1<=upper[0] and box.y0>=lower[1] and box.y1<=upper[1]
        assert inside,(obj.get_text(),box.bounds)
        annotations.append({'text':obj.get_text(),'within_own_cell':bool(inside),'contrast_ratio':contrast})
    texts=[]
    for obj in fig.findobj(matplotlib.text.Text):
        if not obj.get_visible() or not obj.get_text().strip(): continue
        box=obj.get_window_extent(renderer)
        texts.append({'text':obj.get_text(),'font_size_pt':obj.get_fontsize(),'bounds_px':list(box.bounds),
                      'within_canvas':bool(box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height)})
    assert all(t['within_canvas'] for t in texts),[t['text'] for t in texts if not t['within_canvas']]
    stem=out/'figS01_synthetic_injection_stress_test'
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.pdf'),metadata={'Creator':'ERPy','CreationDate':None,'ModDate':None})
    fig.savefig(stem.with_suffix('.png'),dpi=600)
    with Image.open(stem.with_suffix('.png')) as img:
        rgb=img.convert('RGB'); pixels=img.size
        rgb.save(stem.with_suffix('.png'),dpi=(600,600))
        rgb.save(stem.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw')
    fig.savefig(out/'publication_scale.png',dpi=160)
    plt.close(fig)
    metadata={'journal':journal,'width_mm':width,'height_mm':height,'dpi':600,
              'pixel_dimensions':pixels,'minimum_text_pt':min(t['font_size_pt'] for t in texts),
              'panel_labels':'ab' if VARIANTS[journal][1] else 'AB',
              'generator_sha256':sha(__file__),
              'source_tables':{p.name:sha(p) for p in [data_dir/'summary.json',data_dir/'family_results.csv']},
              'plotted_cells':rows,
              'cell_annotations':annotations,'marker_bounds':[],'text_bounds':texts,
              'assets':{ext:{'name':stem.with_suffix('.'+ext).name,'sha256':sha(stem.with_suffix('.'+ext))} for ext in ['svg','pdf','png','tiff']},
              'visual_qa_status':'pending_direct_inspection'}
    (out/'figure_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return {'journal':journal,'plotted_cells':len(rows),'dimensions_mm':[width,height],'output':str(out)}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--journal',choices=[*VARIANTS,'all'],default='all')
    a=p.parse_args()
    for j in VARIANTS if a.journal=='all' else [a.journal]:
        print(json.dumps(render(j,a.output_dir/j,a.data_dir)))
