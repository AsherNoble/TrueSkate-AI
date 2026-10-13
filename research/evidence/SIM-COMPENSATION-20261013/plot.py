"""Render standalone scientific comparison from frozen prospective measurements."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import study

ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT/'results.json').read_text())
frozen = json.loads((ROOT/'FROZEN.json').read_text())
rows = [json.loads(line) for line in gzip.open(ROOT/'requests.jsonl.gz','rt')]
names = [n for n,c in result['cases'].items() if c['split']=='heldout']
fig, axes = plt.subplots(2,2,figsize=(13,9),layout='constrained')
colors = {'nominal':'#376a9c','adjusted':'#c46b28'}
for ax,metric,title,threshold in [(axes[0,0],'rms_normalized','Trajectory RMS (four-repeat median)',.005),
                                   (axes[0,1],'max_normalized','Maximum position error (four-repeat median)',.01)]:
    xx = np.arange(len(names))
    for variant,offset in [('nominal',-.19),('adjusted',.19)]:
        ax.bar(xx+offset,[result['cases'][n][variant][metric] for n in names],width=.36,label=variant,color=colors[variant])
    ax.axhline(threshold,color='#555',linestyle='--',linewidth=1,label='frozen position threshold')
    ax.set_xticks(xx,names,rotation=75,ha='right',fontsize=8)
    ax.set_ylabel('Normalized Euclidean error')
    ax.set_title(title)
    ax.legend(fontsize=8)
for ax,name in [(axes[1,0],'straight463'),(axes[1,1],'middle_hold131')]:
    case = frozen['cases'][name]
    for variant in ['nominal','adjusted']:
        row = next(r for r in rows if r['stage']=='measurement' and r['case']==name and r['variant']==variant and r['repeat']==0)
        begin = next(e['ts'] for e in row['touches'] if e['ph']=='B')
        end = next(e['ts'] for e in row['touches'] if e['ph']=='E')
        ss = study.samples(row)
        ts = (np.array([x[0] for x in ss])-begin)*1000
        xy = np.array([x[1] for x in ss])
        err = np.linalg.norm((xy-study.reference(case,ts))/study.SCALE,axis=1)
        ax.plot(ts,err,color=colors[variant],label=variant)
        ax.axvline((end-begin)*1000,color=colors[variant],linestyle=':',linewidth=1)
    for i in range(len(case['points'])-1):
        if case['points'][i]==case['points'][i+1]:
            ax.axvspan(case['times_ms'][i],case['times_ms'][i+1],color='#777',alpha=.12)
    ax.axhline(.01,color='#555',linestyle='--',linewidth=1)
    ax.axvline(case['times_ms'][-1],color='black',linestyle='--',linewidth=1,label='desired lift')
    ax.set_title(name+' — repeat0 illustration')
    ax.set_xlabel('Raw touch time after began (ms)')
    ax.set_ylabel('Normalized position error')
    ax.legend(fontsize=8)
fig.suptitle('Prospective simulator compensation: partial improvement, frozen full-effect gate failed',fontsize=14)
fig.savefig(ROOT/'comparison.png',dpi=180)
fig.savefig(ROOT/'comparison.svg')
