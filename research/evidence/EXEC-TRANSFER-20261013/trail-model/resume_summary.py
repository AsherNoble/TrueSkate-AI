"""Offline, clip-balanced geometry/fade audit. Does not write to prior lag outputs.

Profiles are observations in native pixels. Fade age zero is the observed local
maximum, not the calibrated command timestamp; no absolute latency is implied.
"""
import json
from collections import defaultdict
import numpy as np
from common import OUT, manifest, Path_
from xsec import load, xprofile, valid_mask


def stats(vals):
    a = np.asarray(vals, float); a = a[np.isfinite(a)]
    return dict(n=len(a), median=float(np.median(a)), p10=float(np.percentile(a,10)),
                p25=float(np.percentile(a,25)), p75=float(np.percentile(a,75)),
                p90=float(np.percentile(a,90))) if len(a) else dict(n=0)


def crossing(t, a, level):
    good=np.where(a <= level)[0]
    if not len(good): return None
    k=good[0]
    if k==0:return float(t[0])
    return float(np.interp(level,[a[k],a[k-1]],[t[k],t[k-1]]))


def main():
    profiles=[]; clip_stats={}; exclusions=defaultdict(int)
    for sid in sorted(manifest()):
        if not (OUT/f'data/{sid}.npz').exists():continue
        d=load(sid); t=d['elapsed']; ss=d['s']; pre=t < -50
        pk=[];fw=[];ridge=[]
        for j in range(len(t)):
            a,b,c=xprofile(d['EX'][j]);pk.append(a);ridge.append(b);fw.append(c)
        pk=np.asarray(pk);fw=np.asarray(fw);ridge=np.asarray(ridge)
        base=np.median(pk[pre],0) if pre.any() else np.zeros(len(ss))
        p0=(d['speed']>=.5)&(d['speed']<=4)&(d['vert_d']>70)&(d['ov']>160)&d['inb']&(ss>70)&(ss<d['S'][-1]-70)&(base<12)
        rows=[]
        for i in np.where(p0)[0]:
            win=np.where((t>=d['t_first'][i]-70)&(t<=d['t_first'][i]+120)&~d['BOARD'][:,i])[0]
            if not len(win):continue
            j=int(win[np.argmax(pk[win,i])]); amp=float(pk[j,i]-base[i])
            if amp<40 or abs(ridge[j,i])>8 or not 20<fw[j,i]<65 or d['NOISE'][j,i]>8:continue
            age=t-t[j]; future=(age>=0)&(age<=340)
            if future.sum()<15 or d['BOARD'][future,i].any():continue
            ix=np.where(future)[0]; vals=pk[ix,i]-base[i]
            # A later floor stripe or board invalidates a simple one-pass fade.
            if np.median(vals[-3:])>12:continue
            half=crossing(age[ix],vals,amp/2); ten=crossing(age[ix],vals,10)
            if half is None or ten is None:continue
            row=dict(sid=sid,s=float(ss[i]),k_peak=int(d['ks'][j]),speed=float(d['speed'][i]),
                     amplitude=amp,half_ms=half,above10_ms=ten, widths={}, levels={})
            for a in [0,50,100,150,200,250,300]:
                jj=int(np.argmin(abs(age-a)))
                if abs(age[jj]-a)<9 and not d['BOARD'][jj,i]:
                    row['levels'][a]=float(pk[jj,i]-base[i]);row['widths'][a]=float(fw[jj,i])
            rows.append(row)
        if rows:
            profiles.extend(rows)
            clip_stats[sid]=dict(n_spots=len(rows), half_ms=float(np.median([x['half_ms'] for x in rows])),
                                above10_ms=float(np.median([x['above10_ms'] for x in rows])),
                                widths={a:float(np.nanmedian([x['widths'].get(a,np.nan) for x in rows])) for a in [0,50,100,150,200]},
                                levels={a:float(np.nanmedian([x['levels'].get(a,np.nan) for x in rows])) for a in [0,50,100,150,200,250,300]})
    summary=dict(method='Observed local peak alignment; clip medians have equal weight; selection in script',
                 n_clips=len(clip_stats),n_spots=len(profiles),per_clip=clip_stats,
                 half_ms=stats([x['half_ms'] for x in clip_stats.values()]),
                 above10_ms=stats([x['above10_ms'] for x in clip_stats.values()]),
                 widths={a:stats([x['widths'][a] for x in clip_stats.values()]) for a in [0,50,100,150,200]},
                 levels={a:stats([x['levels'][a] for x in clip_stats.values()]) for a in [0,50,100,150,200,250,300]})
    (OUT/'data/resume-fade-spots.json').write_text(json.dumps(profiles,indent=2))
    (OUT/'data/resume-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='per_clip'},indent=2))

if __name__=='__main__':main()
