"""Frozen selected-region comparison; all output stays beside this script."""
from pathlib import Path
import argparse, hashlib, json, subprocess
from collections import Counter, defaultdict
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

OUT = Path(__file__).resolve().parent
SP = Path('/private/tmp/claude-503/-Users-ashernoble-Projects-Robotics---hardware-TrueSkate-AI/e0f1c48e-973a-4ff3-a26a-f5f2350d9a43/scratchpad')
MIRROR = SP/'audit6-mirror'
PROFILES = SP/'research/2-trail-model/data'
TRANSFER = SP/'research/1-xctest-transfer/resume-20261013/r6-transfer.json'
W,H = 828,1792

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def inputs():
    samples=json.loads((MIRROR/'audit-6/manifest.json').read_text())['samples']
    return {s['id']:s for s in samples},json.loads(TRANSFER.read_text())

def recording(s): return s['sample_dir'].split('/')[1]

def splitmap(samples):
    starts={}
    for sid,s in samples.items(): starts[recording(s)]=min(sid,starts.get(recording(s),sid))
    return {r:('development' if starts[r] in [51,71,91,111,131] else 'heldout') for r in starts}

def at_s(d,s):
    return np.array([np.interp(s,d['S'],d['Pp'][:,q]) for q in [0,1]])

def extract():
    samples,transfer=inputs(); splits=splitmap(samples); rows=[]; counts=Counter(); files={}
    for sid,sample in sorted(samples.items()):
        fn=PROFILES/f'{sid}.npz'; files[str(fn)]=sha(fn); d=dict(np.load(fn))
        # Existing raw profiles, no old timing estimates.
        core=np.max(d['EX'][:,18:23,:].astype(float),axis=1)
        pre=d['elapsed']<=-50
        baseline=np.median(core[pre],axis=0) if pre.any() else np.full(len(d['s']),np.inf)
        signal=gaussian_filter1d(core-baseline[None],1.,axis=1,mode='nearest')
        ss=d['s']; stotal=d['S'][-1]
        for threshold in [20,25,30]:
            for j,t in enumerate(d['elapsed']):
                if not 0<=t<=d['Pt'][-1]+100: continue
                counts[f'{threshold}:active_frames']+=1; sig=signal[j]
                cross=np.where((sig[:-1]>=threshold)&(sig[1:]<threshold))[0]
                good=[]
                for i in cross:
                    edge=float(ss[i]+4*(sig[i]-threshold)/(sig[i]-sig[i+1]))
                    behind=(ss>=edge-48)&(ss<=edge-16)
                    ahead=(ss>=edge+16)&(ss<=edge+48)
                    corridor=(ss>=edge-60)&(ss<=edge+60)
                    if behind.sum()<6 or ahead.sum()<6 or corridor.sum()<25: continue
                    if np.median(sig[behind])<40 or np.median(sig[ahead])>10: continue
                    if np.mean(sig[behind]>=threshold)<.8 or np.any(sig[ahead]>=threshold): continue
                    counts[f'{threshold}:photometric_edges']+=1
                    geo=(d['inb'] & (d['vert_d']>70) & (ss>70) & (ss<stotal-70)
                         & (d['speed']>=.5) & (d['speed']<=4) & (d['ov']>160))
                    if not geo[corridor].all(): continue
                    counts[f'{threshold}:geometry_edges']+=1
                    if d['BOARD'][j,corridor].any(): continue
                    if np.median(d['NOISE'][j,corridor])>8: continue
                    if np.any(baseline[corridor]>=12): continue
                    good.append(edge)
                if len(good)!=1:
                    counts[f'{threshold}:none_or_ambiguous']+=1; continue
                edge=good[0]; q=int(np.clip(np.searchsorted(d['S'],edge,side='right')-1,0,len(d['S'])-2))
                direction=d['Pp'][q+1]-d['Pp'][q]; direction=direction/np.linalg.norm(direction)
                rows.append(dict(sid=sid,recording=recording(sample),split=splits[recording(sample)],
                    k=int(d['ks'][j]),t_ms=float(t),threshold=threshold,s_edge=edge,
                    edge_xy=at_s(d,edge).tolist(),direction=direction.tolist(),
                    speed=float(d['speed'][np.argmin(abs(ss-edge))]),segment=q))
                counts[f'{threshold}:accepted_frames']+=1
        print('extract',sid,flush=True)
    dump('candidates.json',rows)
    byclip={sid:Counter(r['threshold'] for r in rows if r['sid']==sid) for sid in samples}
    dump('selection.json',dict(counts=counts,by_clip={str(k):dict(v) for k,v in byclip.items() if v},
        split=splits,recording_count=len(splits),candidate_clip_count=sum(bool(v) for v in byclip.values())))
    dump('provenance.json',dict(frozen_plan_sha256=sha(OUT/'PLAN-FROZEN.md'),
        input_files={str(MIRROR/'audit-6/manifest.json'):sha(MIRROR/'audit-6/manifest.json'),
            str(TRANSFER):sha(TRANSFER),**files},
        source_media={str(MIRROR/samples[sid]['sample_dir']/'clip.mp4'):samples[sid]['media_sha256']
            for sid in sorted(set(r['sid'] for r in rows))},
        note='Media expected hashes are from manifest; visual stage checks actual hashes.'))
    visual(rows,samples)

def visual(rows,samples):
    selected=defaultdict(list)
    for r in rows: selected[r['sid']].append(r)
    tiles=[]; rawhash={}; visual_frames={}
    for sid,rr in sorted(selected.items()):
        frames=sorted(set(r['k'] for r in rr)); groups=[]
        for k in frames:
            if not groups or k-groups[-1][-1]>3: groups.append([k])
            else: groups[-1].append(k)
        chosen=sorted(set(k for group in groups for k in [group[0],group[len(group)//2],group[-1]]))
        visual_frames[sid]=chosen
        p=MIRROR/samples[sid]['sample_dir']/'clip.mp4'; rawhash[str(p)]=sha(p)
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','bgr24','pipe:1'])
        video=np.frombuffer(raw,np.uint8).reshape(-1,H,W,3)
        for k in chosen:
            opts=[r for r in rr if r['k']==k]; r=min(opts,key=lambda x:abs(x['threshold']-25))
            xy=np.array(r['edge_xy']); x,y=np.rint(xy).astype(int)
            frame=video[k].copy()
            cv2.circle(frame,(x,y),5,(0,255,0),2)
            ctr=xy-29*np.array(r['direction']);cv2.drawMarker(frame,tuple(np.rint(ctr).astype(int)),(255,255,0),cv2.MARKER_CROSS,14,2)
            context=cv2.resize(frame,(207,448))
            x0=max(0,min(W-320,x-160)); y0=max(0,min(H-448,y-224))
            crop=frame[y0:y0+448,x0:x0+320]
            tile=np.hstack([context,crop]);cv2.putText(tile,f'{sid} k{k} t{r["t_ms"]:.0f} thr{r["threshold"]}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            tiles.append(tile)
    for i in range(0,len(tiles),12):
        page=tiles[i:i+12]
        while len(page)%3: page.append(np.zeros_like(page[0]))
        cv2.imwrite(str(OUT/f'visual-{i//12+1:02d}.jpg'),np.vstack([np.hstack(page[q:q+3]) for q in range(0,len(page),3)]),[cv2.IMWRITE_JPEG_QUALITY,92])
    dump('visual-frames.json',visual_frames); dump('actual-media-sha256.json',rawhash)

def predict(samples,transfer,sid,t,model,mode='interpolate'):
    if model=='linear':
        g=samples[sid]['gesture'];tt=np.asarray(g['times_ms']);pp=np.asarray(g['command_points'])*[W,H]
    else:
        vals=transfer['commands'][f'r6_{sid:03d}']['delivered_samples'];tt=np.array([v['t_ms'] for v in vals]);pp=2*np.array([[v['x_pt'],v['y_pt']] for v in vals])
        if mode=='hold': return pp[np.clip(np.searchsorted(tt,t,side='right')-1,0,len(tt)-1)]
    return np.stack([np.interp(t,tt,pp[:,q]) for q in [0,1]],axis=-1)

def aggregate(rows,errors,split):
    clip=defaultdict(list); rec=defaultdict(list)
    for r,e in zip(rows,errors):
        if r['split']==split: clip[(r['recording'],r['sid'])].append(float(e))
    cc={str(sid):float(np.median(v)) for (rr,sid),v in clip.items()}
    for (rr,sid),v in clip.items(): rec[rr].append(float(np.median(v)))
    rc={r:float(np.mean(v)) for r,v in rec.items()}
    return dict(n_frames=sum(len(v) for v in clip.values()),n_clips=len(clip),n_recordings=len(rc),
        mean_recording=float(np.mean(list(rc.values()))) if rc else None,
        clip_medians=cc,recording_means=rc)

def fit():
    samples,transfer=inputs();rows=json.loads((OUT/'candidates.json').read_text());review=json.loads((OUT/'visual-review.json').read_text())
    rows=[r for r in rows if str(r['sid']) not in review['exclude_clips']]
    results=[];primary=[]
    for threshold in [20,25,30]:
        rr=[r for r in rows if r['threshold']==threshold]
        for cap in [27,29,31]:
            t=np.array([r['t_ms'] for r in rr]);obs=np.array([np.array(r['edge_xy'])-cap*np.array(r['direction']) for r in rr]);dirs=np.array([r['direction'] for r in rr])
            for mode in ['interpolate','hold']:
                fitted={}
                for model in ['linear','simulator']:
                    def pred(shift): return np.array([predict(samples,transfer,r['sid'],tm-shift,model,mode) for r,tm in zip(rr,t)])
                    def err(shift): return np.linalg.norm(pred(shift)-obs,axis=1)
                    grid=np.arange(-150,151);scores=[aggregate(rr,err(sh),'development')['mean_recording'] for sh in grid]
                    best=int(grid[np.argmin(scores)]);pe=pred(best)-obs;dist=np.linalg.norm(pe,axis=1)
                    fitted[model]=dict(shift_ms=best,development=aggregate(rr,dist,'development'),heldout=aggregate(rr,dist,'heldout'),
                        unshifted_heldout=aggregate(rr,err(0),'heldout'),
                        signed_heldout=aggregate(rr,np.sum(pe*dirs,axis=1),'heldout'),
                        frame_sensitivity={str(a):aggregate(rr,err(best+a),'heldout')['mean_recording'] for a in [-1000/60,1000/60]})
                    if threshold==25 and cap==29 and mode=='interpolate':
                        for r,e,p in zip(rr,dist,pred(best)):
                            if model=='linear':primary.append({**r,'obs_xy':(np.array(r['edge_xy'])-cap*np.array(r['direction'])).tolist(),'linear_xy':p.tolist(),'linear_error':float(e)})
                            else:
                                pr=next(x for x in primary if x['sid']==r['sid'] and x['k']==r['k']);pr.update(simulator_xy=p.tolist(),simulator_error=float(e))
                diff={r:fitted['linear']['heldout']['recording_means'][r]-v for r,v in fitted['simulator']['heldout']['recording_means'].items()}
                dv=np.array(list(diff.values()));rng=np.random.default_rng(20261013)
                boot=np.mean(rng.choice(dv,(20000,len(dv)),replace=True),axis=1) if len(dv) else np.array([np.nan])
                results.append(dict(threshold=threshold,cap_px=cap,sampling=mode,models=fitted,
                    linear_minus_simulator_px_by_recording=diff,improvement_px=float(np.mean(dv)),
                    recording_bootstrap95_px=np.percentile(boot,[2.5,97.5]).tolist()))
                print('fit',threshold,cap,mode,results[-1]['improvement_px'],flush=True)
    dump('results.json',results);dump('primary-rows.json',primary)
    makefig(primary,results)

def makefig(primary,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    p=next(r for r in results if r['threshold']==25 and r['cap_px']==29 and r['sampling']=='interpolate')
    rec=p['linear_minus_simulator_px_by_recording'];fig,ax=plt.subplots(figsize=(9,4))
    labels=[x.split('-')[-1] for x in rec];ax.bar(labels,list(rec.values()),color=['#0072B2' if v>0 else '#D55E00' for v in rec.values()]);ax.axhline(0,color='black',lw=.8)
    ax.set(xlabel='Held-out recording (suffix)',ylabel='Linear error − simulator error (native px)',title='Selected moving trail regions: positive favours simulator')
    fig.tight_layout();fig.savefig(OUT/'heldout-recordings.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,thr in zip(axes,[20,25,30]):
        for mode,color in [('interpolate','#0072B2'),('hold','#D55E00')]:
            z=[r for r in results if r['threshold']==thr and r['sampling']==mode]
            ax.plot([r['cap_px'] for r in z],[r['improvement_px'] for r in z],marker='o',color=color,label=mode)
        ax.axhline(0,color='black',lw=.8);ax.set(title=f'Excess threshold {thr}',xlabel='Assumed cap radius (native px)')
    axes[0].set_ylabel('Linear − simulator residual (native px)');axes[-1].legend()
    fig.suptitle('Measurement and sampling sensitivity; all shifts fit on development only');fig.tight_layout();fig.savefig(OUT/'sensitivity.png',dpi=160);plt.close(fig)
    held=[r for r in primary if r['split']=='heldout'];byclip=defaultdict(list)
    for r in held:byclip[r['sid']].append(r)
    fig,axes=plt.subplots(max(1,len(byclip)),1,figsize=(10,max(3,len(byclip)*1.7)),squeeze=False)
    for ax,(sid,rr) in zip(axes[:,0],sorted(byclip.items())):
        # Project onto each local tangent to show signed forward positional residual.
        for key,col in [('linear','#D55E00'),('simulator','#0072B2')]:
            yy=[np.dot(np.array(r[key+'_xy'])-r['obs_xy'],r['direction']) for r in rr]
            ax.scatter([r['t_ms'] for r in rr],yy,s=15,color=col,label=key)
        ax.axhline(0,color='black',lw=.7);ax.set_ylabel(f'{sid}\nerror px')
    axes[0,0].legend(ncol=2);axes[-1,0].set_xlabel('Movie command-clock time (ms)');fig.suptitle('Held-out signed along-path residuals (measured centre is cap-corrected edge)');fig.tight_layout();fig.savefig(OUT/'forward-residuals.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['extract','fit']);a=ap.parse_args()
    extract() if a.stage=='extract' else fit()
