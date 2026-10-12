"""Analysis of direct UITouch observations; raw timestamps retained, no video inference."""
import json,pathlib,collections,numpy as np,csv
ROOT=pathlib.Path(__file__).resolve().parents[1]
P=ROOT/'resume-20261013'; M=json.load((P/'manifest.json').open()); R=[json.loads(l) for l in (P/'requests.jsonl').open()]
CP=np.array(json.load((P/'recap-control-points.json').open())['touch_control_points'])
def stats(v):
 a=np.asarray(v,float);return {'n':len(a),'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max()),'mean':float(a.mean()),'sd':float(a.std()),'p95':float(np.percentile(a,95))} if len(a) else None
def ease(x):
 x=np.clip(np.asarray(x,float),0,1);lo=np.zeros_like(x);hi=np.ones_like(x)
 for _ in range(45):
  u=(lo+hi)/2;bx=3*(1-u)**2*u*CP[0]+3*(1-u)*u**2*CP[2]+u**3;lo=np.where(bx<x,u,lo);hi=np.where(bx<x,hi,u)
 return 3*(1-u)**2*u*CP[1]+3*(1-u)*u**2*CP[3]+u**3

def samples(r):
 # Use coalesced sequence alone when available; it includes the main sample.
 out=[];seen=set()
 for e in r['touches']:
  for v in e.get('co') or [[e['ts'],e['x'],e['y'],e['ph']]]:
   key=tuple(v)
   if key not in seen:out.append(v);seen.add(key)
 return out

def dist_poly(pts,cmd):
 a=np.asarray(cmd,float)[:,1:];d=a[1:]-a[:-1];den=(d*d).sum(1);delta=pts[:,None,:]-a[:-1][None,:,:]
 u=np.divide((delta*d).sum(2),den[None,:],out=np.zeros((len(pts),len(d))),where=den[None,:]>0);u=np.clip(u,0,1)
 return np.linalg.norm(delta-u[:,:,None]*d,axis=2).min(1)
by=collections.defaultdict(list);rows=[];allinter=[];cb_lags=[];onsets=[];linerrs=[];beziererrs=[]
for r in R:
 name=r['name'];meta=M['probes'][name];cmd=np.array(meta['cmd'],float);events=r['touches'];s=samples(r);b=next(e for e in events if e.get('ph')=='B');end=next(e for e in events if e.get('ph')=='E');ss=np.array([v[:3] for v in s]);tt=(ss[:,0]-b['ts'])*1000;pt=ss[:,1:]
 duration=(end['ts']-b['ts'])*1000;frames=np.round(tt*60/1000).astype(int)
 pred=np.column_stack([np.interp(tt,cmd[:,0],cmd[:,k]) for k in (1,2)]);lin=np.linalg.norm(pt-pred,axis=1)
 dpoly=dist_poly(pt,cmd)
 row={'index':r['index'],'name':name,'repeat':r['repeat'],'kind':r['kind'],'command_duration_ms':cmd[-1,0],'observed_duration_ms':duration,'duration_error_ms':duration-cmd[-1,0],'callback_count':len(events),'unique_sample_count':len(s),'linear_error_rms_pt':float(np.sqrt(np.mean(lin**2))),'linear_error_max_pt':float(lin.max()),'polyline_max_dist_pt':float(dpoly.max()),'frame_count':int(frames.max())}
 if r.get('timing') and abs(r['timing']['request_entered']['monotonic_s']-r['http_start_monotonic'])<.1:
  stamp=r['timing'];sub=stamp['submitted_to_ios']['monotonic_s'];row['onset_minus_submit_ms']=(b['ts']-sub)*1000;onsets.append(row['onset_minus_submit_ms']);row['callback_minus_ended_ms']=(stamp['ios_completion_callback']['monotonic_s']-end['ts'])*1000
 for e in events:cb_lags.append((e['now']-e['ts'])*1000)
 dts=np.diff(np.unique(ss[:,0]))*1000;allinter.extend(dts[(dts>5)&(dts<25)])
 if r['kind']=='single' and name not in ['single_16','single_33','single_50']:
  # nearest synthesis cadence separates timestamp delivery jitter from position law
  uu=frames*1000/60/cmd[-1,0];predbez=cmd[0,1:]+ease(uu)[:,None]*(cmd[-1,1:]-cmd[0,1:]);err=np.linalg.norm(predbez-pt,axis=1)
  # Ended can snap to endpoint at rounded time; validate only interpolation samples.
  err=err[np.array([v[3] not in ('E','C') for v in s])]
  row['bezier_error_rms_pt']=float(np.sqrt(np.mean(err**2)));row['bezier_error_max_pt']=float(err.max());beziererrs.extend(err);linerrs.extend(lin)
 by[name].append((r,row,frames,tt,pt));rows.append(row)
summary={}
for name,rs in by.items():
 sets=[]
 for r,row,ff,tt,pt in rs:
  sets.append({int(f):(t,p) for f,t,p in zip(ff,tt,pt)})
 common=set.intersection(*(set(s) for s in sets));sdpos=[];sdts=[]
 for f in common:
  loc=np.array([s[f][1] for s in sets]);t=np.array([s[f][0] for s in sets]);sdpos.append(float(np.sqrt(np.var(loc,axis=0).sum())));sdts.append(float(t.std()))
 summary[name]={'kind':rs[0][1]['kind'],'n_repeats':len(rs),'duration_error_ms':stats([r[1]['duration_error_ms'] for r in rs]),'observed_duration_ms':stats([r[1]['observed_duration_ms'] for r in rs]),'common_frames':len(common),'position_repeat_sd_pt':stats(sdpos),'timestamp_repeat_sd_ms':stats(sdts),'linear_error_rms_pt':stats([r[1]['linear_error_rms_pt'] for r in rs]),'polyline_max_dist_pt':stats([r[1]['polyline_max_dist_pt'] for r in rs])}
kind={}
for k in sorted({r['kind'] for r in rows}):
 rr=[r for r in rows if r['kind']==k];kind[k]={'requests':len(rr),'duration_error_ms':stats([r['duration_error_ms'] for r in rr]),'linear_error_rms_pt':stats([r['linear_error_rms_pt'] for r in rr]),'polyline_max_dist_pt':stats([r['polyline_max_dist_pt'] for r in rr])}
result={'request_count':len(R),'unique_commands':len(by),'all_complete':len(R)==len(M['order']),'cadence_interval_ms':stats(allinter),'app_callback_lag_ms':stats(cb_lags),'onset_minus_wda_submit_ms':stats(onsets),'single_bezier_point_error_pt':stats(beziererrs),'single_linear_point_error_pt':stats(linerrs),'kind':kind,'by_command':summary}
(P/'summary.json').write_text(json.dumps(result,indent=2));(P/'request-metrics.json').write_text(json.dumps(rows,indent=2))
keys=sorted(set().union(*(r.keys() for r in rows)))
with (P/'request-metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
print(json.dumps({k:v for k,v in result.items() if k!='by_command'},indent=2))
