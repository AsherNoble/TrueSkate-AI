import json,pathlib,collections,numpy as np,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];P=ROOT/'resume-20261013';M=json.load((P/'manifest.json').open());R=[json.loads(l) for l in (P/'requests.jsonl').open()]
by=collections.defaultdict(list)
for r in R:
 if r['kind']!='r6':continue
 b=next(e['ts'] for e in r['touches'] if e.get('ph')=='B');one={}
 for e in r['touches']:
  for ts,x,y,ph in e.get('co') or [[e['ts'],e['x'],e['y'],e['ph']]]:
   frame=round((ts-b)*60);one[frame]=(x,y,ph)
 by[r['name']].append(one)
res={}
for n,rs in by.items():
 allframes=sorted(set().union(*(set(r) for r in rs)));pts=[]
 for f in allframes:
  arr=np.array([r[f][:2] for r in rs if f in r]);xy=np.median(arr,axis=0);pts.append({'frame':f,'t_ms':f*1000/60,'x_pt':float(xy[0]),'y_pt':float(xy[1]),'n_repeats':len(arr),'xy_sd_pt':float(np.sqrt(np.var(arr,axis=0).sum())),'phases':sorted(set(r[f][2] for r in rs if f in r))})
 res[n]={'id':int(n[3:]),'command':M['probes'][n]['cmd'],'delivered_samples':pts,'delivered_duration_ms':pts[-1]['t_ms'],'repeat_count':len(rs)}
obj={'method':'dedicated XR iOS18.6 simulator UITouch/coalesced samples, five repeats each; time anchored to actualtouchbegan, quantizedto60Hzcadence. Video onsetcalibrationdelta is separate. Linear interpolation of delivered samples is an analysis convenience, not verified game rendering.','viewport_pt':[414,896],'wda':'ae50404aac12d9f8c41f6c3fa8776e97975eaef5','source_requests_sha256':hashlib.sha256((P/'requests.jsonl').read_bytes()).hexdigest(),'commands':res}
(P/'r6-transfer.json').write_text(json.dumps(obj,indent=2));print(len(res),'commands;',max(p['xy_sd_pt'] for r in res.values() for p in r['delivered_samples']),'maxXYsd')
