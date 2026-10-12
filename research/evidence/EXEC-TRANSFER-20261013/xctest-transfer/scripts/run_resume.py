"""Bounded local simulator-only probe runner. Never contacts real devices."""
import json, time, random, pathlib, sys, hashlib
import wda
from probes import probes,r6
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'resume-20261013'
OUT.mkdir(exist_ok=False)
LOG=pathlib.Path('/Users/ashernoble/Library/Developer/CoreSimulator/Devices/A0EF1497-8235-4DAE-855A-16FDF8D60796/data/Containers/Data/Application/61ABD531-CA24-4B24-909A-C809F207A86B/Documents/touches.jsonl')
(OUT/'prior-touch-log.jsonl').write_bytes(LOG.read_bytes())
P=probes(); P.update(r6(ROOT/'payloads/r6.json'))
items=[(name,rep) for name in P for rep in range(5)]
random.Random(20261013).shuffle(items)
manifest={'simulator_udid':'A0EF1497-8235-4DAE-855A-16FDF8D60796','simulator_runtime':'iOS 18.6','simulator_type':'iPhone XR','xcode':'26.6 (17F113)','wda_expected_revision':'ae50404aac12d9f8c41f6c3fa8776e97975eaef5','seed':20261013,'repeats':5,'probes':{k:{'payload':v[0][0],'cmd':v[0][1],'kind':v[1]} for k,v in P.items()},'order':items}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
status=wda.call('GET','/status');(OUT/'wda-status.json').write_text(json.dumps(status,indent=2))
sid=wda.new_session(); timing=wda.timing(sid,True)
assert timing['value']['build_revision']==manifest['wda_expected_revision'],timing
(OUT/'session.json').write_text(json.dumps({'session_id':sid,'timing_initial':timing},indent=2))
f=LOG.open();f.seek(0,2)
start=time.monotonic()
with (OUT/'requests.jsonl').open('w') as dest:
 for idx,(name,rep) in enumerate(items):
  (payload,cmd),kind=P[name]
  response,t0,t1=wda.actions(sid,payload)
  time.sleep(.03)
  lines=f.readlines();rows=[json.loads(l) for l in lines if l.strip()]
  records=wda.timing(sid)['value']['records']
  item={'index':idx,'name':name,'repeat':rep,'kind':kind,'http_start_monotonic':t0,'http_end_monotonic':t1,'response':response,'touches':rows,'timing':records[-1] if records else None}
  dest.write(json.dumps(item)+'\n');dest.flush()
  assert rows and sum(r.get('ph')=='B' for r in rows)==1 and sum(r.get('ph')=='E' for r in rows)==1,(name,rows)
  assert not any(r.get('ph')=='C' for r in rows),(name,rows)
  if (idx+1)%25==0: print(json.dumps({'completed':idx+1,'total':len(items),'elapsed_s':round(time.monotonic()-start,1),'last':name}),flush=True)
(OUT/'completed.json').write_text(json.dumps({'requests':len(items),'elapsed_s':time.monotonic()-start},indent=2))
print('COMPLETE',len(items),flush=True)
