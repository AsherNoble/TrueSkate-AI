"""255 bounded simulator-only cases with native WDA record dumps."""
import json,time,random,pathlib
import wda
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'boundary-20261013';OUT.mkdir(exist_ok=False)
P=json.load((ROOT/'boundary-probes.json').open());items=[(n,r) for n in P for r in range(5)];random.Random(71).shuffle(items)
(OUT/'manifest.json').write_text(json.dumps({'probes':P,'order':items,'repeats':5,'seed':71,'diagnostic_patch':'../diagnostic-dump.patch'},indent=2))
LOG=pathlib.Path('/Users/ashernoble/Library/Developer/CoreSimulator/Devices/A0EF1497-8235-4DAE-855A-16FDF8D60796/data/Containers/Data/Application/61ABD531-CA24-4B24-909A-C809F207A86B/Documents/touches.jsonl')
DUMP=ROOT/'diagnostic-wda-run.log';sid=wda.new_session();f=LOG.open();f.seek(0,2);g=DUMP.open();g.seek(0,2);start=time.monotonic()
with (OUT/'requests.jsonl').open('w') as out:
 for i,(n,r) in enumerate(items):
  # Enabling resets the bounded WDA diagnostics buffer, so sequence0 is always fresh.
  wda.timing(sid,True)
  response,t0,t1=wda.actions(sid,P[n]['payload']);time.sleep(.03)
  rows=[json.loads(l) for l in f.readlines() if l.strip()]
  native=[json.loads(l.split('L0_SYNTH_RECORD ',1)[1]) for l in g.readlines() if 'L0_SYNTH_RECORD ' in l]
  timing=wda.timing(sid)['value']['records'];assert len(timing)==1
  assert len(native)==1,(i,n,len(native))
  item={'index':i,'name':n,'repeat':r,'kind':P[n]['kind'],'response':response,'touches':rows,'native_dump':native[0],'timing':timing[0],'http_start_monotonic':t0,'http_end_monotonic':t1}
  out.write(json.dumps(item)+'\n');out.flush()
  assert sum(v.get('ph')=='B' for v in rows)==1 and sum(v.get('ph')=='E' for v in rows)==1,(n,rows)
  assert abs(native[0]['maximum_offset_s']*1000-P[n]['cmd'][-1][0])<1e-5
  if(i+1)%25==0:print(json.dumps({'completed':i+1,'total':len(items),'elapsed_s':time.monotonic()-start}),flush=True)
(OUT/'completed.json').write_text(json.dumps({'requests':len(items),'elapsed_s':time.monotonic()-start}))
print('COMPLETE',len(items),flush=True)
