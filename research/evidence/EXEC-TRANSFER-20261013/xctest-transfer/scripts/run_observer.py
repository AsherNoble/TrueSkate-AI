"""ABA direct callback-file-IO vs buffered memory logger control, 90 local requests."""
import pathlib,json,random,time,subprocess
import wda
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'observer-20261013';OUT.mkdir(exist_ok=False)
M=json.load((ROOT/'resume-20261013/manifest.json').open());names=['r6_053','r6_080','r6_153','single_1200','hold_first_100','corner90_100'];order=[(n,r) for n in names for r in range(5)];random.Random(612).shuffle(order)
(OUT/'manifest.json').write_text(json.dumps({'groups':['synchronous-before','buffered','synchronous-after'],'commands':{n:M['probes'][n] for n in names},'order':order,'seed':612,'buffered_source':'../touchlogger-buffered/src/main.m'},indent=2))
with (OUT/'requests.jsonl').open('w') as out:
 for group,bundle in [('synchronous-before','com.l0.touchlogger'),('buffered','com.l0.touchlogger.memory'),('synchronous-after','com.l0.touchlogger')]:
  sid=wda.new_session(bundle)
  container=subprocess.check_output(['xcrun','simctl','get_app_container','A0EF1497-8235-4DAE-855A-16FDF8D60796',bundle,'data'],text=True).strip();log=pathlib.Path(container)/'Documents/touches.jsonl';f=log.open();f.seek(0,2)
  for i,(n,rep) in enumerate(order):
   wda.timing(sid,True);response,t0,t1=wda.actions(sid,M['probes'][n]['payload']);rows=[]
   deadline=time.monotonic()+2
   while time.monotonic()<deadline:
    time.sleep(.01);rows.extend(json.loads(l) for l in f.readlines() if l.strip())
    if any(r.get('ph')=='E' for r in rows):break
   assert sum(r.get('ph')=='B' for r in rows)==1 and sum(r.get('ph')=='E' for r in rows)==1,(group,n,rows)
   out.write(json.dumps({'group':group,'index':i,'name':n,'repeat':rep,'touches':rows,'timing':wda.timing(sid)['value']['records'][0],'http_start_monotonic':t0,'http_end_monotonic':t1})+'\n');out.flush()
  print(group,'complete',flush=True)
(OUT/'completed.json').write_text(json.dumps({'requests':90}))
