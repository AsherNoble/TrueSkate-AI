"""Bounded localhost-only simulator measurement; no retries or physical APIs."""
import hashlib
import json
import pathlib
import random
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT.parents[2]
OUT = WORK/'tmp/sim-compensation'
UDID = 'A0EF1497-8235-4DAE-855A-16FDF8D60796'
BUNDLE = 'com.trueskate.simcomp.logger'
BASE = 'http://127.0.0.1:8199'

def call(method, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE+route, data=data, method=method, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=15) as response:
        result = json.load(response)
    if isinstance(result.get('value'),dict) and result['value'].get('error'):
        raise RuntimeError(result)
    return result

def main():
    frozen = json.loads((ROOT/'FROZEN.json').read_text())
    assert frozen['request_limit']==160 and frozen['setup_smoke_limit']==10
    assert not (OUT/'attempts.jsonl').exists(), 'Never resume/retry a partially executed run'
    session = call('POST','/session',{'capabilities':{'alwaysMatch':{'bundleId':BUNDLE,'shouldWaitForQuiescence':False,'shouldUseSingletonTestManager':False}}})
    sid = session.get('sessionId') or session['value']['sessionId']
    route = '/session/'+sid
    call('POST',route+'/wda/actionTiming',{'enabled':True})
    info = call('GET',route+'/wda/actionTiming')
    (OUT/'session.json').write_text(json.dumps({'session':session,'timing':info},indent=2))
    assert info['value']['build_revision']=='ae50404aac12d9f8c41f6c3fa8776e97975eaef5'
    container = subprocess.check_output(['xcrun','simctl','get_app_container',UDID,BUNDLE,'data'],text=True).strip()
    log = pathlib.Path(container)/'Documents/touches.jsonl'
    source = log.open(); source.seek(0,2)
    work = [('setup','dev_straight','nominal',i) for i in range(2)]
    # Smokes use a separate simple400ms payload, not scored development output.
    smokes = {'actions':[{'type':'pointer','id':'setup','parameters':{'pointerType':'touch'},'actions':[
        {'type':'pointerMove','duration':0,'x':120,'y':448,'origin':'viewport'},
        {'type':'pointerDown','button':0},
        {'type':'pointerMove','duration':400,'x':290,'y':448,'origin':'viewport'},
        {'type':'pointerUp','button':0}]}]}
    order = [('measurement',name,variant,rep) for name in frozen['cases'] for variant in ['nominal','adjusted'] for rep in range(4)]
    random.Random(frozen['seed']).shuffle(order)
    work.extend(order)
    (OUT/'run-manifest.json').write_text(json.dumps({'frozen_sha256':hashlib.sha256((ROOT/'FROZEN.json').read_bytes()).hexdigest(),'order':work,'udid':UDID,'bundle':BUNDLE},indent=2))
    with (OUT/'attempts.jsonl').open('x') as ledger, (OUT/'requests.jsonl').open('x') as sink:
        for i,(stage,name,variant,rep) in enumerate(work):
            assert i<170
            case = frozen['cases'][name]
            payload = smokes if stage=='setup' else case['nominal_payload'] if variant=='nominal' else case['adjusted']['payload']
            entered = time.monotonic()
            ledger.write(json.dumps({'index':i,'stage':stage,'case':name,'variant':variant,'repeat':rep,'entered_monotonic':entered})+'\n'); ledger.flush()
            call('POST',route+'/wda/actionTiming',{'enabled':True})
            response = call('POST',route+'/actions',payload)
            returned = time.monotonic()
            events = []
            deadline = time.monotonic()+2
            while time.monotonic()<deadline:
                events.extend(json.loads(line) for line in source.readlines() if line.strip())
                if any(e.get('ph')=='E' for e in events):
                    break
                time.sleep(.01)
            timing = call('GET',route+'/wda/actionTiming')
            row = {'index':i,'stage':stage,'case':name,'variant':variant,'repeat':rep,
                   'http_entered_monotonic':entered,'http_returned_monotonic':returned,
                   'response':response,'touches':events,'timing':timing}
            sink.write(json.dumps(row)+'\n');sink.flush()
            assert sum(e.get('ph')=='B' for e in events)==1 and sum(e.get('ph')=='E' for e in events)==1, (i,events)
            assert not any(e.get('ph')=='C' for e in events)
            if (i+1)%20==0:
                print('Completed',i+1,'of',len(work),flush=True)
    (OUT/'completed.json').write_text(json.dumps({'measurement_requests':160,'setup_requests':2,'total':162},indent=2))
    print('COMPLETE 160 measurement +2 setup',flush=True)

if __name__=='__main__':
    main()
