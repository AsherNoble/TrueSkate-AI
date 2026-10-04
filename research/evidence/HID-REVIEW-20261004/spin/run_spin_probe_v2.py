"""One explicitly bounded XR2 diagnostic; no corpus or training admission."""
import argparse
import base64
import concurrent.futures
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RUNTIME = Path('/Users/training-server/trueskate-ai-runtime/tmp/hid-pointer')
OUT = RUNTIME / 'agent-spin-codex-20261004'
sys.path[:0] = [str(RUNTIME), str(RUNTIME / 'code/src')]
from dotenv import load_dotenv
load_dotenv('/Users/training-server/trueskate-ai/.env')
from hid_client import PointerClient
from trueskate_ai.control import hid_pointer as hp
from trueskate_ai.sim.device import DeviceSession, DEVICES, BUNDLE_ID
from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder

BASE = 'http://127.0.0.1:8103'

def http(path, body=None, timeout=20):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data, {'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        result = json.loads(r.read())
    if isinstance(result.get('value'), dict) and result['value'].get('error'):
        raise RuntimeError(str(result['value']))
    return result

def touch(x, y, ms):
    t0=time.time()
    result=http('/wda/perform_trick_gestures', {'gestures':[{'waypoints':[
        {'x':x,'y':y,'duration_ms':0}, {'x':x,'y':y,'duration_ms':ms}]}]}, timeout=15)
    return {'host_call_epoch':t0, 'host_return_epoch':time.time(), 'response':result}

def screenshot(path):
    path.write_bytes(base64.b64decode(http('/screenshot')['value']))

def guard(client, worker=None):
    state = http('/status')['value']
    if not state.get('ready'):
        raise RuntimeError('WDA not ready')
    bundle = http('/wda/activeAppInfo')['value'].get('bundleId')
    if bundle != BUNDLE_ID:
        raise RuntimeError(f'unexpected foreground {bundle}')
    if worker is not None and worker.driver.query_app_state(BUNDLE_ID) != 4:
        raise RuntimeError('Appium foreground guard failed')
    if http('/wda/video')['value'] is not None:
        raise RuntimeError('recorder not idle')
    lines=client.ask('STATUS',0.5)
    status=next((l for l in lines if l.startswith('STATUS')), '')
    if not all(k in status for k in ('connected=1','auth=1','subscribed=1','interval_us=15000')):
        raise RuntimeError(f'pointer not ready: {status}')
    return {'wda':state,'active_bundle':bundle,'pointer_status':status,'recorder_idle':True}

def upload(client, events):
    client.ask('CLEAR',0.1)
    client.send('\n'.join(f'E {t} {dx} {dy} {b}' for t,dx,dy,b in events))
    oks=0; end=time.monotonic()+10
    while oks < len(events) and time.monotonic() < end:
        lines=client.read_lines(0.05)
        if any(l.startswith('ERR') for l in lines):
            raise RuntimeError(f'schedule rejected: {lines}')
        oks += sum(l == 'OK' for l in lines)
    if oks != len(events):
        raise RuntimeError(f'schedule acknowledgements {oks}/{len(events)}')

def play(client, events):
    go=time.time(); client.send('GO')
    lines=client.wait_for('DONE',timeout=events[-1][0]/1e6+5)
    sent=[int(l.split()[2]) for l in lines if l.startswith('SENT')]
    if len(sent) != len(events):
        raise RuntimeError(f'schedule sent count {len(sent)}/{len(events)}')
    late=[s-e[0] for s,e in zip(sent,events)]
    return {'go_epoch':go,'sent_us':sent,'min_late_us':min(late),'max_late_us':max(late),'board_lines':lines}

def main():
    p=argparse.ArgumentParser(); p.add_argument('condition',choices=['pointer','hold','combined']); p.add_argument('run_id'); a=p.parse_args()
    if len(list(OUT.glob('*/run.json'))) >= 12:
        raise RuntimeError('12-run cap reached')
    destination=OUT/a.run_id
    if destination.exists():
        raise RuntimeError('run output must be new')
    daemon=subprocess.run(['/bin/launchctl','print','system/com.trueskate.remotexpc-tunnel'],capture_output=True,text=True,check=True,timeout=5).stdout
    if 'state = running' not in daemon:
        raise RuntimeError('root RemoteXPC tunnel not running')
    cleanup=subprocess.run(['bash','/Users/training-server/trueskate-ai/scripts/recover_remotexpc_attachments.sh','--dry-run','xr2'],capture_output=True,text=True,check=True,timeout=45).stdout
    if 'Found 0 UUID-shaped attachment' not in cleanup:
        raise RuntimeError('XR2 attachments not empty: '+cleanup)
    client=PointerClient(port=8765); worker=None; recorder=None; executor=None
    result={'condition':a.condition,'device':'iPhone_XR2','training_admission':False,
            'park':'unverified indoor gameplay scene; no park selection or label inferred',
            'tunnel':daemon,'pre_attachments':cleanup,'runs_cap':12,'recording_cap_seconds':45,
            'spin_point_pt':[25,362],'wda_spin_hold_ms':4000}
    try:
        result['preflight']=guard(client)
        cfg=next(d for d in DEVICES if d['name']=='iPhone_XR2')
        worker=DeviceSession(cfg); worker.connect()
        result['connected_guard']=guard(client,worker)
        destination.mkdir(parents=True)
        screenshot(destination/'before.png')
        rows=json.loads((OUT/'flick-a.json').read_text())
        strokes=[hp.Stroke(name,[tuple(row) for row in points]) for name,points in rows.items()]
        events,info=hp.build_schedule(strokes,lift='immediate')
        result['events']=events; result['schedule_info']=info; result['strokes']=rows
        if a.condition != 'hold':
            upload(client,events)
        result['reset']=touch(207,49,50)
        time.sleep(3)
        result['post_reset_guard']=guard(client,worker)
        screenshot(destination/'after-reset.png')
        recorder=XCTestScreenRecorder(worker.driver,fps=60)
        result['recording_start']=recorder.start()  # Single start attempt; stop on failure.
        segment_started=time.monotonic()
        time.sleep(0.8)
        result['start_calibration']=touch(207,448,50)
        time.sleep(0.4)
        if a.condition in ('hold','combined'):
            executor=concurrent.futures.ThreadPoolExecutor(max_workers=1)
            hold=executor.submit(touch,25,362,4000)
            # Recovered measured WDA onset ~0.8--0.9 s, +/-0.1 s. The first
            # pointer touch occurs another ~2.28 s after GO, inside the hold.
            time.sleep(1.35)
            if a.condition == 'combined':
                result['pointer']=play(client,events)
            result['hold']=hold.result(timeout=10)
        else:
            result['pointer']=play(client,events)
        time.sleep(2)
        result['end_calibration']=touch(207,448,50)
        time.sleep(0.5)
        result['recording_elapsed_before_stop']=time.monotonic()-segment_started
        if result['recording_elapsed_before_stop'] >= 45:
            raise RuntimeError('recording cap unexpectedly reached')
        recording=recorder.stop_and_save(destination/'recording.mov')
        result['recording']=recording.summary()
        # Exact source-frame decode. AVFoundation/OpenCV yielded 447 frames
        # for the SHA-verified 446-sample control; FFmpeg and ffprobe agree.
        # Passthrough disables synthetic frame-rate duplication.
        decode=subprocess.run(['/usr/local/bin/ffmpeg','-nostdin','-v','error','-i',str(destination/'recording.mov'),'-map','0:v:0','-fps_mode','passthrough','-f','framemd5',str(destination/'decoded-frames.framemd5')],capture_output=True,text=True,check=True,timeout=30)
        decoded=sum(bool(line) and not line.startswith('#') for line in (destination/'decoded-frames.framemd5').read_text().splitlines())
        if decoded < 30:
            raise RuntimeError(f'decoded only {decoded} frames')
        result['decoded_frames']=decoded
        result['decoder']='FFmpeg source frames; fps_mode passthrough'
        probe=subprocess.run(['/usr/local/bin/ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_frames','-show_entries','stream=avg_frame_rate,r_frame_rate,duration,width,height,nb_frames,nb_read_frames:frame=best_effort_timestamp_time,pkt_duration_time','-of','json',str(destination/'recording.mov')],capture_output=True,text=True,check=True,timeout=30)
        (destination/'ffprobe.json').write_text(probe.stdout)
        data=json.loads(probe.stdout)
        if len(data.get('frames',[])) != decoded or int(data['streams'][0]['nb_read_frames']) != decoded:
            raise RuntimeError('decoded and probed frame counts disagree')
        result['postflight']=guard(client,worker)
        screenshot(destination/'after.png')
        cleanup_after=subprocess.run(['bash','/Users/training-server/trueskate-ai/scripts/recover_remotexpc_attachments.sh','--dry-run','xr2'],capture_output=True,text=True,check=True,timeout=45).stdout
        result['post_attachments']=cleanup_after
        if 'Found 0 UUID-shaped attachment' not in cleanup_after:
            raise RuntimeError('recording attachment survived stop')
        result['completed']=True
    except BaseException as exc:
        result['completed']=False; result['error']=repr(exc)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        result['release']=client.ask('NOW 0 0 0',0.3)
        if recorder is not None and recorder.is_recording:
            try:
                result['error_recording']=recorder.stop_and_save(destination/'error-recording.mov').summary()
            except Exception as exc:
                result['stop_error']=repr(exc)
        if worker is not None:
            worker.disconnect()
        client.close()
        if destination.exists():
            (destination/'run.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'out':str(destination),'completed':result.get('completed'),
                          'error':result.get('error'),'decoded_frames':result.get('decoded_frames'),
                          'recording':result.get('recording'),'release':result.get('release')},indent=2),flush=True)

if __name__=='__main__':
    main()
