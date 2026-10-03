"""Freeze or execute the ten alternating bounded segments; abort without replacement."""
import argparse,json,signal,socket,subprocess,time
from pathlib import Path
from trueskate_ai.research.curved_audit import manifest,verify_manifest,save_new,reset_payload
from trueskate_ai.research.audit_execution import run_recording
from trueskate_ai.research.audit_calibration import verify_recording

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--freeze',action='store_true')
    p.add_argument('--out',type=Path);p.add_argument('--schedule',default='v1',choices=('v1','v2'))
    p.add_argument('--wda-revision')
    a=p.parse_args()
    if a.freeze:save_new(a.manifest,manifest(a.schedule));return
    if not a.out or not a.wda_revision or 'training-server' not in socket.gethostname():p.error('execution requires rig, new --out and --wda-revision')
    frozen=json.loads(a.manifest.read_text());verify_manifest(frozen)
    a.out.mkdir(parents=True,exist_ok=False)
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    from trueskate_ai.sim.device import DeviceSession,DEVICES,BUNDLE_ID
    from trueskate_ai.collection.wda_action_timing import WDAActionTimingCapture,_http_json
    from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
    worker=None
    try:
        for n,segment in enumerate(frozen['recordings'],1):
            tunnel=subprocess.check_output(['launchctl','print','system/com.trueskate.remotexpc-tunnel'],text=True)
            if 'state = running' not in tunnel:raise RuntimeError('root recording tunnel unavailable')
            cfg=next(d for d in DEVICES if d['name']==segment['device']);port=cfg['wda_port']
            base=f'http://127.0.0.1:{port}'
            # Strict preflight before connect can activate an app or issue loading taps.
            if _http_json(base+'/wda/activeAppInfo')['value']['bundleId']!=BUNDLE_ID:raise RuntimeError('True Skate must already be frontmost')
            if _http_json(base+'/wda/video').get('value') is not None:raise RuntimeError('recorder not idle')
            worker=DeviceSession(cfg);worker.connect();driver=worker.driver
            def guard():
                if driver.query_app_state(BUNDLE_ID)!=4 or worker._active_bundle_id()!=BUNDLE_ID:raise RuntimeError('foreground lost/unavailable')
            metadata=dict(identity=frozen['identity'],manifest_sha256=frozen['sha256'],device=segment['device'],park=segment['park'],
                          park_source='operator task',segment=n,gameplay_review='human',automated_gameplay_scan=False)
            settle=frozen.get('pre_roll_reset_settle_s')
            if settle:  # v2: reset before recording so the start marker fires on a settled board
                driver.execute('actions',reset_payload());time.sleep(settle);guard()
            out=a.out/f'recording_{n}'
            run_recording(recorder=XCTestScreenRecorder(driver,fps=30),timing=WDAActionTimingCapture(wda_port=port,expected_revision=a.wda_revision),
                          commands=segment['commands'],perform=lambda s:driver.execute('actions',s['payload']),guard=guard,
                          out=out,revision=a.wda_revision,metadata=metadata)
            if _http_json(base+'/wda/video').get('value') is not None:raise RuntimeError('recorder did not return idle')
            worker.disconnect();worker=None
            verify_recording(out,a.wda_revision)
            print(f'{n}/10 segments verified',flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        save_new(a.out/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',no_replacements=True));raise
    finally:
        if worker:worker.disconnect()

if __name__=='__main__':main()
