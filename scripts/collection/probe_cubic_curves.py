"""Execute one frozen CURVE-EXEC recording on training-server; abort, never retry."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import signal
import subprocess
from trueskate_ai.research.curve_protocol import verify_manifest,save_new,EXPERIMENT,DEVICES
from trueskate_ai.research.curve_probe import scheduled_commands,pointer_for,run_recording


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--stage',choices=['pilot','main','confirmation'],required=True)
    p.add_argument('--device',choices=DEVICES,required=True)
    p.add_argument('--segment-index',type=int,required=True)
    p.add_argument('--park',required=True)
    p.add_argument('--wda-revision',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gate',type=Path,help='frozen pilot/main decision required for later stages')
    a=p.parse_args()
    def interrupted(signum,frame):raise KeyboardInterrupt(f'operator/process signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    if a.out.exists():p.error('preserve existing attempts; use new output')
    if 'training-server' not in socket.gethostname():p.error('run on training-server')
    manifest=json.loads(a.manifest.read_text());verify_manifest(manifest)
    rows=manifest['devices'][a.device][a.stage]
    if a.stage!='pilot':
        if not a.gate:p.error('later stage requires a frozen passing gate')
        gate=json.loads(a.gate.read_text())
        if gate.get('manifest_sha256')!=manifest['sha256'] or gate.get('next_stage')!=a.stage or not gate.get('allow_execution'):
            p.error('gate does not authorize this stage/manifest')
        if a.stage=='confirmation':rows=rows[str(gate['selected_rule_ms'])]
    if a.segment_index<0 or a.segment_index*8>=len(rows):p.error('segment outside bounded workload')
    # Earlier recordings must pass recording/calibration admission before proceeding.
    if a.segment_index:
        previous=a.out.parent/f'segment_{a.segment_index-1:02d}'/'admission.json'
        if not previous.exists() or not json.loads(previous.read_text()).get('accepted'):
            p.error('previous recording has no accepted timing/calibration admission; do not replace failures')
    tunnel=subprocess.check_output(['launchctl','print','system/com.trueskate.remotexpc-tunnel'],text=True)
    if 'state = running' not in tunnel:p.error('root recording tunnel is not running')
    from trueskate_ai.sim.device import DeviceSession,DEVICES as CONFIGS,BUNDLE_ID
    from trueskate_ai.sim.touch_actions import reset_position
    from trueskate_ai.collection.gameplay_filter import is_menu_frame,is_editor_frame
    from trueskate_ai.collection.scene_settle import wait_for_centre_settle
    from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
    from trueskate_ai.collection.wda_action_timing import WDAActionTimingCapture
    worker=DeviceSession(next(d for d in CONFIGS if d['name']==a.device))
    try:
        worker.connect()
        driver=worker.driver
        def guard():
            if driver.query_app_state(BUNDLE_ID)!=4 or worker._active_bundle_id() not in (None,BUNDLE_ID):
                raise RuntimeError('True Skate foreground lost')
            png=driver.get_screenshot_as_png()
            if is_editor_frame(png) or is_menu_frame(png,allow_idle_navigation=True):
                raise RuntimeError('gameplay contamination')
        guard()
        reset_position(driver,worker.device_w,worker.device_h)
        settle=wait_for_centre_settle(driver.get_screenshot_as_png,threshold=2.,max_wait_s=10.)
        if not settle.settled:raise RuntimeError('centre failed to settle before recording')
        guard()
        size=(int(worker.device_w),int(worker.device_h))
        commands=scheduled_commands(rows[a.segment_index*8:(a.segment_index+1)*8])
        pointers=[pointer_for(s,size) for s in commands]
        pointer_map={id(spec):finger for spec,finger in zip(commands,pointers)}
        # One W3C request per curve/control; no release requests inside timing capture.
        def perform(spec):driver.execute('actions',{'actions':[pointer_map[id(spec)].encode()]})
        timing=WDAActionTimingCapture(wda_port=worker._cfg['wda_port'],expected_revision=a.wda_revision)
        metadata=dict(experiment=EXPERIMENT,manifest_sha256=manifest['sha256'],device=a.device,
                      stage=a.stage,segment_index=a.segment_index,park=a.park,park_source='operator observation',
                      allow_idle_navigation=True,settle=settle.summary(),device_size=list(size),
                      encoded_actions=[f.encode() for f in pointers],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        run_recording(recorder=XCTestScreenRecorder(driver,fps=30),timing=timing,commands=commands,
                      perform=perform,guard=guard,out=a.out,revision=a.wda_revision,metadata=metadata)
        from trueskate_ai.research.curve_measurement import admit_recording
        admit_recording(a.out,a.wda_revision)
        print(a.out,flush=True)
    finally:worker.disconnect()
if __name__=='__main__':main()
