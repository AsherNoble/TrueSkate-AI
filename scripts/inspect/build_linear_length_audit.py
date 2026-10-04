"""Export opaque, shuffled native-frame clips; private mapping stays outside web root."""
import argparse
import json
import hashlib
import time
from pathlib import Path
import cv2
from trueskate_ai.research.linear_speed_probe import verify_manifest
from trueskate_ai.research.linear_length_probe import blind_key
from trueskate_ai.research.curve_measurement import frame_pts, _decode_source_frames
from trueskate_ai.research.review_provenance import verify_execution,bundle,file_sha256


def build(manifest_path, recordings, out, wait_seconds=0):
    deadline=time.monotonic()+wait_seconds
    frozen=json.loads(manifest_path.read_text());verify_manifest(frozen)
    key=blind_key(frozen)
    out.mkdir(parents=True,exist_ok=False)
    public={}; private=[]; proofs={}
    private_path=recordings/'audit-private-source-map-v2.json'
    if private_path.exists():raise ValueError('preserve existing private map')
    for number,commands in enumerate(frozen['recordings'],1):
        run=recordings/f'recording_{number}'
        while not (run/'timing-diagnostic.json').exists():
            if time.monotonic() >= deadline:raise ValueError('missing completed timing diagnostic')
            time.sleep(2)
        execution=json.loads((run/'execution.json').read_text())
        if execution['error']:raise ValueError('execution incomplete')
        timing=json.loads((run/'timing-diagnostic.json').read_text())
        if not timing['timing_checks_pass']:raise ValueError('timing calibration failed')
        report=json.loads((run/'wda-timing.json').read_text());fit=timing['fit']
        planned=json.loads((run/'planned.json').read_text())
        proof=verify_execution(frozen,commands,planned,execution,report,[c['payload'] for c in commands])
        proof.update(original_sha256=file_sha256(run/'original.mov'),calibration=timing)
        proofs[str(number)]=proof
        pts=frame_pts(run/'original.mov');windows=[]
        for i,event in enumerate(execution['events']):
            c=event['spec']
            if c['kind']!='diagnostic':continue
            item=next(x for x in key['items'] if x['command_id']==c['command_id'])
            onset=fit['intercept_s']+fit['rate']*report['records'][i]['submitted_to_ios']['monotonic_s']
            # Same window width for every duration: no duration-dependent cut cue.
            start,on_end=onset-.7,onset+1.5
            selected=[j for j,t in enumerate(pts) if start<=t<=on_end]
            if not selected:raise ValueError('empty clip')
            token=item['token'];(out/token).mkdir()
            public[token]=dict(id=token,frames=[])
            private.append(dict(**item,onset_s=onset,source_frames=selected,
                                source_pts_s=[float(pts[j]) for j in selected]))
            windows.append((token,selected[0],selected[-1],float(pts[selected[0]])))
        count=[0]
        def inspect(frame):
            j=count[0];count[0]+=1
            for token,start,end,first_pts in windows:
                if not start<=j<=end:continue
                n=j-start;filename=f'{token}/{n:03d}.jpg'
                if not cv2.imwrite(str(out/filename),frame,[cv2.IMWRITE_JPEG_QUALITY,90]):raise ValueError('frame write failed')
                public[token]['frames'].append(dict(path=filename,time_s=float(pts[j])-first_pts,sha256=file_sha256(out/filename)))
        _decode_source_frames(run/'original.mov',pts,keep=False,inspect=inspect)
        print(f'exported recording {number}/15',flush=True)
    clips=[public[x['token']] for x in key['items']]
    if len(clips)!=135 or any(len(c['frames'])<2 for c in clips):raise ValueError('incomplete audit')
    private_bundle=bundle('blind-linear-length-v2',clips,dict(frozen_manifest=frozen,executions=proofs),dict(key=key,items=private))
    data=dict(schema=private_bundle['schema'],version=2,bundle_sha256=private_bundle['bundle_sha256'],clips=clips)
    (out/'data.js').write_text('const DATA='+json.dumps(data).replace('<','\\u003c')+';\n')
    (out/'index.html').write_text((Path(__file__).with_name('templates')/'linear_length_audit.html').read_text())
    (out/'review_integrity.js').write_bytes((Path(__file__).with_name('templates')/'review_integrity.js').read_bytes())
    private_path.write_text(json.dumps(private_bundle,indent=2)+'\n')
    print('135 opaque clips complete',flush=True)
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','recordings','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--wait-seconds',type=float,default=0,help='Bounded wait for verified local segments')
    a=p.parse_args();build(a.manifest,a.recordings,a.out,a.wait_seconds)
