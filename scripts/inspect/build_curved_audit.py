"""Build native-frame audit with per-frame commanded targets and private source mapping."""
import argparse,json,random,hashlib
from pathlib import Path
import cv2
from trueskate_ai.research.curved_audit import verify_manifest,SEED,digest,save_new
from trueskate_ai.research.audit_video import frame_pts,_decode_source_frames
from trueskate_ai.sim.timed_waypoints import TimedWaypoints

def build(manifest_path,recordings,out):
    frozen=json.loads(manifest_path.read_text());verify_manifest(frozen);IDENTITY=frozen['identity']
    if recordings.resolve().is_relative_to(out.resolve()):raise ValueError('private recordings must be outside web root')
    out.mkdir(parents=True,exist_ok=False)
    paths={p['path_id']:p for p in frozen['paths']};clips=[];private=[]
    for n,segment in enumerate(frozen['recordings'],1):
        run=recordings/f'recording_{n}';cal=json.loads((run/'calibration.json').read_text())
        execution=json.loads((run/'execution.json').read_text())
        if not cal['accepted'] or execution['error'] or execution['manifest_sha256']!=frozen['sha256']:raise ValueError('unverified segment')
        report=json.loads((run/'wda-timing.json').read_text());pts=frame_pts(run/'original.mov');fit=cal['fit'];windows=[]
        for i,c in enumerate(segment['commands']):
            if c['kind']!='sample':continue
            path=paths[c['path_id']];timed=TimedWaypoints(tuple(map(tuple,path['points'])),tuple(path['times_ms']))
            onset=fit['intercept_s']+fit['rate']*report['records'][i]['submitted_to_ios']['monotonic_s']
            selected=[j for j,t in enumerate(pts) if onset-.7<=t<onset+2.3]
            if not selected or pts[0]>onset-.7 or pts[-1]<onset+2.3:raise ValueError('truncated clip')
            token=hashlib.sha256(f'{IDENTITY}:{n}:{c["path_id"]}:blind'.encode()).hexdigest()[:20]
            (out/token).mkdir();clip=dict(id=token,frames=[]);clips.append(clip)
            private.append(dict(id=token,segment=n,device=segment['device'],park=segment['park'],**path,onset_s=onset,
                                source_frames=selected,source_pts_s=[float(pts[j]) for j in selected]))
            windows.append((clip,selected[0],selected[-1],onset,timed,fit['rate']))
        count=0
        def inspect(frame):
            nonlocal count
            j=count;count+=1
            for clip,start,end,onset,timed,rate in windows:
                if not start<=j<=end:continue
                filename=f'{clip["id"]}/{j-start:03d}.jpg'
                if not cv2.imwrite(str(out/filename),frame,[cv2.IMWRITE_JPEG_QUALITY,95]):raise ValueError('frame write failed')
                target=timed.position((float(pts[j])-onset)/rate*1000)
                clip['frames'].append(dict(path=filename,time_s=float(pts[j]-pts[start]),target=target))
        _decode_source_frames(run/'original.mov',pts,keep=False,inspect=inspect)
        print(f'exported segment {n}/10',flush=True)
    random.Random(SEED+999).shuffle(clips)
    if len(clips)!=100:raise ValueError('100 clips required')
    data=dict(identity=IDENTITY,bundle_sha256=digest(clips),clips=clips)
    save_new(recordings/'audit-private-source-map.json',dict(identity=IDENTITY,bundle_sha256=data['bundle_sha256'],items=private))
    (out/'data.js').write_text('const DATA='+json.dumps(data).replace('<','\\u003c')+';\n')
    (out/'index.html').write_text((Path(__file__).with_name('templates')/'curved_audit.html').read_text())
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','recordings','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build(a.manifest,a.recordings,a.out)
