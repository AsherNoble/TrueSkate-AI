"""Build native-frame audit with per-frame commanded targets and private source mapping."""
import argparse,json,random,hashlib
from pathlib import Path
import cv2
from trueskate_ai.research.curved_audit import verify_manifest,SEED,digest,save_new
from trueskate_ai.research.audit_video import frame_pts,_decode_source_frames
from trueskate_ai.sim.timed_waypoints import TimedWaypoints
from trueskate_ai.research.review_provenance import bundle, curved_execution, file_sha256

def _same_conditions(a,b):
    # Slots/marker resets may differ in an authorized replacement. The full
    # ordered sample specifications and outgoing payloads must remain identical.
    samples=lambda r:[{k:v for k,v in c.items() if k!='slot_s'} for c in r['commands'] if c['kind']=='sample']
    return a['device']==b['device'] and a['park']==b['park'] and samples(a)==samples(b)

def build(manifest_path,recordings,out,replacement=None):
    """replacement: (manifest_path, recordings, segments) for user-authorized re-runs of named segments."""
    frozen=json.loads(manifest_path.read_text());verify_manifest(frozen);IDENTITY=frozen['identity']
    sources={n:(frozen,recordings) for n in range(1,len(frozen['recordings'])+1)}
    if replacement:
        rm,rr,segments=replacement;other=json.loads(rm.read_text());verify_manifest(other)
        if other['paths']!=frozen['paths']:raise ValueError('replacement paths differ')
        for n in segments:
            if not _same_conditions(other['recordings'][n-1],frozen['recordings'][n-1]):raise ValueError('replacement conditions differ')
            sources[n]=(other,rr)
    for _,rec in sources.values():
        if rec.resolve().is_relative_to(out.resolve()):raise ValueError('private recordings must be outside web root')
    out.mkdir(parents=True,exist_ok=False)
    paths={p['path_id']:p for p in frozen['paths']};clips=[];private=[];proofs=[]
    for n in sorted(sources):
        source,rec=sources[n];segment=source['recordings'][n-1]
        run=rec/f'recording_{n}';cal=json.loads((run/'calibration.json').read_text())
        execution=json.loads((run/'execution.json').read_text())
        planned=json.loads((run/'planned.json').read_text())
        report=json.loads((run/'wda-timing.json').read_text())
        proof=curved_execution(source,n,planned,execution,report)
        if cal.get('accepted') is not True:raise ValueError('unverified segment')
        proofs.append(dict(**proof,frozen_manifest=source,segment=n,calibration=cal,
                           original_sha256=file_sha256(run/'original.mov')))
        pts=frame_pts(run/'original.mov');fit=cal['fit'];windows=[]
        for i,c in enumerate(segment['commands']):
            if c['kind']!='sample':continue
            path=paths[c['path_id']];timed=TimedWaypoints(tuple(map(tuple,path['points'])),tuple(path['times_ms']))
            onset=fit['intercept_s']+fit['rate']*report['records'][i]['submitted_to_ios']['monotonic_s']
            selected=[j for j,t in enumerate(pts) if onset-.7<=t<onset+2.3]
            if not selected or pts[0]>onset-.7 or pts[-1]<onset+2.3:raise ValueError('truncated clip')
            token=hashlib.sha256(f'{IDENTITY}:{n}:{c["path_id"]}:blind'.encode()).hexdigest()[:20]
            (out/token).mkdir();clip=dict(id=token,frames=[]);clips.append(clip)
            private.append(dict(id=token,segment=n,source_manifest=source['identity'],source_manifest_sha256=source['sha256'],device=segment['device'],park=segment['park'],**path,onset_s=onset,
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
                clip['frames'].append(dict(path=filename,time_s=float(pts[j]-pts[start]),target=target,sha256=file_sha256(out/filename)))
        _decode_source_frames(run/'original.mov',pts,keep=False,inspect=inspect)
        print(f'exported segment {n}/10',flush=True)
    random.Random(SEED+999).shuffle(clips)
    if len(clips)!=100:raise ValueError('100 clips required')
    descriptor=bundle('blind-curved-execution-v2',clips,dict(frozen_manifest=frozen,recordings=proofs),
                      dict(identity=IDENTITY,items=private))
    data=dict(identity=IDENTITY,schema=descriptor['schema'],version=descriptor['version'],bundle_sha256=descriptor['bundle_sha256'],clips=clips)
    save_new(recordings/'audit-private-source-map-v2.json',descriptor)
    (out/'data.js').write_text('const DATA='+json.dumps(data).replace('<','\\u003c')+';\n')
    (out/'index.html').write_text((Path(__file__).with_name('templates')/'curved_audit.html').read_text())
    (out/'review_integrity.js').write_bytes((Path(__file__).with_name('templates')/'review_integrity.js').read_bytes())
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','recordings','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--replacement',nargs=3,metavar=('MANIFEST','RECORDINGS','SEGMENTS'),
                   help='take comma-separated segments from an authorized re-run')
    a=p.parse_args()
    r=(Path(a.replacement[0]),Path(a.replacement[1]),[int(x) for x in a.replacement[2].split(',')]) if a.replacement else None
    build(a.manifest,a.recordings,a.out,r)
