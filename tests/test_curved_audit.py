import json
from collections import Counter
import pytest
from trueskate_ai.research.curved_audit import manifest,verify_manifest,intervals
from trueskate_ai.sim.timed_waypoints import TimedWaypoints
from trueskate_ai.data.control_hitboxes import segment_is_safe
from trueskate_ai.research.audit_calibration import fit_markers
from trueskate_ai.collection.die_five_calibration import die_five_consensus

def test_frozen_balance_and_payload():
    m=manifest();verify_manifest(json.loads(json.dumps(m)));assert m==manifest()
    assert len(m['paths'])==50 and len(m['recordings'])==10
    assert sorted(Counter(p['waypoint_count'] for p in m['paths']).values())==[12,12,13,13]
    assert set(Counter(p['timing_profile'] for p in m['paths']).values())=={10}
    assert set(Counter((p['family'],p['duration_ms']) for p in m['paths']).values())=={2}
    for p in m['paths']:
        a=p['payload']['actions'];assert len(a)==1
        a=a[0]['actions'];assert Counter(x['type'] for x in a)==dict(pointerMove=p['waypoint_count'],pointerDown=1,pointerUp=1)
        moves=[x for x in a if x['type']=='pointerMove'][1:]
        assert sum(x['duration'] for x in moves)==p['duration_ms'] and min(x['duration'] for x in moves)>0
        q=p['quantized_points'];assert all(segment_is_safe((x/414,y/896),(v/414,w/896)) for (x,y),(v,w) in zip(q,q[1:]))
        if p['timing_profile']=='pause':assert any(a==b for a,b in zip(q,q[1:]))
    for device in ('iPhone_XR','iPhone_XR2'):
        ids=[c['path_id'] for r in m['recordings'] if r['device']==device for c in r['commands'] if c['kind']=='sample']
        assert len(ids)==len(set(ids))==50
    assert [r['device'] for r in m['recordings']]==['iPhone_XR','iPhone_XR2']*5
    m['paths'][0]['duration_ms']=0
    with pytest.raises(ValueError):verify_manifest(m)

def test_position_pause_reverse_and_cap():
    p=TimedWaypoints(((.5,.5),(.75,.5),(.75,.5),(.5,.5)),(0,100,200,300))
    assert p.position(-1) is None and p.position(300) is None
    assert p.position(100)==p.position(150)==p.quantized()[1]
    assert p.position(50)==p.position(250)
    assert p.position(0)==p.quantized()[0]
    with pytest.raises(ValueError):TimedWaypoints(((.5,.5),)*16,tuple(range(16)))
    with pytest.raises(ValueError):TimedWaypoints(((.5,.5),)*2,(0,0))
    with pytest.raises(ValueError):TimedWaypoints(((.5,.5),(0,0)),(0,100)).payload()

def test_consensus_and_independent_middle():
    assert die_five_consensus([10,10,11,10,None])[0]==10
    assert die_five_consensus([10,10,20,20,None])[0] is None
    stamps={'start':1.,'middle':30.,'end':57.}
    fit,error=fit_markers(stamps,dict(start=1.1,middle=30.13,end=57.1),1/30)
    assert abs(error-.03)<1e-8
    with pytest.raises(ValueError):fit_markers(stamps,dict(start=1.1,middle=30.2,end=57.1),1/30)
    with pytest.raises(ValueError):fit_markers(stamps,dict(start=1.1,end=57.1),1/30)

def test_integer_intervals():
    assert sum(intervals([1,7,3],150))==150
    assert min(intervals([1e-10,100],150))==1

def test_importer_counts_only_explicit_ratings():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/inspect/report_curved_audit.py'
    spec=importlib.util.spec_from_file_location('audit_report',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mapping=dict(identity='fixture',bundle_sha256='a',items=[dict(id=str(i),family='arcs',duration_ms=150,waypoint_count=5,device='XR1',park='Inbound') for i in range(2)])
    export=dict(schema='blind-curved-execution-v1',bundle_sha256='a',assessments={'0':dict(rating='major',comments='test')})
    r=module.report(mapping,export)
    assert r['reviewed']==r['missing']==1 and r['overall']['major']==1 and r['overall']['good']==0
    export['bundle_sha256']='wrong'
    with pytest.raises(ValueError):module.report(mapping,export)

def test_native_decode_requires_exact_count(tmp_path):
    import subprocess,numpy as np
    from trueskate_ai.research.audit_video import frame_pts,_decode_source_frames
    video=tmp_path/'native.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=32x64:r=30','-t','0.5','-c:v','libx264',str(video)],check=True)
    pts=frame_pts(video)
    frames,times,indices=_decode_source_frames(video,pts)
    assert len(frames)==len(times)==len(indices)==len(pts)==15
    with pytest.raises(ValueError):_decode_source_frames(video,np.append(pts,pts[-1]+1/30),keep=False)
