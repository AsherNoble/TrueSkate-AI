"""Source-PTS video admission and command-blind orange trail measurements.

Extraction accepts only pixels, PTS and frozen colour parameters. Command
geometry is consulted exclusively in subsequent scoring, never mask selection.
"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from trueskate_ai.research.curve_protocol import PROTOCOL,save_new
from trueskate_ai.collection.tap_timing_calibration import detect_tap_onset,fit_two_anchor_timeline
from trueskate_ai.collection.wda_action_timing import validate_action_timing_report


def frame_pts(video):
    result=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
                                    '-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)])
    frames=json.loads(result)['frames']
    pts=np.asarray([float(f['best_effort_timestamp_time']) for f in frames])
    if len(pts)<2 or not np.isfinite(pts).all() or np.any(np.diff(pts)<=0):
        raise ValueError('missing or nonchronological original frame PTS')
    if abs(float(np.median(np.diff(pts)))-1/30)>.005:
        raise ValueError('recording is not native ~30fps')
    return pts


def read_native_frames(video,pts,*,start=-np.inf,end=np.inf):
    # OpenCV emitted 1773 frames for an XCTest file with 1772 source PTS.
    # Use the same FFmpeg decoding/edit-list semantics as ffprobe and the
    # canonical exact-source-frame extractor. Never truncate to make counts fit.
    return _decode_source_frames(video,pts,start=start,end=end)


def _decode_source_frames(video,pts,*,start=-np.inf,end=np.inf,keep=True,inspect=None):
    metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                        '-show_entries','stream=width,height','-of','json',str(video)]))['streams'][0]
    width,height=metadata['width'],metadata['height']
    selected=[i for i,t in enumerate(pts) if start<=t<=end]
    if not selected:return [],[],[]
    args=['ffmpeg','-v','error','-i',str(video),'-map','0:v:0']
    if selected!=list(range(len(pts))):
        args+=['-vf',f'select=between(n\\,{selected[0]}\\,{selected[-1]})']
    args+=['-fps_mode','passthrough','-f','rawvideo','-pix_fmt','bgr24','pipe:1']
    # Bound stderr separately so a full diagnostic pipe cannot deadlock.
    import tempfile
    frames=[];size=width*height*3;count=0
    with tempfile.TemporaryFile() as stderr:
        process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=stderr)
        try:
            while True:
                raw=process.stdout.read(size)
                if not raw:break
                if len(raw)!=size:raise ValueError('incomplete decoded source frame')
                if count>=len(selected):raise ValueError('decoded frame count exceeds original PTS selection')
                if keep or inspect is not None:
                    image=np.frombuffer(raw,np.uint8).reshape(height,width,3)
                    if keep:frames.append(image.copy())
                    if inspect is not None:inspect(image)
                count+=1
            status=process.wait()
            if status:
                stderr.seek(0);raise ValueError('FFmpeg source decode failed: '+stderr.read().decode(errors='replace'))
            if count!=len(selected):raise ValueError('decoded frame count differs from original PTS selection')
        finally:
            process.stdout.close()
            if process.poll() is None:process.kill();process.wait()
    return frames,[float(pts[i]) for i in selected],selected


def admit_recording(out,revision):
    out=Path(out)
    planned=json.loads((out/'planned.json').read_text())
    execution=json.loads((out/'execution.json').read_text())
    timing=json.loads((out/'wda-timing.json').read_text())
    try:
        if execution['error']:raise ValueError('execution failed')
        records=validate_action_timing_report(timing,expected_revision=revision,expected_count=len(planned['commands']))
        pts=frame_pts(out/'original.mov')
        from trueskate_ai.collection.gameplay_filter import is_menu_frame,is_editor_frame
        def inspect(frame):
            png=cv2.imencode('.png',frame,[cv2.IMWRITE_PNG_COMPRESSION,0])[1].tobytes()
            if is_editor_frame(png) or is_menu_frame(png,allow_idle_navigation=True):
                raise ValueError('original recording contains gameplay contamination')
        _decode_source_frames(out/'original.mov',pts,keep=False,inspect=inspect)
        started=execution['video']['started_at_epoch_s']
        onsets={}; stamps={}; source_frames={}
        for i,spec in enumerate(planned['commands']):
            if spec['kind']!='control':continue
            approximate=records[i]['submitted_to_ios']['epoch_s']-started
            frames,times,indices=read_native_frames(out/'original.mov',pts,start=approximate-.7,end=approximate+.7)
            detection=detect_tap_onset(frames,times,point_xy=(.5,.5),command_s=approximate)
            if detection is None:raise ValueError(f"control {spec['role']} onset not observable")
            role=spec['role'];onsets[role]=detection.onset_s
            stamps[role]=records[i]['submitted_to_ios']['monotonic_s']
            source_frames[role]=indices[times.index(detection.onset_s)]
        fit=fit_two_anchor_timeline(stamps['start'],onsets['start'],stamps['end'],onsets['end'],min_anchor_span_s=55.)
        middle_error=onsets['middle']-fit.video_time_s(stamps['middle'])
        frame_s=float(np.median(np.diff(pts)))
        if abs(middle_error)>2*frame_s:raise ValueError('held-out middle control exceeds two native frames')
        if not (onsets['start']<onsets['middle']<onsets['end']):raise ValueError('control onsets overlap')
        result=dict(accepted=True,fit=asdict(fit),control_onsets_s=onsets,control_source_frames=source_frames,
                    heldout_middle_error_s=middle_error,frame_count=len(pts),native_frame_s=frame_s,
                    max_frame_gap_s=float(np.max(np.diff(pts))),
                    original_pts_s=pts.tolist(),training_admission=False)
    except Exception as exc:
        save_new(out/'admission.json',dict(accepted=False,error=str(exc),training_admission=False))
        raise
    save_new(out/'admission.json',result)
    return result


def orange_mask(image,parameters=None):
    parameters=parameters or PROTOCOL['orange_hsv']
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv,np.array(parameters['lower'],np.uint8),np.array(parameters['upper'],np.uint8))>0


def thin_mask(mask):
    """Zhang-Suen thinning; preserves pixel geometry, branches and gaps."""
    ys,xs=np.nonzero(mask)
    if not len(xs):return np.zeros_like(mask,dtype=bool)
    y0,y1=int(ys.min()),int(ys.max())+1
    x0,x1=int(xs.min()),int(xs.max())+1
    result=np.zeros_like(mask,dtype=bool)
    result[y0:y1,x0:x1]=_thin_core(mask[y0:y1,x0:x1])
    return result


def _thin_core(mask):
    a=np.pad(mask.astype(np.uint8),1)
    while True:
        removed=0
        for step in (0,1):
            p=a[1:-1,1:-1]
            n=[a[:-2,1:-1],a[:-2,2:],a[1:-1,2:],a[2:,2:],a[2:,1:-1],a[2:,:-2],a[1:-1,:-2],a[:-2,:-2]]
            count=sum(n)
            transitions=sum((n[i]==0)&(n[(i+1)%8]==1) for i in range(8))
            if step==0:
                condition=(n[0]*n[2]*n[4]==0)&(n[2]*n[4]*n[6]==0)
            else:
                condition=(n[0]*n[2]*n[6]==0)&(n[0]*n[4]*n[6]==0)
            remove=(p==1)&(count>=2)&(count<=6)&(transitions==1)&condition
            removed+=int(remove.sum());p[remove]=0
        if not removed:break
    return a[1:-1,1:-1]>0


def skeleton_points(mask):
    y,x=np.nonzero(mask)
    # Full-screen render coordinates map to normalized screen coordinates.
    return np.stack([x/mask.shape[1],y/mask.shape[0]],axis=-1)


def extract_observation(frames,times,*,baseline_count,parameters=None):
    """No requested path or segment metadata enters this API."""
    if not frames or len(frames)!=len(times) or not 1<=baseline_count<len(frames):
        raise ValueError('native frames, PTS and pre-command baseline required')
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('nonchronological source PTS')
    shape=frames[0].shape
    if any(f.shape!=shape for f in frames):raise ValueError('inconsistent original frame dimensions')
    background=np.logical_or.reduce([orange_mask(f,parameters) for f in frames[:baseline_count]])
    previous=np.zeros(shape[:2],bool);union=previous.copy();contacts=[];growth=[];areas=[];ambiguous=[]
    for index,(frame,pts) in enumerate(zip(frames[baseline_count:],times[baseline_count:]),start=baseline_count):
        mask=orange_mask(frame,parameters)&~background
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        components=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=8]
        if len(components)>1:ambiguous.append(index)
        cleaned=np.isin(labels,components)
        added=cleaned&~cv2.dilate(previous.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        union|=cleaned;areas.append(dict(frame=index,pts_s=pts,pixels=int(cleaned.sum())))
        if added.sum()>=4:
            growth.append(dict(frame=index,pts_s=pts,new_pixels=int(added.sum())))
            skeleton=thin_mask(cleaned)
            neighbors=convolve(skeleton.astype(int),np.ones((3,3),int),mode='constant')-skeleton
            endpoints=skeleton&(neighbors==1)
            candidates=endpoints&cv2.dilate(added.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
            y,x=np.nonzero(candidates)
            if len(x)==1 and len(components)==1:
                contacts.append(dict(frame=index,pts_s=pts,xy=[float(x[0]/shape[1]),float(y[0]/shape[0])]))
        previous=cleaned
    skeleton=thin_mask(union)
    count,_,_,_=cv2.connectedComponentsWithStats(skeleton.astype(np.uint8),8)
    neighbors=convolve(skeleton.astype(int),np.ones((3,3),int),mode='constant')-skeleton
    branches=int(np.sum(skeleton&(neighbors>2)))
    reasons=[]
    if not skeleton.any():reasons.append('no visible orange centreline')
    if ambiguous:reasons.append('competing orange components')
    if count>2:reasons.append('gapped centreline')
    if branches:reasons.append('branched or overlapping centreline')
    if len(contacts)<3:reasons.append('insufficient identifiable contact positions')
    # Growth bounds motion only while the visible leading endpoint is identified.
    duration_evaluable=len(growth)>=2 and len(contacts)>=3 and not ambiguous
    if not duration_evaluable:reasons.append('visible motion boundaries indeterminate')
    contact_frames=[c['frame'] for c in contacts]
    continuous=(duration_evaluable and not reasons and contact_frames
                and contact_frames[0]==growth[0]['frame'] and contact_frames[-1]==growth[-1]['frame']
                and all(b-a==1 for a,b in zip(contact_frames,contact_frames[1:])))
    whole_trail_once=(bool(areas) and bool(union.any()) and
                      next((a['pixels'] for a in areas if a['pixels']),0)>=.95*int(union.sum()))
    return dict(centreline=skeleton_points(skeleton).tolist(),contacts=contacts,growth=growth,areas=areas,
                background_orange_fraction=float(background.mean()),branches=branches,indeterminate_reasons=reasons,
                shape_evaluable=not any(r in reasons for r in ['no visible orange centreline','competing orange components','gapped centreline','branched or overlapping centreline']),
                duration_evaluable=duration_evaluable,
                visible_motion_interval_s=[growth[0]['pts_s'],growth[-1]['pts_s']] if duration_evaluable else None,
                collapse_suspected=bool(whole_trail_once or (len(growth)==1 and skeleton.any())),
                interruption=False if continuous else None,
                frame_width=shape[1],frame_height=shape[0])


def symmetric_distance(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=2 or b.shape[1]!=2 or not len(a) or not len(b):
        raise ValueError('nonempty finite 2D centrelines required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite centreline')
    return float(max(cKDTree(a).query(b)[0].max(),cKDTree(b).query(a)[0].max()))


def score_observation(observation,compiled,curve,*,onset_s,frame_s,uncertainty=0.):
    """Score immutable observations after extraction; never overwrite their positions."""
    from trueskate_ai.data.touch_labels import cubic_command_label
    shape=None;command_shape=None;endpoint=None
    dense_t=np.linspace(0,compiled.boundaries_ms[-1]/1000,10001)
    intended=curve.evaluate(np.linspace(0,1,10001))
    command=np.array([[l.x,l.y] for l in (cubic_command_label(compiled,t) for t in dense_t)])
    # Conservative allowance for finite reference sampling from maximum speed.
    max_speed=3*float(np.linalg.norm(np.diff(np.asarray(curve.coefficients),axis=0),axis=1).max())
    reference_allowance=max_speed/10000
    if observation['shape_evaluable']:
        shape=symmetric_distance(observation['centreline'],intended)+reference_allowance
        command_shape=symmetric_distance(observation['centreline'],command)+reference_allowance
    contacts=observation['contacts']
    command_errors=[];cubic_errors=[]
    for contact in contacts:
        t=contact['pts_s']-onset_s
        label=cubic_command_label(compiled,t)
        if label.active:
            # Retain one-frame alignment uncertainty as a spatial allowance.
            time_allowance=max_speed*frame_s/curve.duration_s
            command_errors.append(float(np.linalg.norm(np.array(contact['xy'])-[label.x,label.y]))+time_allowance)
            cubic_errors.append(float(np.linalg.norm(np.array(contact['xy'])-curve.evaluate(np.clip(t/(compiled.boundaries_ms[-1]/1000),0,1))))+time_allowance)
    total=compiled.boundaries_ms[-1]/1000
    end_contacts=[c for c in contacts if abs(c['pts_s']-onset_s-total)<=2*frame_s]
    if end_contacts:
        endpoint=float(np.linalg.norm(np.array(end_contacts[-1]['xy'])-curve.evaluate(1)))
    duration_error=None
    if observation['duration_evaluable']:
        first,last=observation['visible_motion_interval_s']
        duration_error=abs(last-first-total)+2*frame_s
    progress=[]
    if curve.duration_s>=.6:
        for u in (.25,.5,.75):
            target=curve.evaluate(u)
            candidates=[c for c in contacts if np.linalg.norm(np.array(c['xy'])-target)<=.005]
            # Multiple visits on a reversal cannot be resolved by snapping to expected time.
            if not candidates or max(c['pts_s'] for c in candidates)-min(c['pts_s'] for c in candidates)>2*frame_s:
                progress.append(None)
            else:
                mid=(max(c['pts_s'] for c in candidates)+min(c['pts_s'] for c in candidates))/2
                progress.append(abs(mid-(onset_s+u*total))+frame_s)
    return dict(shape_error=shape,command_shape_error=command_shape,endpoint_error=endpoint,
                max_time_aligned_command_error=max(command_errors,default=None),
                max_time_aligned_cubic_error=max(cubic_errors,default=None),
                duration_error_s=duration_error,progress_errors_s=progress,
                uncertainty=uncertainty,frame_time_uncertainty_s=frame_s,
                collapse=observation['collapse_suspected'],interruption=observation.get('interruption'),
                evaluable=shape is not None and endpoint is not None and duration_error is not None and all(p is not None for p in progress)
                          and observation.get('interruption') is not None)


def measure_recording(out, manifest_path):
    out=Path(out)
    admission=json.loads((out/'admission.json').read_text())
    if not admission['accepted']:raise ValueError('recording not admitted')
    planned=json.loads((out/'planned.json').read_text())
    from trueskate_ai.research.review_provenance import curve_execution,file_sha256
    frozen=json.loads(Path(manifest_path).read_text())
    execution=json.loads((out/'execution.json').read_text())
    report=json.loads((out/'wda-timing.json').read_text())
    proof=curve_execution(frozen,planned,execution,report)
    proof.update(frozen_manifest=frozen,original_sha256=file_sha256(out/'original.mov'),calibration=admission)
    records=report['records']
    pts=np.asarray(admission['original_pts_s'])
    result=[]
    from trueskate_ai.sim.cubic_curve import CubicInTime,compile_curve
    for i,spec in enumerate(planned['commands']):
        if spec['kind']!='diagnostic':continue
        onset=admission['fit']['intercept_s']+admission['fit']['rate']*records[i]['submitted_to_ios']['monotonic_s']
        total=spec['compiled']['boundaries_ms'][-1]/1000
        frames,times,indices=read_native_frames(out/'original.mov',pts,start=onset-.4,end=onset+total+.25)
        baseline_count=sum(t<onset-.1 for t in times)
        observation=extract_observation(frames,times,baseline_count=baseline_count)
        for key in ('contacts','growth','areas'):
            for row in observation[key]:row['source_frame']=indices[row['frame']]
        kwargs={'n':spec['requested_n']} if spec['requested_n'] is not None else {'max_segment_duration_ms':spec['rule_ms']}
        curve=CubicInTime.from_dict(spec['curve']);compiled=compile_curve(curve,**kwargs)
        scored=score_observation(observation,compiled,curve,onset_s=onset,frame_s=admission['native_frame_s'])
        result.append(dict(command_id=spec['command_id'],device=planned['device'],stage=spec['stage'],
                           family=spec['family'],duration_s=curve.duration_s,curve_id=spec['curve_id'],repeat=spec['repeat'],
                           rule_ms=spec['rule_ms'],n=spec['compiled']['n'],
                           min_segment_duration_ms=spec['compiled']['min_segment_duration_ms'],
                           max_segment_duration_ms=spec['compiled']['max_segment_duration_ms'],
                           offline_eligible=spec['offline_eligible'],observation=observation,metrics=scored,
                           request_overhead_s=records[i]['request_finished']['monotonic_s']-records[i]['submitted_to_ios']['monotonic_s']-total,
                           submission_to_completion_s=records[i]['ios_completion_callback']['monotonic_s']-records[i]['submitted_to_ios']['monotonic_s'],
                           original_video=str(out/'original.mov'),onset_s=onset,execution_provenance=proof))
    save_new(out/'measurement.json',dict(rows=result,extractor_parameters=PROTOCOL['orange_hsv'],
                                        automated_evaluable=sum(r['metrics']['evaluable'] for r in result)))
    return result
