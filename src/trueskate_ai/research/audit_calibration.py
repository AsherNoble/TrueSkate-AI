"""Five-point consensus with independent middle marker and full native decode."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from trueskate_ai.research.audit_video import frame_pts,read_native_frames,_decode_source_frames
from trueskate_ai.research.curved_audit import save_new
from trueskate_ai.collection.die_five_calibration import detect_die_five_onset,die_five_points
from trueskate_ai.collection.tap_timing_calibration import fit_two_anchor_timeline
from trueskate_ai.collection.wda_action_timing import validate_action_timing_report

def fit_markers(stamps,onsets,frame_s):
    if set(onsets)!={'start','middle','end'} or any(v is None for v in onsets.values()):raise ValueError('missing marker')
    fit=fit_two_anchor_timeline(stamps['start'],onsets['start'],stamps['end'],onsets['end'],min_anchor_span_s=55.)
    error=onsets['middle']-fit.video_time_s(stamps['middle'])
    if not onsets['start']<onsets['middle']<onsets['end'] or abs(error)>2*frame_s:
        raise ValueError('independent middle marker exceeds two native frames')
    return fit,error

def verify_recording(out,revision):
    out=Path(out);result={'accepted':False,'training_admission':False}
    try:
        planned=json.loads((out/'planned.json').read_text());execution=json.loads((out/'execution.json').read_text())
        if execution['error']:raise ValueError('execution failed')
        records=validate_action_timing_report(json.loads((out/'wda-timing.json').read_text()),expected_revision=revision,expected_count=len(planned['commands']))
        pts=frame_pts(out/'original.mov');_decode_source_frames(out/'original.mov',pts,keep=False)
        result.update(frame_count=len(pts),native_decoded_count=len(pts),original_pts_s=pts.tolist())
        onsets={};stamps={};detections={}
        for i,c in enumerate(planned['commands']):
            if c['kind']!='control':continue
            stamp=records[i]['submitted_to_ios'];approx=stamp['epoch_s']-execution['video']['started_at_epoch_s']
            frames,times,numbers=read_native_frames(out/'original.mov',pts,start=approx-.75,end=approx+.75)
            detection=detect_die_five_onset(frames,times,command_s=approx,points=die_five_points(71),reference_window_s=.75)
            role=c['role'];detections[role]=dict(**asdict(detection),source_frames=numbers)
            result['detections']=detections
            if not detection.accepted:raise ValueError('missing/disagreeing five-point marker: '+role)
            onsets[role]=detection.onset_s;stamps[role]=stamp['monotonic_s']
        fit,error=fit_markers(stamps,onsets,float(np.median(np.diff(pts))))
        result.update(accepted=True,fit=asdict(fit),heldout_middle_error_s=error,control_onsets_s=onsets)
    except Exception as exc:
        result['error']=str(exc);save_new(out/'calibration.json',result);raise
    save_new(out/'calibration.json',result)
    return result
