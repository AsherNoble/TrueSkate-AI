"""Preserve separate timing diagnostics and the first gameplay flag; never override admission.

Run with one absolute recording-directory argument. Outputs must not already exist.
"""
import json, cv2, numpy as np
from pathlib import Path
from dataclasses import asdict
from trueskate_ai.research.curve_measurement import frame_pts,read_native_frames,_decode_source_frames
from trueskate_ai.collection.gameplay_filter import is_menu_frame,is_editor_frame
from trueskate_ai.collection.tap_timing_calibration import detect_tap_onset,fit_two_anchor_timeline
from trueskate_ai.collection.wda_action_timing import validate_action_timing_report
import sys
run=Path(sys.argv[1]);
if any((run/name).exists() for name in ("timing-diagnostic.json", "flag-diagnostic.json", "first-flag.png")):
    raise SystemExit("Preserve existing diagnostics; use a new directory containing recording inputs")
revision="b5ace21788b5f5dc4cf0e0759f8bb8a79ab83ae6"
planned=json.loads((run/"planned.json").read_text()); execution=json.loads((run/"execution.json").read_text()); report=json.loads((run/"wda-timing.json").read_text())
records=validate_action_timing_report(report,expected_revision=revision,expected_count=len(planned["commands"]))
pts=frame_pts(run/"original.mov"); results={"training_admission":False,"separate_timing_diagnostic":True,"frame_count":len(pts)}
try:
    onsets={}; stamps={}; indices={}
    for i,spec in enumerate(planned["commands"]):
        if spec["kind"]!="control":continue
        approximate=records[i]["submitted_to_ios"]["epoch_s"]-execution["video"]["started_at_epoch_s"]
        frames,times,numbers=read_native_frames(run/"original.mov",pts,start=approximate-.7,end=approximate+.7)
        detection=detect_tap_onset(frames,times,point_xy=(.5,.5),command_s=approximate)
        if detection is None:raise ValueError("missing control "+spec["role"])
        role=spec["role"];onsets[role]=detection.onset_s;stamps[role]=records[i]["submitted_to_ios"]["monotonic_s"];indices[role]=numbers[times.index(detection.onset_s)]
    fit=fit_two_anchor_timeline(stamps["start"],onsets["start"],stamps["end"],onsets["end"],min_anchor_span_s=55.)
    error=onsets["middle"]-fit.video_time_s(stamps["middle"])
    results.update(fit=asdict(fit),control_onsets_s=onsets,control_source_frames=indices,heldout_middle_error_s=error,timing_checks_pass=abs(error)<=2*np.median(np.diff(pts)) and onsets["start"]<onsets["middle"]<onsets["end"])
except Exception as e:results.update(timing_checks_pass=False,timing_error=str(e))
(run/"timing-diagnostic.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps(results),flush=True)
number=[0];flag={}
def inspect(image):
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    menu=is_menu_frame(rgb,allow_idle_navigation=True);editor=is_editor_frame(rgb)
    if menu or editor:
        flag.update(source_frame=number[0],pts_s=float(pts[number[0]]),menu=menu,editor=editor)
        cv2.imwrite(str(run/"first-flag.png"),image)
        raise ValueError("first gameplay flag saved")
    number[0]+=1
try:_decode_source_frames(run/"original.mov",pts,keep=False,inspect=inspect)
except ValueError as e:
    if not flag:raise
(run/"flag-diagnostic.json").write_text(json.dumps(flag,indent=2)+"\n")
print(json.dumps(flag),flush=True)
