import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('anchors',Path(__file__).resolve().parents[1]/'scripts/inspect/compare_timing_anchors.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_middle_label_does_not_influence_fit_and_failure_is_retained():
    x=[i*5.6 for i in range(11)]
    y=[1+v*.998 for v in x]
    records=[dict(sequence=i,outcome='success',submitted_to_ios=dict(monotonic_s=v+100)) for i,v in enumerate(x)]
    marks=[dict(frame=i,time_s=v,kind='calibration' if i in (0,5,10) else 'gesture') for i,v in enumerate(y)]
    frames=[dict(best_effort_timestamp_time=v) for v in y]
    args=[dict(records=records,dropped_records=0),dict(gestures=[{}]*11),dict(labels=marks,frame_count=11),dict(frames=frames)]
    result=mod.compare(*args)
    assert result['initial_test_pass'] and result['rate']==pytest.approx(.998)
    marks[5]['time_s']+=.1;frames[5]['best_effort_timestamp_time']+=.1
    bad=mod.compare(*args)
    assert bad['rate']==result['rate'] and not bad['initial_test_pass']
    assert bad['stats']['two_anchor']['within_one_frame']==8
