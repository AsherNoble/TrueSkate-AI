import cv2
import numpy as np
from trueskate_ai.research.curve_measurement import extract_observation,symmetric_distance


def frames(orange_background=False,branch=False,gap=False,collapse=False,fade=False):
    base=np.zeros((180,300,3),np.uint8)
    if orange_background:cv2.rectangle(base,(0,0),(40,40),(0,140,255),-1)
    result=[base.copy() for _ in range(3)]
    for i in range(1,8):
        f=base.copy();end=220 if collapse else 70+i*20
        cv2.line(f,(70,100),(end,100),(0,140,255),7)
        if branch:cv2.line(f,(120,100),(120,150),(0,140,255),7)
        if gap:cv2.rectangle(f,(125,90),(135,110),(0,0,0),-1)
        if fade and i>4:f=base.copy()
        result.append(f)
    return result


def observe(**kwargs):
    images=frames(**kwargs)
    return extract_observation(images,[i/30 for i in range(len(images))],baseline_count=3)


def test_blind_extractor_static_orange_background_and_real_geometry():
    observation=observe(orange_background=True)
    assert observation['shape_evaluable'] and len(observation['contacts'])>=3
    assert all(p[0]>.2 for p in observation['centreline'])
    assert symmetric_distance(observation['centreline'],[[70/300,100/180],[210/300,100/180]])>.1 # complete line, not snapped endpoints


def test_branches_and_gaps_are_indeterminate():
    assert not observe(branch=True)['shape_evaluable']
    assert not observe(gap=True)['shape_evaluable']


def test_whole_trail_in_one_frame_is_collapse_candidate():
    result=observe(collapse=True)
    assert result['collapse_suspected'] and not result['duration_evaluable']


def test_fading_does_not_become_finger_liftoff():
    result=observe(fade=True)
    assert 'liftoff_s' not in result
    assert result['growth'][-1]['pts_s']<9/30


def test_ffmpeg_native_decode_matches_pts_and_selected_pixels(tmp_path):
    import shutil,subprocess,pytest
    from trueskate_ai.research.curve_measurement import frame_pts,read_native_frames,_decode_source_frames
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):pytest.skip('FFmpeg tools not installed')
    video=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=160x120:rate=30',
                    '-t','0.3','-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
    pts=frame_pts(video);images,times,indices=read_native_frames(video,pts)
    assert len(images)==len(pts)==9 and times==pts.tolist()
    selected,stamps,ids=read_native_frames(video,pts,start=pts[2],end=pts[5])
    assert ids==[2,3,4,5] and stamps==pts[2:6].tolist()
    for a,b in zip(selected,images[2:6]):np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='exceeds'):_decode_source_frames(video,pts[:-1],keep=False)
    with pytest.raises(ValueError,match='differs'):_decode_source_frames(video,np.append(pts,.3),keep=False)
