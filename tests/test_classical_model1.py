import importlib.util
from pathlib import Path
import cv2
import numpy as np
import pytest
from trueskate_ai.model1.classical.predictor import Config, extract, predict, metrics


def clip(reverse=False, fade=0):
    frames=np.zeros((32,288,128,3),np.uint8)
    points=np.linspace([30,130],[90,160],12)
    if reverse: points=points[::-1]
    for i,p in enumerate(points,8): cv2.circle(frames[i],tuple(np.rint(p).astype(int)),3,(0,140,255),-1)
    for i in range(20,min(32,20+fade)):
        cv2.circle(frames[i],tuple(np.rint(points[-1]).astype(int)),3,(0,140,255),-1)
    return frames,np.arange(32)*.075,points


@pytest.mark.parametrize('reverse',[False,True])
def test_linear_direction_and_lingering(reverse):
    frames,times,points=clip(reverse,fade=6)
    p=predict(frames,times)
    assert p is not None
    assert np.linalg.norm(np.asarray(p[:2])-points[0]/[127,287]) < .015
    assert np.linalg.norm(np.asarray(p[2:4])-points[-1]/[127,287]) < .015
    assert abs(p[-1]-.825)<.08


def test_blank_is_missing_and_counts_as_failure():
    p=predict(np.zeros((32,20,20,3),np.uint8),np.arange(32)*.1)
    assert p is None
    m=metrics([None,[0,0,.5,.5,.5]],[[0,0,.5,.5,.5]]*2)
    assert m['samples']==2 and m['missing']==1 and m['recovery']==.5


def test_metric_boundaries():
    t=[0,0,0,0,0]
    assert metrics([[.03,0,0,.03,.10]],[t])['recovered']==1
    assert metrics([[.030001,0,0,.03,.10]],[t])['recovered']==0


def test_invalid_times_and_elapsed_shift():
    f,t,_=clip()
    with pytest.raises(ValueError): extract(f,np.zeros(32))
    assert np.allclose(predict(f,t),predict(f,t+100))


def test_predictor_has_no_metadata_input():
    import inspect
    assert list(inspect.signature(predict).parameters)==['frames','elapsed_times','config']


@pytest.mark.parametrize('reverse',[False,True])
def test_consensus_recovery(reverse):
    frames,times,points=clip(reverse,fade=6)
    p=predict(frames,times,Config(tracking='consensus'))
    assert p is not None
    assert np.linalg.norm(np.asarray(p[:2])-points[0]/[127,287])<.02
    assert np.linalg.norm(np.asarray(p[2:4])-points[-1]/[127,287])<.02
    assert abs(p[-1]-.825)<.08
