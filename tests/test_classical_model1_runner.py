import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

P=Path(__file__).resolve().parents[1]/'scripts/model1/classical_model1.py'
spec=importlib.util.spec_from_file_location('classical_runner',P)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def test_manifest_digest_changes_for_membership_or_content():
    a=[{'path':'a','content_sha256':'abc'}]
    assert r.digest(a)!=r.digest([{'path':'a','content_sha256':'def'}])
    assert r.digest(a)!=r.digest(a+[{'path':'b','content_sha256':'abc'}])


def test_atomic_replacement(tmp_path):
    p=tmp_path/'state.json';r.atomic(p,{'step':1});r.atomic(p,{'step':2})
    assert json.loads(p.read_text())=={'step':2}
    assert not p.with_suffix('.json.tmp').exists()


def test_test_requires_freeze(tmp_path):
    from argparse import Namespace
    m=tmp_path/'manifest.json';r.atomic(m,{'entries':[],'fingerprint':r.digest([])})
    args=Namespace(manifest=m,config=None,partition='test',out=tmp_path/'runs',limit=0)
    with pytest.raises(FileNotFoundError):r.evaluate(args)


def test_resume_and_changed_content_refusal(tmp_path):
    from argparse import Namespace
    import cv2
    sample=tmp_path/'sample';sample.mkdir()
    frames=np.zeros((32,40,40,3),np.uint8)
    for i in range(8,20):cv2.circle(frames[i],(8+i,20),2,(0,140,255),-1)
    for i,f in enumerate(frames):cv2.imwrite(str(sample/f'frame_{i:06d}.png'),f)
    r.atomic(sample/'meta.json',{'frame_times':(np.arange(32)*.075-.5).tolist(),
        'waypoints':[[.4,.5],[.7,.5]],'duration':.8})
    entry={'path':str(sample),'partition':'train','command':'synthetic','content_sha256':r.content_hash(sample)}
    manifest=tmp_path/'manifest.json';r.atomic(manifest,{'entries':[entry],'fingerprint':r.digest([entry])})
    args=Namespace(manifest=manifest,config=None,partition='train',out=tmp_path/'runs',limit=0,
                   workers=1,seconds=60,hypothesis='synthetic')
    r.evaluate(args);first=json.loads((args.out/'latest.json').read_text())
    r.evaluate(args);second=json.loads((args.out/'latest.json').read_text())
    assert first['run_id']==second['run_id'] and first['metrics']==second['metrics']
    (sample/'meta.json').write_text('{}')
    with pytest.raises(ValueError,match='content changed'):r.evaluate(args)
