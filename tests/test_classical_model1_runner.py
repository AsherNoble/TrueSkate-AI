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
