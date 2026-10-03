import importlib.util
import json
from pathlib import Path
import pytest
from trueskate_ai.research.linear_length_probe import manifest,blind_key


def load_module():
    path=Path(__file__).resolve().parents[1]/'scripts/inspect/report_linear_length_audit.py'
    spec=importlib.util.spec_from_file_location('length_report',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture(tmp_path):
    evidence=tmp_path/'evidence';evidence.mkdir();f=manifest();key=blind_key(f)
    (evidence/'manifest.json').write_text(json.dumps(f));(evidence/'private-key.json').write_text(json.dumps(key));(evidence/'summary.json').write_text(json.dumps(dict(blinded_bundle_sha256='fixture')))
    marks={x['token']:dict(trace_visible='flicker' if x['duration_ms']==10 else 'trace',board_moved=x['duration_ms']!=10,comments='',updated_at='fixture') for x in key['items']}
    export=dict(schema='blind-linear-length-v1',bundle_sha256='fixture',assessments=marks);p=tmp_path/'export.json';p.write_text(json.dumps(export));return evidence,p,export


def test_counts_and_exact_preservation(tmp_path):
    evidence,p,export=fixture(tmp_path);out=tmp_path/'out';raw=p.read_bytes();r=load_module().report(p,evidence,out)
    assert r['overall']==dict(n=135,trace=120,hold=0,flicker=15,board_moved=120)
    assert (out/'operator-assessments.json').read_bytes()==raw
    assert all(c['n']==3 for c in r['by_cell'])
    assert r['by_duration'][-1]['board_moved']==0


@pytest.mark.parametrize('mode',['bundle','missing','board','extra'])
def test_bad_export_does_not_write_output(tmp_path,mode):
    evidence,p,export=fixture(tmp_path);token=next(iter(export['assessments']))
    if mode=='bundle':export['bundle_sha256']='wrong'
    elif mode=='missing':del export['assessments'][token]
    elif mode=='board':export['assessments'][token]['board_moved']='true'
    else:export['assessments'][token]['duration_ms']=999
    p.write_text(json.dumps(export));out=tmp_path/'out'
    with pytest.raises(ValueError):load_module().report(p,evidence,out)
    assert not out.exists()


def test_duplicate_keys_rejected():
    with pytest.raises(ValueError):json.loads('{"a":1,"a":2}',object_pairs_hook=load_module().unique_object)
