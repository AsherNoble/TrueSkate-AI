import json
from collections import Counter
import pytest
from trueskate_ai.research.linear_length_probe import manifest, blind_key
from trueskate_ai.research.linear_speed_probe import verify_manifest
from trueskate_ai.data.control_hitboxes import segment_is_safe


def test_factorial_balance_and_payloads():
    frozen=manifest();verify_manifest(json.loads(json.dumps(frozen)))
    cells=[]
    for commands in frozen['recordings']:
        assert len(commands)==21
        assert [c['role'] for c in commands if c['kind']=='control']==['start','middle','end']
        for i,c in enumerate(commands):
            if c['kind']!='diagnostic':continue
            assert commands[i-1]['kind']=='reset' and c['slot_s']-commands[i-1]['slot_s']==3
            assert segment_is_safe(*c['path'])
            cells.append((c['length_fraction'],c['duration_ms']))
            a=c['payload']['actions'][0]['actions']
            assert [x['type'] for x in a]==['pointerMove','pointerDown','pointerMove','pointerUp']
            assert a[2]['duration']==c['duration_ms']
            assert (a[2]['x'],a[2]['y'])==tuple(int(v*s) for v,s in zip(c['path'][1],(414,896)))
    assert len(cells)==135 and len(Counter(cells))==45 and set(Counter(cells).values())=={3}


def test_blind_order_unique_and_independent():
    f=manifest();key=blind_key(f)
    assert len({x['token'] for x in key['items']})==135
    original=[c['command_id'] for commands in f['recordings'] for c in commands if c['kind']=='diagnostic']
    assert [x['command_id'] for x in key['items']]!=original
    f=json.loads(json.dumps(f));f['recordings'][0][2]['payload']['actions'][0]['actions'][2]['duration']=999
    with pytest.raises(ValueError):verify_manifest(f)
