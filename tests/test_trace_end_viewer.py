import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('trace_end_viewer', Path(__file__).resolve().parents[1] / 'scripts/inspect/build_trace_end_viewer.py')
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def test_end_labels_preserve_starts_and_reject_bad_pairing(tmp_path):
    times = [i / 30 for i in range(50)]
    (tmp_path / 'frame_index.json').write_text(json.dumps({'frames': [{'best_effort_timestamp_time': t} for t in times]}))
    (tmp_path / 'provenance.json').write_text(json.dumps({'recording': 'synthetic/video.mov'}))
    starts = {'recording': 'synthetic/video.mov', 'frame_count': 50, 'frame_numbering': 'zero-based',
              'labels': [{'frame': i, 'time_s': times[i], 'kind': 'gesture', 'note': ''} for i in (10, 30)]}
    source = tmp_path / 'starts.json'
    source.write_text(json.dumps(starts))
    output = viewer.build(tmp_path, source)
    assert json.loads(source.read_text()) == starts
    with pytest.raises(ValueError, match='already exists'):
        viewer.build(tmp_path, source)
    if not shutil.which('node'):
        pytest.skip('Node unavailable for control logic check')
    js = output.read_text().split('<script>')[1].split('</script>')[0]
    harness = '''const assert = require('node:assert/strict');
const elements = new Map();
function element(){return {value:'0',textContent:'',append(){},replaceChildren(){},className:''}}
global.document={getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id)},createElement:element,createTextNode:x=>x,addEventListener(){}};
global.localStorage={getItem(){return null},setItem(){}};
'''
    assertions = '''
const original=JSON.stringify(STARTS.labels);
show(20);mark();assert.equal(payload().end_labels[0].start_frame,10);assert.equal(payload().end_labels[0].frame,20);
show(9);mark();assert.equal(payload().end_labels[0].frame,20);
show(30);mark();assert.equal(payload().end_labels[0].frame,20);
$('gesture').value='1';change();show(49);mark();assert.equal(payload().end_labels.length,1);
mark(true);assert.equal(payload().end_labels[1].frame,null);assert.equal(payload().end_labels[1].status,'uncertain');
assert.equal(JSON.stringify(payload().labels),original);
'''
    script = tmp_path / 'controls.js'
    script.write_text(harness + js + assertions)
    subprocess.run(['node', str(script)], check=True)
