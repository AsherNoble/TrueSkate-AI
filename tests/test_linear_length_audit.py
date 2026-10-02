"""Synthetic export proves condition/source labels stay out of the public payload."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from trueskate_ai.research.linear_length_probe import manifest,blind_key


def test_blinded_export_has_no_condition_or_source_leaks(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/inspect/build_linear_length_audit.py'
    spec=importlib.util.spec_from_file_location('length_audit',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frozen=manifest();mp=tmp_path/'manifest.json';mp.write_text(json.dumps(frozen))
    recordings=tmp_path/'recordings';recordings.mkdir();out=tmp_path/'public'
    pts=np.arange(0,59,1/30)
    monkeypatch.setattr(module,'frame_pts',lambda _:pts)
    def decode(video,times,keep,inspect):
        for _ in times:inspect(np.zeros((2,2,3),dtype=np.uint8))
    monkeypatch.setattr(module,'_decode_source_frames',decode)
    def write(path,image,options):Path(path).write_bytes(b'fixture');return True
    monkeypatch.setattr(module.cv2,'imwrite',write)
    for n,commands in enumerate(frozen['recordings'],1):
        run=recordings/f'recording_{n}';run.mkdir()
        (run/'execution.json').write_text(json.dumps(dict(error=None,events=[dict(spec=c) for c in commands])))
        (run/'timing-diagnostic.json').write_text(json.dumps(dict(timing_checks_pass=True,fit=dict(intercept_s=0.,rate=1.))))
        (run/'wda-timing.json').write_text(json.dumps(dict(records=[dict(submitted_to_ios=dict(monotonic_s=c['slot_s'])) for c in commands])))
    data=module.build(mp,recordings,out)
    assert len(data['clips'])==135
    assert [c['id'] for c in data['clips']]==[x['token'] for x in blind_key(frozen)['items']]
    payload=(out/'data.js').read_text()
    for forbidden in ('duration_ms','length_fraction','repetition','recording','command_id','source_frame','onset_s'):assert forbidden not in payload
    assert not (out/'private-key.json').exists()
    assert (recordings/'audit-private-source-map.json').exists()
    for clip in data['clips']:
        assert clip['frames'][0]['time_s']==0.
        assert all((out/f['path']).exists() for f in clip['frames'])


def test_review_defaults_and_saved_false_survive_navigation():
    import subprocess,shutil,re
    import pytest
    node=shutil.which('node')
    if node is None:pytest.skip('Node required for offline player check')
    template=Path(__file__).resolve().parents[1]/'scripts/inspect/templates/linear_length_audit.html'
    script=re.findall(r'<script>(.*?)</script>',template.read_text(),re.S)[0]
    harness=r'''
const assert=require('node:assert/strict');
class Element {
 constructor(){this.style={};this.value='';this.checked=false;this.children=[];this.disabled=false;}
 set src(v){this._src=v;if(this.onload)queueMicrotask(()=>this.onload());}
 get src(){return this._src;}
 append(v){this.children.push(v)} replaceChildren(){this.children=[]} click(){if(this.onclick)this.onclick()}
}
const elements=new Map();
global.document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id)},createElement(){return new Element()},addEventListener(){}};
global.Image=Element;const storage=new Map();global.localStorage={getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)};
const DATA={bundle_sha256:'fixture',clips:[0,1,2].map(i=>({id:'opaque'+i,frames:[0,1,2].map(j=>({path:'opaque'+i+'/'+j+'.jpg',time_s:j/30}))}))};
'''
    checks=r'''
(async()=>{
 await new Promise(setImmediate);
 assert.equal($('board').checked,true);
 assert.equal($('mark').disabled,true);
 $('feedback').value='flicker';$('feedback').onchange();$('board').checked=false;$('note').value='comment';$('mark').click();
 assert.equal(marks.opaque0.board_moved,false);assert.equal(marks.opaque0.trace_visible,'flicker');assert.equal(marks.opaque0.comments,'comment');
 assert.equal($('board').checked,true);
 // A new clip cannot save against an image which has not loaded yet.
 $('feedback').value='hold';save();assert.equal(marks.opaque1,undefined);
 await new Promise(setImmediate);$('mark').click();assert.equal(marks.opaque1.board_moved,true);
 choose(0);assert.equal($('board').checked,false);assert.equal($('feedback').value,'flicker');
 await new Promise(setImmediate);$('forward').click();await new Promise(setImmediate);assert.equal($('stamp').textContent,'Frame 2 / 3');
 assert.deepEqual(Object.keys(marks.opaque0).sort(),['board_moved','comments','trace_visible','updated_at']);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run([node,'-'],input=harness+script+checks,text=True,capture_output=True,check=True)
