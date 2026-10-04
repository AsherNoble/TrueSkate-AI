"""Synthetic export proves condition/source labels stay out of the public payload."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from trueskate_ai.research.linear_length_probe import manifest,blind_key
from trueskate_ai.research.curve_protocol import digest
from trueskate_ai.collection.wda_action_timing import BOUNDARIES


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
        metadata=dict(manifest_sha256=frozen['sha256'],device=frozen['device'],park=frozen['park'],experiment=frozen['experiment'],wda_revision='fixture')
        (run/'original.mov').write_bytes(b'synthetic original')
        (run/'planned.json').write_text(json.dumps(dict(**metadata,commands=commands)))
        events=[dict(spec=c,payload=c['payload'],payload_sha256=digest(c['payload']),success=True,
                     call_start_monotonic_s=c['slot_s'],call_end_monotonic_s=c['slot_s']+.1) for c in commands]
        (run/'execution.json').write_text(json.dumps(dict(**metadata,execution_schema='research-execution-v2',error=None,events=events)))
        (run/'timing-diagnostic.json').write_text(json.dumps(dict(timing_checks_pass=True,fit=dict(intercept_s=0.,rate=1.))))
        records=[]
        for i,c in enumerate(commands):
            row=dict(sequence=i,outcome='success',session_id='fixture',missing_ios_callback=False,ios_callback_result=True)
            row.update({name:dict(monotonic_s=c['slot_s']+k*.001,epoch_s=1000+c['slot_s']+k*.001) for k,name in enumerate(BOUNDARIES)})
            records.append(row)
        (run/'wda-timing.json').write_text(json.dumps(dict(schema_version=1,build_revision='fixture',dropped_records=0,records=records)))
    data=module.build(mp,recordings,out)
    assert data['schema']=='blind-linear-length-v2'
    assert len(data['clips'])==135
    assert [c['id'] for c in data['clips']]==[x['token'] for x in blind_key(frozen)['items']]
    payload=(out/'data.js').read_text()
    for forbidden in ('duration_ms','length_fraction','repetition','recording','command_id','source_frame','onset_s'):assert forbidden not in payload
    assert not (out/'private-key.json').exists()
    assert (recordings/'audit-private-source-map-v2.json').exists()
    for clip in data['clips']:
        assert clip['frames'][0]['time_s']==0.
        assert all((out/f['path']).exists() for f in clip['frames'])
    # Import the actual new bundle with full receipts, then mutate a displayed JPEG.
    import pytest
    report_script=script.with_name('report_linear_length_audit.py')
    report_spec=importlib.util.spec_from_file_location('length_v2_report',report_script)
    reporter=importlib.util.module_from_spec(report_spec);report_spec.loader.exec_module(reporter)
    evidence=tmp_path/'evidence';evidence.mkdir()
    (evidence/'manifest.json').write_text(json.dumps(frozen))
    (evidence/'private-key.json').write_text(json.dumps(blind_key(frozen)))
    (evidence/'summary.json').write_text(json.dumps(dict(blinded_bundle_sha256=data['bundle_sha256'])))
    marks={c['id']:dict(trace_visible='trace',board_moved=True,comments='',updated_at='fixture') for c in data['clips']}
    export=tmp_path/'export.json';export.write_text(json.dumps(dict(schema=data['schema'],bundle_sha256=data['bundle_sha256'],assessments=marks)))
    kwargs=dict(bundle_map=recordings/'audit-private-source-map-v2.json',media_root=out)
    result=reporter.report(export,evidence,tmp_path/'result',**kwargs)
    assert result['overall']['n']==135 and result['provenance'].startswith('v2')
    (out/data['clips'][0]['frames'][0]['path']).write_bytes(b'mutated JPEG')
    with pytest.raises(ValueError,match='bytes changed'):
        reporter.report(export,evidence,tmp_path/'invalid-result',**kwargs)
    assert not (tmp_path/'invalid-result').exists()


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
 setAttribute(k,v){this[k]=v} getAttribute(k){return this[k]}
 append(v){this.children.push(v)} replaceChildren(){this.children=[]} click(){if(this.onclick)this.onclick()}
}
const elements=new Map();
global.document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id)},createElement(){return new Element()},addEventListener(){}};
global.reviewFrameUrl=async f=>f.path;global.Image=Element;const storage=new Map();global.localStorage={getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)};
const DATA={bundle_sha256:'fixture',clips:[0,1,2].map(i=>({id:'opaque'+i,frames:[0,1,2].map(j=>({path:'opaque'+i+'/'+j+'.jpg',time_s:j/30}))}))};
'''
    checks=r'''
(async()=>{
 await new Promise(setImmediate);
 assert.equal($('board').checked,true);
 assert.equal($('mark').disabled,false);
 assert.equal(feedback,'trace');
 $('feedback-flicker').click();$('board').checked=false;$('note').value='comment';$('mark').click();
 assert.equal(marks.opaque0.board_moved,false);assert.equal(marks.opaque0.trace_visible,'flicker');assert.equal(marks.opaque0.comments,'comment');
 assert.equal($('board').checked,true);
 // A new clip cannot save against an image which has not loaded yet.
 $('feedback-hold').click();save();assert.equal(marks.opaque1,undefined);
 await new Promise(setImmediate);$('mark').click();assert.equal(marks.opaque1.board_moved,true);
 choose(0);assert.equal($('board').checked,false);assert.equal(feedback,'flicker');
 assert.equal($('feedback-flicker').getAttribute('aria-checked'),'true');
 assert.equal($('feedback-hold').getAttribute('aria-checked'),'false');
 assert.equal($('feedback-trace').getAttribute('aria-checked'),'false');
 await new Promise(setImmediate);$('forward').click();await new Promise(setImmediate);assert.equal($('stamp').textContent,'Frame 2 / 3');
 const existing=JSON.stringify(marks);choose(2);choose(0);assert.equal(JSON.stringify(marks),existing);
 assert.deepEqual(Object.keys(marks.opaque0).sort(),['board_moved','comments','trace_visible','updated_at']);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run([node,'-'],input=harness+script+checks,text=True,capture_output=True,check=True)

    preserved=r"""
const before=JSON.stringify({opaque0:{trace_visible:'hold',board_moved:false,comments:'saved comment',updated_at:'old'}});
storage.set('blind-linear-length:fixture',before);
"""
    restore_checks=r"""
(async()=>{
 await new Promise(setImmediate);
 assert.equal(cursor,1);assert.equal(feedback,'trace');assert.equal($('board').checked,true);
 choose(0);assert.equal(feedback,'hold');assert.equal($('board').checked,false);assert.equal($('note').value,'saved comment');
 assert.equal(storage.get(KEY),before);assert.equal(JSON.stringify(marks),before);
 assert.equal($('feedback-hold').getAttribute('aria-checked'),'true');
 choose(2);assert.equal(feedback,'trace');assert.equal($('board').checked,true);
 assert.equal(storage.get(KEY),before);
})().catch(e=>{console.error(e);process.exitCode=1});
"""
    subprocess.run([node,'-'],input=harness+preserved+script+restore_checks,text=True,capture_output=True,check=True)
