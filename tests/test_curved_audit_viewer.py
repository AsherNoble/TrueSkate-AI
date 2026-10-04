"""Synthetic export proves condition/source labels stay out of the public payload."""
import importlib.util
import json
import copy
from pathlib import Path
import numpy as np
from trueskate_ai.research.curved_audit import manifest,digest
from trueskate_ai.collection.wda_action_timing import BOUNDARIES
import pytest


def write_run(frozen, root, n):
    segment=frozen['recordings'][n-1];commands=segment['commands']
    run=root/f'recording_{n}';run.mkdir()
    metadata=dict(identity=frozen['identity'],segment=n,manifest_sha256=frozen['sha256'],
                  device=segment['device'],park=segment['park'],wda_revision='fixture')
    (run/'original.mov').write_bytes(b'synthetic original')
    (run/'planned.json').write_text(json.dumps(dict(**metadata,commands=commands)))
    events=[dict(spec=c,payload=c['payload'],payload_sha256=digest(c['payload']),success=True,
                 call_start_monotonic_s=c['slot_s'],call_end_monotonic_s=c['slot_s']+.1) for c in commands]
    (run/'execution.json').write_text(json.dumps(dict(**metadata,execution_schema='research-execution-v2',error=None,events=events)))
    (run/'calibration.json').write_text(json.dumps(dict(accepted=True,fit=dict(intercept_s=0.,rate=1.))))
    records=[]
    for i,c in enumerate(commands):
        row=dict(sequence=i,outcome='success',session_id='fixture',missing_ios_callback=False,ios_callback_result=True)
        row.update({name:dict(monotonic_s=c['slot_s']+k*.001,epoch_s=1000+c['slot_s']+k*.001) for k,name in enumerate(BOUNDARIES)})
        records.append(row)
    (run/'wda-timing.json').write_text(json.dumps(dict(schema_version=1,build_revision='fixture',dropped_records=0,records=records)))


def test_blinded_export_has_no_condition_or_source_leaks(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/inspect/build_curved_audit.py'
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
    for n in range(1,11):write_run(frozen,recordings,n)
    data=module.build(mp,recordings,out)
    assert len(data['clips'])==100
    payload=(out/'data.js').read_text()
    for forbidden in ('duration_ms','family','waypoint_count','timing_profile','device','park','path_id','source_frame','onset_s'):assert forbidden not in payload
    assert not (out/'private-key.json').exists()
    assert (recordings/'audit-private-source-map-v2.json').exists()
    for clip in data['clips']:
        assert clip['frames'][0]['time_s']==0.
        assert all((out/f['path']).exists() for f in clip['frames'])

    report_script=script.with_name('report_curved_audit.py')
    rs=importlib.util.spec_from_file_location('curved_report',report_script)
    reporter=importlib.util.module_from_spec(rs);rs.loader.exec_module(reporter)
    mapping=json.loads((recordings/'audit-private-source-map-v2.json').read_text())
    export=dict(schema=data['schema'],bundle_sha256=data['bundle_sha256'],
                assessments={data['clips'][0]['id']:dict(rating='minor',comments='fixture')})
    result=reporter.report(mapping,export,media_root=out)
    assert result['reviewed']==1 and result['missing']==99 and result['provenance'].startswith('v2')
    for fault in ('same_id_different_payload','failed_callback','wrong_condition','incomplete_proofs'):
        bad=copy.deepcopy(mapping);marks=copy.deepcopy(export)
        identity=bad['identity']
        if fault=='same_id_different_payload':
            event=identity['provenance']['recordings'][0]['execution']['events'][1]
            event['payload']={'actions':[]};event['payload_sha256']=digest(event['payload'])
        elif fault=='failed_callback':
            identity['provenance']['recordings'][0]['wda_timing']['records'][1]['ios_callback_result']=False
        elif fault=='wrong_condition':identity['mapping']['items'][0]['family']='unexecuted condition'
        else:identity['provenance']['recordings'].pop()
        bad['bundle_sha256']=marks['bundle_sha256']=digest(identity)
        with pytest.raises(ValueError):reporter.report(bad,marks,media_root=out)
    (out/data['clips'][0]['frames'][0]['path']).write_bytes(b'changed JPEG')
    with pytest.raises(ValueError,match='bytes changed'):reporter.report(mapping,export,media_root=out)




def test_replacement_segment_must_match_conditions(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/inspect/build_curved_audit.py'
    spec=importlib.util.spec_from_file_location('length_audit_r',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pts=np.arange(0,59,1/30)
    monkeypatch.setattr(module,'frame_pts',lambda _:pts)
    def decode(video,times,keep,inspect):
        for _ in times:inspect(np.zeros((2,2,3),dtype=np.uint8))
    monkeypatch.setattr(module,'_decode_source_frames',decode)
    monkeypatch.setattr(module.cv2,'imwrite',lambda path,image,options:Path(path).write_bytes(b'x') or True)
    def runs(frozen,root,segments):
        root.mkdir()
        for n in segments:write_run(frozen,root,n)
    v2,v3=manifest('v2'),manifest('v3')
    m2=tmp_path/'v2.json';m2.write_text(json.dumps(v2));m3=tmp_path/'v3.json';m3.write_text(json.dumps(v3))
    runs(v2,tmp_path/'run3',range(1,10));runs(v3,tmp_path/'run4',[10])
    data=module.build(m2,tmp_path/'run3',tmp_path/'public',(m3,tmp_path/'run4',[10]))
    assert len(data['clips'])==100
    items=json.loads((tmp_path/'run3'/'audit-private-source-map-v2.json').read_text())['identity']['mapping']['items']
    assert {i['source_manifest'] for i in items if i['segment']==10}=={v3['identity']}
    tampered=json.loads(json.dumps(v3));tampered['recordings'][9]['park']='Other'
    from trueskate_ai.research.curved_audit import digest
    tampered['sha256']=digest({k:v for k,v in tampered.items() if k!='sha256'})
    mt=tmp_path/'bad.json';mt.write_text(json.dumps(tampered))
    import pytest
    with pytest.raises(ValueError):module.build(m2,tmp_path/'run3',tmp_path/'public2',(mt,tmp_path/'run4',[10]))
    # An ID-only comparison would incorrectly accept this changed payload.
    segment=copy.deepcopy(v3['recordings'][9]);changed=copy.deepcopy(segment)
    sample=next(c for c in changed['commands'] if c['kind']=='sample')
    sample['payload']={'actions':[]}
    assert not module._same_conditions(segment,changed)
    shifted=copy.deepcopy(segment)
    for c in shifted['commands']:c['slot_s']+=.1
    assert module._same_conditions(segment,shifted)


def test_player():
    import re,subprocess
    template=Path(__file__).resolve().parents[1]/'scripts/inspect/templates/curved_audit.html'
    text=template.read_text()
    assert 'r="10" fill="none"' in text and 'viewBox="0 0 414 896"' in text
    script=re.findall(r'<script>(.*?)</script>',text,re.S)[0]
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
const DATA={bundle_sha256:'fixture',clips:[0,1,2].map(i=>({id:'opaque'+i,frames:[0,1,2].map(j=>({path:'opaque'+i+'/'+j+'.jpg',time_s:j/30,target:j===1?[207,448]:null}))}))};
'''
    preserved=r"""
storage.set('blind-linear-length:fixture','old ratings');
const before=JSON.stringify({opaque0:{rating:'major',comments:'saved',updated_at:'old'}});
storage.set('blind-curved-execution:fixture',before);
"""
    checks=r"""
(async()=>{
 await new Promise(setImmediate);assert.equal(cursor,1);assert.equal(feedback,'good');
 assert.equal($('ring').getAttribute('visibility'),'hidden');
 $('overlay').checked=true;$('forward').click();await new Promise(setImmediate);
 assert.equal($('ring').getAttribute('visibility'),'visible');assert.equal($('ring').getAttribute('cx'),207);
 assert.equal($('stamp').textContent,'Frame 2 / 3');
 $('overlay').checked=false;$('overlay').onchange();assert.equal($('ring').getAttribute('visibility'),'hidden');
 $('feedback-minor').click();$('note').value='new';save();
 assert.equal(marks.opaque1.rating,'minor');assert.equal(marks.opaque1.comments,'new');
 assert.equal(feedback,'good');save();assert.equal(marks.opaque2,undefined);
 choose(0);assert.equal(feedback,'major');assert.equal($('note').value,'saved');
 for(const v of ['good','minor','major','unclear'])assert.equal($('feedback-'+v).getAttribute('aria-checked'),String(v==='major'));
 assert.equal(storage.get('blind-linear-length:fixture'),'old ratings');
 await new Promise(setImmediate);$('overlay').checked=true;$('forward').click();await new Promise(setImmediate);
 $('forward').click();await new Promise(setImmediate);assert.equal($('ring').getAttribute('visibility'),'hidden');
 // A changed frame blocks saved marks even after navigating to a valid clip.
 global.reviewFrameUrl=async()=>{throw Error('changed bytes')};show(1);save();
 await new Promise(setImmediate);assert.equal(loaded,false);assert.equal(integrityFailed,true);
 assert.equal($('mark').disabled,true);assert.equal($('ring').getAttribute('visibility'),'hidden');
 global.reviewFrameUrl=async f=>f.path;choose(2);await new Promise(setImmediate);save();
 assert.equal(marks.opaque2,undefined);
})().catch(e=>{console.error(e);process.exitCode=1});
"""
    result=subprocess.run(['node','-'],input=harness+preserved+script+checks,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
