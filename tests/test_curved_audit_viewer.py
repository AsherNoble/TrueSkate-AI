"""Synthetic export proves condition/source labels stay out of the public payload."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from trueskate_ai.research.curved_audit import manifest


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
    for n,segment in enumerate(frozen['recordings'],1):
        commands=segment['commands']
        run=recordings/f'recording_{n}';run.mkdir()
        (run/'execution.json').write_text(json.dumps(dict(error=None,manifest_sha256=frozen['sha256'],events=[dict(spec=c) for c in commands])))
        (run/'calibration.json').write_text(json.dumps(dict(accepted=True,fit=dict(intercept_s=0.,rate=1.))))
        (run/'wda-timing.json').write_text(json.dumps(dict(records=[dict(submitted_to_ios=dict(monotonic_s=c['slot_s'])) for c in commands])))
    data=module.build(mp,recordings,out)
    assert len(data['clips'])==100
    payload=(out/'data.js').read_text()
    for forbidden in ('duration_ms','family','waypoint_count','timing_profile','device','park','path_id','source_frame','onset_s'):assert forbidden not in payload
    assert not (out/'private-key.json').exists()
    assert (recordings/'audit-private-source-map.json').exists()
    for clip in data['clips']:
        assert clip['frames'][0]['time_s']==0.
        assert all((out/f['path']).exists() for f in clip['frames'])



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
global.Image=Element;const storage=new Map();global.localStorage={getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)};
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
})().catch(e=>{console.error(e);process.exitCode=1});
"""
    result=subprocess.run(['node','-'],input=harness+preserved+script+checks,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
