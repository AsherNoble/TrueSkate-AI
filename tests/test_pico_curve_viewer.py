"""Review marks must refer to loaded native pixels and preserve unknowns."""
import re
import subprocess
from pathlib import Path


def test_native_review_player_preserves_frame_identity_and_blocks_failed_media():
    template = (Path(__file__).resolve().parents[1]/'scripts/inspect/templates/pico_curve_pilot.html').read_text()
    script = re.findall(r'<script>(.*?)</script>', template, re.S)[0]
    harness = r'''
const assert=require('node:assert/strict');
class Element {
 constructor(){this.style={};this.parentElement={};this.value='';this.checked=false;this.disabled=false;}
 set src(v){this._src=v;if(this.onload)queueMicrotask(()=>this.onload())}
 get src(){return this._src}
 setAttribute(k,v){this[k]=v}getAttribute(k){return this[k]}removeAttribute(k){delete this[k]}
 click(){if(!this.disabled&&this.onclick)this.onclick()}
 createSVGPoint(){return {x:0,y:0,matrixTransform(){return {x:100,y:300}}}}
 getScreenCTM(){return {inverse(){return {}}}}
}
const elements=new Map();global.document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id)},createElement(){return new Element()},addEventListener(){}};
const storage=new Map();global.localStorage={getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)};
global.reviewFrameUrl=async f=>f.path;
const DATA={schema:'pico-usb-curves-review-v1',version:2,bundle_sha256:'fixture',clips:[{id:'s-150-r0',purpose:'curve',frames:[0,1,2].map(i=>({source_frame:i+100,pts_s:10+i*(i===2?.02:.016),path:i+'.jpg',target:i===1?[.5,.48]:null}))},{id:'control-start',purpose:'control',frames:[0,1,2].map(i=>({source_frame:i+200,pts_s:20+i/60,path:'c'+i+'.jpg',target:null}))}]};
'''
    checks = r'''
(async()=>{
 await new Promise(setImmediate);assert(loaded);assert.equal(row().frames.length,0);assert.equal(row().interrupted,null);
 $('uncertainty').value='2';$('contact').checked=true;$('screen').onclick({clientX:1,clientY:1});
 assert.equal(row().frames[0].source_frame,100);assert.equal(row().frames[0].contact_confirmed,true);
 step(1);saveFrame([200,400]);assert.equal(row().frames.length,1); // old pixels cannot mark the new frame
 await new Promise(setImmediate);$('overlay').checked=true;await show();
 assert.equal($('ring').r,undefined);assert.equal($('ring').getAttribute('cx'),207);assert.equal($('ring').getAttribute('cy'),430.08);
 assert.equal($('ring').getAttribute('visibility'),'visible');
 $('onset').click();assert.equal(row().first_contact_frame,101);assert.equal(row().boundary_confirmed,false);
 $('boundaryConfirmed').checked=true;$('boundaryConfirmed').onchange();assert.equal(row().boundary_confirmed,true);
 $('overlay').checked=false;await show();assert.equal($('ring').getAttribute('visibility'),'hidden');
 $('next').click();await new Promise(setImmediate);assert.equal(row().first_contact_frame,undefined);assert.equal(row().frames.length,0);
 $('onset').click();assert.equal(row().first_contact_frame,undefined); // no clear preceding displayed frame
 step(1);await new Promise(setImmediate);$('onset').click();assert.equal(row().first_contact_frame,201);
 global.reviewFrameUrl=async()=>{throw Error('changed bytes')};step(1);await new Promise(setImmediate);
 assert(integrityFailed);assert(!loaded);assert($('export').disabled);saveFrame([0,0]);assert.equal(row().frames.length,0);
 global.reviewFrameUrl=async f=>f.path;$('previous').click();await new Promise(setImmediate);
 assert($('export').disabled);assert.equal(row().frames.length,1);assert.equal($('ring').getAttribute('visibility'),'hidden');
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result = subprocess.run(['node', '-'], input=harness+script+checks, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'r="10" fill="none"' in template
