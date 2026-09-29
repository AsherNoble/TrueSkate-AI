import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
const source=await readFile(new URL('../scripts/control/web/pointer.js',import.meta.url),'utf8');
const {mapPoint,Capture}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
test('resize and letterbox map the same screen position',()=>{
  for(const [width,height] of [[414,896],[207,448],[800,448],[207,800]]) {
    const rect={left:10,top:20,width,height};
    assert.deepEqual(mapPoint(10+width/2,20+height/2,rect),{x:.5,y:.5});
  }
  assert.equal(mapPoint(11,21,{left:10,top:20,width:800,height:448}),null);
  assert.equal(mapPoint(11,21,{left:10,top:20,width:207,height:800}),null);
});
test('tap, hold and curved drag preserve elapsed time',()=>{
  for(const duration of [10,50,900,5000]){
    const c=new Capture();c.start(1,{x:.2,y:.3},100);
    assert.deepEqual(c.finish(1,{x:.2,y:.3},100+duration),[{x:.2,y:.3,t:0},{x:.2,y:.3,t:Math.max(30,duration)}]);
  }
  const c=new Capture();c.start(1,{x:.2,y:.3},100);c.move(1,{x:.8,y:.5},300);
  assert.deepEqual(c.finish(1,{x:.3,y:.7},600),[{x:.2,y:.3,t:0},{x:.8,y:.5,t:200},{x:.3,y:.7,t:500}]);
});
test('cancellation, out-of-screen, duration and wrong pointer never submit',()=>{
  const c=new Capture();c.start(1,{x:0,y:0},0);assert.equal(c.start(2,{x:1,y:1},2),false);
  c.cancel();assert.equal(c.finish(1,{x:0,y:0},100),null);
  c.start(1,{x:0,y:0},0);c.move(1,null,10);assert.equal(c.finish(1,{x:0,y:0},100),null);
  c.start(1,{x:0,y:0},0);assert.equal(c.finish(1,{x:0,y:0},5001),null);
  c.start(1,{x:0,y:0},0);assert.equal(c.finish(2,{x:0,y:0},100),null);
});
