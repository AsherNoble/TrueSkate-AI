#!/usr/bin/env python3
"""Build a blinded frame-onset annotator for selected linear clips."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model 1 trace-onset check</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif}
*{box-sizing:border-box}body{margin:0;background:#101214;color:#edf0f2}
main{max-width:1150px;margin:auto;padding:16px 22px}
h1{font-size:1.4rem;margin:0 0 8px}p{color:#b8c3cc;line-height:1.45}
.layout{display:grid;grid-template-columns:minmax(360px,1fr) 280px;gap:22px}
.stage{display:flex;flex-direction:column;align-items:center;gap:10px;min-width:0}
.video-shell{max-width:100%;background:#050607;border:1px solid #30363c;border-radius:8px;overflow:hidden}
video{display:block;width:auto;height:min(68vh,720px);max-width:100%;object-fit:contain}
button,input,textarea{font:inherit;color:inherit;background:#20252a;border:1px solid #46505a;border-radius:6px}
button{padding:8px 11px;cursor:pointer}button:hover{background:#2c343b}
.transport{display:flex;gap:7px;flex-wrap:wrap;justify-content:center}
#seek{width:min(100%,620px);padding:0}#frame,#progress{font-variant-numeric:tabular-nums}
aside{padding:15px;border:1px solid #30363c;border-radius:8px;align-self:start;background:#15191d}
.actions{display:grid;gap:8px;margin:14px 0}.primary{background:#215d45}.unsure{background:#544521}
textarea{width:100%;min-height:78px;padding:8px;resize:vertical}
.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:5px;margin-top:12px}
.grid button{padding:5px 0;font-size:.82rem}.grid .marked{color:#70d8a1}
.grid .uncertain{color:#f3c863}.grid .current{outline:2px solid #fff}
@media(max-width:780px){.layout{grid-template-columns:1fr}video{height:min(60vh,600px)}}
</style></head><body tabindex="-1"><main>
<h1>First visible trace · 24-clip check</h1>
<p>Use ← / → to change one frame. At the <b>first frame where a new swipe trace appears</b>,
press <b>M</b> to mark it and advance. Press <b>U</b> if it is unclear. The clips are
shuffled, and their timing categories are hidden.</p>
<p id="progress"></p>
<div class="layout"><section class="stage">
<div class="video-shell"><video id="video" preload="auto" playsinline></video></div>
<div id="frame">Loading…</div><input id="seek" type="range" min="0" value="0" aria-label="Frame position">
<div class="transport"><button id="backFrame">← Frame</button><button id="play">Play / pause</button>
<button id="nextFrame">Frame →</button></div>
<div class="transport"><button id="previous">[ Previous clip</button><button id="next">Next clip ]</button></div>
</section><aside>
<strong id="clip"></strong><p id="currentMark"></p>
<div class="actions"><button id="mark" class="primary">M · Mark this frame &amp; next clip</button>
<button id="unsure" class="unsure">U · Unclear / no visible trace</button>
<button id="clear">Clear this mark</button></div>
<label for="note">Optional note</label><textarea id="note" placeholder="What made this clip difficult?"></textarea>
<p>Marks save in this browser. <b>Export the JSON</b> when finished. Press Escape after
typing a note to return to frame shortcuts.</p>
<button id="export">Export onset labels</button><div class="grid" id="grid"></div>
</aside></div></main>
<script>
const DATA=__PAYLOAD__;
const KEY='linear-onset-validation-v1:'+DATA.seed;
const video=document.getElementById('video'),el=id=>document.getElementById(id);
let labels={};try{labels=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(_){}
let cursor=0,selectedFrame=0,sourceToken=0,videoObjectUrl='';
const current=()=>DATA.samples[cursor],frameCount=()=>current().n_frames;
function save(){localStorage.setItem(KEY,JSON.stringify(labels));updateProgress();updateGrid();}
function updateProgress(){
  const values=Object.values(labels),marked=values.filter(x=>Number.isInteger(x.frame_index_0based)).length;
  const unsure=values.filter(x=>x.uncertain).length;
  el('progress').textContent=(marked+unsure)+' / '+DATA.samples.length+
    ' reviewed · '+marked+' marked · '+unsure+' unclear';
}
function updateGrid(){
  el('grid').replaceChildren();
  DATA.samples.forEach((sample,index)=>{
    const button=document.createElement('button'),label=labels[sample.id];
    button.textContent=String(index+1);button.title='Clip '+(index+1);
    if(label?.uncertain)button.classList.add('uncertain');
    else if(Number.isInteger(label?.frame_index_0based))button.classList.add('marked');
    if(index===cursor)button.classList.add('current');
    button.onclick=()=>{cursor=index;load();};el('grid').append(button);
  });
}
function mediaFrameIndex(){
  if(!Number.isFinite(video.duration)||video.duration<=0)return 0;
  return Math.max(0,Math.min(frameCount()-1,Math.floor(video.currentTime/video.duration*frameCount())));
}
function seekSelectedFrame(){
  if(video.readyState<1||!Number.isFinite(video.duration)||video.duration<=0)return;
  const target=Math.min(video.duration-.001,(selectedFrame+.5)*video.duration/frameCount());
  if(Math.abs(video.currentTime-target)>.002)video.currentTime=target;
}
function updateFrame(){el('frame').textContent='Frame '+(selectedFrame+1)+' / '+frameCount();el('seek').value=selectedFrame;}
function showFrame(index){
  video.pause();selectedFrame=Math.max(0,Math.min(frameCount()-1,Math.round(index)||0));
  seekSelectedFrame();updateFrame();
}
function move(delta){cursor=(cursor+delta+DATA.samples.length)%DATA.samples.length;load();}
async function loadVideo(sample,token){
  try{
    const response=await fetch(sample.video,{cache:'force-cache'});
    if(!response.ok)throw Error('HTTP '+response.status);
    const blob=await response.blob();if(token!==sourceToken)return;
    const nextUrl=URL.createObjectURL(blob),oldUrl=videoObjectUrl;
    videoObjectUrl=nextUrl;video.src=nextUrl;video.load();
    if(oldUrl)URL.revokeObjectURL(oldUrl);
  }catch(error){if(token===sourceToken)el('frame').textContent='Could not load clip: '+error.message;}
}
function load(){
  const sample=current(),token=++sourceToken,mark=labels[sample.id];
  selectedFrame=0;video.pause();video.removeAttribute('src');video.load();
  el('clip').textContent='Clip '+(cursor+1)+' of '+DATA.samples.length;
  el('seek').max=frameCount()-1;
  el('currentMark').textContent=mark?.uncertain?'Marked unclear':
    Number.isInteger(mark?.frame_index_0based)?'Marked frame '+(mark.frame_index_0based+1):'Not marked yet';
  el('note').value=mark?.note||'';el('frame').textContent='Loading clip…';
  updateGrid();loadVideo(sample,token);
}
function mark(uncertain){
  const frame=uncertain?null:selectedFrame,sample=current();
  labels[sample.id]={frame_index_0based:frame,displayed_frame_1based:frame===null?null:frame+1,
    uncertain:uncertain,note:el('note').value,updated_at:new Date().toISOString()};
  save();move(1);
}
function togglePlay(){if(video.paused){if(video.ended)showFrame(0);video.play();}else video.pause();}
el('previous').onclick=()=>move(-1);el('next').onclick=()=>move(1);
el('backFrame').onclick=()=>showFrame(selectedFrame-1);el('nextFrame').onclick=()=>showFrame(selectedFrame+1);
el('play').onclick=togglePlay;el('mark').onclick=()=>mark(false);el('unsure').onclick=()=>mark(true);
el('clear').onclick=()=>{delete labels[current().id];save();load();};
el('note').oninput=()=>{const id=current().id;if(labels[id]){labels[id].note=el('note').value;save();}};
el('seek').oninput=e=>showFrame(Number(e.target.value));
el('export').onclick=()=>{
  const output={schema:'linear-onset-validation-v1',selection_seed:DATA.seed,
    exported_at:new Date().toISOString(),labels:labels};
  const blob=new Blob([JSON.stringify(output,null,2)],{type:'application/json'});
  const link=document.createElement('a');link.href=URL.createObjectURL(blob);
  link.download='model1-onset-validation-20260924.json';link.click();
  setTimeout(()=>URL.revokeObjectURL(link.href),1000);
};
video.addEventListener('loadedmetadata',seekSelectedFrame);
video.addEventListener('loadeddata',seekSelectedFrame);
video.addEventListener('canplay',seekSelectedFrame);
video.addEventListener('timeupdate',()=>{if(!video.paused){selectedFrame=mediaFrameIndex();updateFrame();}});
video.addEventListener('seeked',()=>{
  if(video.paused&&mediaFrameIndex()!==selectedFrame)requestAnimationFrame(seekSelectedFrame);
});
video.addEventListener('click',togglePlay);
document.addEventListener('keydown',e=>{
  if(['INPUT','TEXTAREA'].includes(e.target.tagName)){
    if(e.key==='Escape'){e.target.blur();document.body.focus();}return;
  }
  if(e.key==='ArrowLeft'){e.preventDefault();showFrame(selectedFrame+(e.shiftKey?-5:-1));}
  else if(e.key==='ArrowRight'){e.preventDefault();showFrame(selectedFrame+(e.shiftKey?5:1));}
  else if(e.key==='[')move(-1);else if(e.key===']')move(1);
  else if(e.key.toLowerCase()==='m')mark(false);
  else if(e.key.toLowerCase()==='u')mark(true);
  else if(e.key===' '){e.preventDefault();togglePlay();}
});
updateProgress();load();
</script></body></html>
"""


def build(selection: Path, selected_root: Path, output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output must be empty: {output}")
    manifest = json.loads(selection.read_text())
    if manifest.get("schema") != "model1-onset-validation-selection-v1":
        raise ValueError("unexpected selection schema")
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir()
    records = []
    for item in manifest["samples"]:
        index = item["order"] - 1
        sample = selected_root / f"{index:03d}" / item["source"]
        meta = json.loads((sample / "meta.json").read_text())
        video = sample / "frames.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        link = assets / f"{index:03d}.mp4"
        link.symlink_to(os.path.relpath(video, assets))
        records.append({
            "id": f"{index:03d}/{item['source']}",
            "video": f"assets/{link.name}",
            "n_frames": int(meta["n_frames"]),
        })
    if len(records) != 24 or any(r["n_frames"] != 32 for r in records):
        raise ValueError("expected 24 selected 32-frame clips")
    payload = json.dumps({"seed": str(manifest["seed"]), "samples": records},
                         separators=(",", ":")).replace("<", "\\u003c")
    (output / "index.html").write_text(HTML.replace("__PAYLOAD__", payload))
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count = build(args.selection.resolve(), args.selected_root.resolve(), args.out.resolve())
    print(f"Wrote {args.out / 'index.html'} with {count} blinded clips")


if __name__ == "__main__":
    main()
