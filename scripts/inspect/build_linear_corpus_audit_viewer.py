#!/usr/bin/env python3
"""Build a small static browser for auditing an admitted linear corpus.

The generated page reads clips directly from the corpus and stores review marks
in browser localStorage.  Export the marks as JSON before clearing browser data.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from trueskate_ai.model1.linear.dataset import discover_basic_linear_samples  # noqa: E402


def _url_from_output(path: Path, output_dir: Path) -> str:
    relative = os.path.relpath(path, output_dir).replace(os.sep, "/")
    return quote(relative, safe="/._-")


def _records(corpus: Path, output_dir: Path, *, link_assets: bool) -> tuple[list[dict], dict]:
    samples, stats = discover_basic_linear_samples(corpus)
    records: list[dict] = []
    assets = output_dir / "assets"
    if link_assets:
        assets.mkdir()
    for index, sample in enumerate(sorted(samples, key=lambda item: item.as_posix())):
        video = sample / "frames.mp4"
        meta_path = sample / "meta.json"
        if not video.is_file():
            raise FileNotFoundError(f"accepted sample has no frames.mp4: {sample}")
        meta = json.loads(meta_path.read_text())
        if link_assets:
            video_link = assets / f"{index:06d}.mp4"
            meta_link = assets / f"{index:06d}.json"
            video_link.symlink_to(Path(os.path.relpath(video, assets)))
            meta_link.symlink_to(Path(os.path.relpath(meta_path, assets)))
            video_url = f"assets/{video_link.name}"
            meta_url = f"assets/{meta_link.name}"
        else:
            video_url = _url_from_output(video, output_dir)
            meta_url = _url_from_output(meta_path, output_dir)
        records.append({
            "id": sample.relative_to(corpus).as_posix(),
            "video": video_url,
            "metaFile": meta_url,
            "meta": meta,
        })
    return records, stats


def _render(*, records: list[dict], stats: dict, corpus: Path, title: str) -> str:
    payload = json.dumps({
        "title": title,
        "corpus": str(corpus),
        "stats": stats,
        "samples": records,
    }, separators=(",", ":")).replace("<", "\\u003c")
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, -apple-system, sans-serif; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #101214; color: #edf0f2; }}
button, select, input, textarea {{ font: inherit; color: inherit; background: #20252a; border: 1px solid #46505a; border-radius: 6px; }}
button {{ padding: .55rem .75rem; cursor: pointer; }}
button:hover {{ background: #2a3137; }}
button.active {{ border-color: #fff; box-shadow: 0 0 0 1px #fff inset; }}
header {{ max-width: 1250px; margin: auto; padding: 1rem 1.25rem 0; display: flex; gap: .65rem; align-items: center; flex-wrap: wrap; }}
header h1 {{ width: 100%; margin: 0; font-size: 1.35rem; }}
.count {{ color: #aeb8c2; }}
main {{ max-width: 1250px; margin: auto; padding: 1rem 1.25rem 2rem; display: grid; grid-template-columns: minmax(320px, 1fr) 360px; gap: 1.5rem; }}
.stage {{ display: flex; flex-direction: column; align-items: center; gap: .7rem; min-width: 0; }}
.video-shell {{ position: relative; display: inline-block; width: auto; height: auto; max-width: 100%; background: #050607; border: 1px solid #30363c; border-radius: 8px; overflow: hidden; line-height: 0; }}
video {{ display: block; width: auto; height: min(68vh, 720px); max-width: 100%; object-fit: contain; image-rendering: auto; }}
#gestureOverlay {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }}
#overlayLegend {{ position: absolute; top: .55rem; right: .55rem; padding: .3rem .45rem; border-radius: 5px; color: #fff; background: #111b; font-size: .72rem; line-height: 1; }}
.transport, .verdicts {{ display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; justify-content: center; }}
.frame {{ min-width: 12rem; text-align: center; color: #c3cbd2; }}
aside {{ padding: 1rem; align-self: start; background: #15191d; border: 1px solid #30363c; border-radius: 8px; }}
.sample-id {{ overflow-wrap: anywhere; font-weight: 700; margin-bottom: .75rem; }}
.facts {{ display: grid; grid-template-columns: auto 1fr; gap: .35rem .8rem; font-size: .86rem; margin: 1rem 0; }}
.facts dt {{ color: #8f9aa4; }} .facts dd {{ margin: 0; overflow-wrap: anywhere; }}
textarea {{ width: 100%; min-height: 6rem; padding: .6rem; resize: vertical; }}
details {{ margin-top: 1rem; }} pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: .72rem; color: #b9c3cb; }}
.good {{ color: #58db8b; }} .mild_issue {{ color: #f0c95b; }} .critical_issue {{ color: #ff7a7a; }}
.verdicts {{ display: grid; grid-template-columns: 1fr; margin: .75rem 0; }}
.verdicts button {{ text-align: left; }}
.help {{ color: #8f9aa4; font-size: .76rem; line-height: 1.5; margin-top: 1rem; }}
select, input {{ padding: .5rem; }} input[type=number] {{ width: 7rem; }}
#frameSeek {{ width: min(100%, 620px); padding: 0; }}
@media (max-width: 820px) {{ main {{ grid-template-columns: 1fr; }} video {{ height: min(64vh, 650px); }} }}
</style>
</head>
<body tabindex="-1">
<header>
  <h1>{safe_title}</h1>
  <span id="progress" class="count"></span>
  <select id="filter" aria-label="Review filter">
    <option value="all">All samples</option><option value="unreviewed">Unreviewed</option>
    <option value="good">Good</option><option value="mild_issue">Mild Issue</option><option value="critical_issue">Critical Issue</option>
  </select>
  <input id="jump" type="number" min="1" aria-label="Jump to visible sample number">
  <button id="random">Random</button><button id="export">Export reviews</button>
</header>
<main>
  <section class="stage">
    <div class="video-shell"><video id="video" preload="auto" playsinline></video><canvas id="gestureOverlay"></canvas><span id="overlayLegend" hidden>Executed swipe</span></div>
    <div id="frame" class="frame">Loading…</div><input id="frameSeek" type="range" min="0" value="0" aria-label="Frame position">
    <div class="transport">
      <button id="backFrame">← Previous frame</button><button id="play">Play</button><button id="nextFrame">Next frame →</button>
      <label>Speed <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option></select></label>
    </div>
    <div class="transport">
      <button id="previous">[ Previous sample</button><button id="next">Next sample ]</button><button id="overlayToggle">Show swipe</button>
    </div>
  </section>
  <aside>
    <p class="help"><b>← / →</b> changes one frame. <b>Shift + arrows</b> changes five. <b>[ / ]</b> changes sample. <b>Space</b> plays or pauses.</p>
    <div id="sampleId" class="sample-id"></div>
    <dl id="facts" class="facts"></dl>
    <div class="verdicts">
      <button id="good" class="good">G · Good</button><button id="mild_issue" class="mild_issue">M · Mild Issue</button>
      <button id="critical_issue" class="critical_issue">C · Critical Issue</button><button id="clear">Clear label</button>
    </div>
    <label for="note">Review note</label><textarea id="note" placeholder="What did you see?"></textarea>
    <details><summary>Full metadata</summary><pre id="metadata"></pre></details>
    <p class="help"><b>G/M/C</b> applies a label without reloading the clip. Press <b>Escape</b> after typing a note to return keyboard control to the viewer. Reviews remain in this browser; export JSON to preserve or share them.</p>
  </aside>
</main>
<script>
const DATA={payload};
const STORAGE_KEY='linear-corpus-audit-v1:'+DATA.corpus;
const OVERLAY_KEY=STORAGE_KEY+':show-swipe';
const video=document.getElementById('video');
const filter=document.getElementById('filter');
const note=document.getElementById('note');
const storedReviews=localStorage.getItem(STORAGE_KEY);
if(storedReviews&&!localStorage.getItem(STORAGE_KEY+':backup-before-ui-v2'))localStorage.setItem(STORAGE_KEY+':backup-before-ui-v2',storedReviews);
let reviews={{}}; try {{ reviews=JSON.parse(storedReviews||'{{}}'); }} catch (_) {{}}
const LEGACY_VERDICTS={{clean:'good',unsure:'mild_issue',issue:'critical_issue'}};
let migrated=false; for(const review of Object.values(reviews)){{if(LEGACY_VERDICTS[review.verdict]){{review.verdict=LEGACY_VERDICTS[review.verdict];migrated=true;}}}}
if(migrated)localStorage.setItem(STORAGE_KEY,JSON.stringify(reviews));
let overlayVisible=localStorage.getItem(OVERLAY_KEY)==='1';
let visible=[]; let cursor=0; let selectedFrame=0;
const el=id=>document.getElementById(id);
const save=()=>{{ localStorage.setItem(STORAGE_KEY,JSON.stringify(reviews)); updateProgress(); }};
const current=()=>visible[cursor];
function reviewFor(sample) {{ return reviews[sample.id]||{{}}; }}
function rebuild(keepId) {{
  const kind=filter.value;
  visible=DATA.samples.filter(s=>kind==='all' ? true : kind==='unreviewed' ? !reviewFor(s).verdict : reviewFor(s).verdict===kind);
  let found=visible.findIndex(s=>s.id===keepId); cursor=found>=0?found:Math.min(cursor,Math.max(0,visible.length-1));
  load();
}}
function frameCount() {{ return Number(current()?.meta?.n_frames)||1; }}
function mediaFrameIndex() {{
  if (!video.duration || !Number.isFinite(video.duration)) return 0;
  return Math.max(0,Math.min(frameCount()-1,Math.floor(video.currentTime/video.duration*frameCount())));
}}
function frameIndex() {{ return selectedFrame; }}
function updateFrame() {{
  const sample=current();
  if (!sample) {{ el('frame').textContent='No samples in this filter'; return; }}
  const idx=frameIndex(); const rel=sample.meta.frame_times?.[idx];
  el('frame').textContent=`Frame ${{idx+1}} / ${{frameCount()}} · ${{video.currentTime.toFixed(3)}} s${{rel===undefined?'':` · ${{Number(rel).toFixed(4)}} s from gesture start`}}`;
  el('frameSeek').value=idx;
}}
function showFrame(index) {{
  if (!current()) return; video.pause(); selectedFrame=Math.max(0,Math.min(frameCount()-1,Math.round(index)||0));
  if(video.duration&&Number.isFinite(video.duration))video.currentTime=Math.min(video.duration-.001,(selectedFrame+.08)*video.duration/frameCount());
  updateFrame(); drawOverlay();
}}
function stepFrame(delta) {{ showFrame(selectedFrame+delta); }}
function togglePlay() {{ if(video.paused){{if(video.ended)showFrame(0);video.play();}}else video.pause(); }}
function relativeVideoTime(sample, mediaTime) {{
  const times=sample?.meta?.frame_times||[];
  if (!times.length) return mediaTime-.5;
  if (!video.duration || !Number.isFinite(video.duration)) return Number(times[frameIndex()]);
  const position=Math.max(0,Math.min(times.length-1,mediaTime/video.duration*times.length));
  const low=Math.floor(position), high=Math.min(times.length-1,low+1), blend=position-low;
  return Number(times[low])+(Number(times[high])-Number(times[low]))*blend;
}}
function drawOverlay(mediaTime=video.currentTime) {{
  const canvas=el('gestureOverlay'); const sample=current();
  const width=video.clientWidth, height=video.clientHeight, ratio=window.devicePixelRatio||1;
  const pixelWidth=Math.max(1,Math.round(width*ratio)), pixelHeight=Math.max(1,Math.round(height*ratio));
  if(canvas.width!==pixelWidth)canvas.width=pixelWidth;if(canvas.height!==pixelHeight)canvas.height=pixelHeight;
  const context=canvas.getContext('2d'); context.setTransform(ratio,0,0,ratio,0,0); context.clearRect(0,0,width,height);
  if (!overlayVisible || !sample) return;
  const points=sample.meta.waypoints||[]; if(points.length<2) return;
  const start={{x:Number(points[0][0])*width,y:Number(points[0][1])*height}};
  const target={{x:Number(points.at(-1)[0])*width,y:Number(points.at(-1)[1])*height}};
  const relative=relativeVideoTime(sample,mediaTime), duration=Math.max(.001,Number(sample.meta.duration)||.001);
  const progress=Math.max(0,Math.min(1,relative/duration));
  const finish={{x:start.x+(target.x-start.x)*progress,y:start.y+(target.y-start.y)*progress}};
  el('overlayLegend').textContent=relative<0?`Executed swipe · starts in ${{(-relative).toFixed(3)}} s`:relative<=duration?`Executed swipe · t=+${{relative.toFixed(3)}} s · ${{Math.round(progress*100)}}%`:`Executed swipe · complete at +${{duration.toFixed(3)}} s`;
  if(relative<0)return;
  const angle=Math.atan2(target.y-start.y,target.x-start.x), travelled=Math.hypot(finish.x-start.x,finish.y-start.y), head=Math.min(14,travelled*.35);
  context.save(); context.strokeStyle='#ff9f1c'; context.fillStyle='#ff9f1c'; context.lineWidth=5; context.lineCap='round'; context.lineJoin='round'; context.shadowColor='#000d'; context.shadowBlur=4;
  context.beginPath(); context.moveTo(start.x,start.y); context.lineTo(finish.x,finish.y); context.stroke();
  if(head>3){{context.beginPath(); context.moveTo(finish.x,finish.y); context.lineTo(finish.x-head*Math.cos(angle-.55),finish.y-head*Math.sin(angle-.55)); context.lineTo(finish.x-head*Math.cos(angle+.55),finish.y-head*Math.sin(angle+.55)); context.closePath(); context.fill();}}
  context.fillStyle='#101214'; context.beginPath(); context.arc(start.x,start.y,8,0,Math.PI*2); context.fill(); context.strokeStyle='#ff9f1c'; context.lineWidth=4; context.stroke();
  context.fillStyle='#fff'; context.beginPath(); context.arc(finish.x,finish.y,5,0,Math.PI*2); context.fill(); context.strokeStyle='#ff9f1c'; context.lineWidth=3; context.stroke();
  context.restore();
}}
let overlayFramePending=false;
function scheduleOverlayFrame() {{
  if(overlayFramePending||video.paused||video.ended)return; overlayFramePending=true;
  if('requestVideoFrameCallback' in video){{video.requestVideoFrameCallback((_now,metadata)=>{{overlayFramePending=false;drawOverlay(metadata.mediaTime);scheduleOverlayFrame();}});}}
  else{{requestAnimationFrame(()=>{{overlayFramePending=false;drawOverlay();scheduleOverlayFrame();}});}}
}}
function updateOverlay() {{
  el('overlayToggle').textContent=overlayVisible?'Hide swipe':'Show swipe';
  el('overlayToggle').classList.toggle('active',overlayVisible); el('overlayLegend').hidden=!overlayVisible; drawOverlay();
}}
function fact(label,value) {{ return `<dt>${{label}}</dt><dd>${{value??'—'}}</dd>`; }}
function refreshReviewUi() {{
  const sample=current(), review=sample?reviewFor(sample):{{}};
  note.value=review.note||'';
  for (const verdict of ['good','mild_issue','critical_issue']) el(verdict).classList.toggle('active',review.verdict===verdict);
}}
function load() {{
  const sample=current(); updateProgress();
  if (!sample) {{ video.removeAttribute('src'); video.load(); el('sampleId').textContent='No samples'; el('facts').innerHTML=''; note.value=''; return; }}
  selectedFrame=0; video.src=sample.video; video.load(); el('frameSeek').max=frameCount()-1;
  el('sampleId').textContent=sample.id;
  const m=sample.meta, points=m.waypoints||[];
  el('facts').innerHTML=fact('Visible index',`${{cursor+1}} / ${{visible.length}}`)+fact('Device',m.device)+fact('Park',m.park)+fact('Session',m.session)+fact('Gesture index',m.gesture_index)+fact('Start',points[0]?.map(v=>Number(v).toFixed(3)).join(', '))+fact('End',points.at(-1)?.map(v=>Number(v).toFixed(3)).join(', '))+fact('Duration',m.duration===undefined?'—':Number(m.duration).toFixed(3)+' s')+fact('Frames',m.n_frames)+fact('Capture offset',m.capture_offset_s===undefined?'—':Number(m.capture_offset_s).toFixed(4)+' s')+fact('Metadata',`<a href="${{sample.metaFile}}" target="_blank">open JSON</a>`);
  el('metadata').textContent=JSON.stringify(m,null,2);
  refreshReviewUi();
  el('jump').value=cursor+1; updateFrame(); requestAnimationFrame(()=>drawOverlay());
}}
function move(delta) {{ if (!visible.length) return; cursor=(cursor+delta+visible.length)%visible.length; load(); }}
function mark(verdict) {{
  const sample=current(); if (!sample) return;
  reviews[sample.id]={{...reviewFor(sample),verdict,note:note.value,updatedAt:new Date().toISOString()}}; save(); refreshReviewUi();
}}
function updateProgress() {{
  const counts={{good:0,mild_issue:0,critical_issue:0}}; Object.values(reviews).forEach(r=>{{if(counts[r.verdict]!==undefined)counts[r.verdict]++;}});
  const reviewed=counts.good+counts.mild_issue+counts.critical_issue;
  el('progress').textContent=`${{reviewed}} / ${{DATA.samples.length}} reviewed · ${{counts.good}} good · ${{counts.mild_issue}} mild · ${{counts.critical_issue}} critical`;
}}
el('previous').onclick=()=>move(-1); el('next').onclick=()=>move(1);
el('backFrame').onclick=()=>stepFrame(-1); el('nextFrame').onclick=()=>stepFrame(1);
el('play').onclick=togglePlay;
el('overlayToggle').onclick=()=>{{overlayVisible=!overlayVisible;localStorage.setItem(OVERLAY_KEY,overlayVisible?'1':'0');updateOverlay();}};
el('good').onclick=()=>mark('good'); el('mild_issue').onclick=()=>mark('mild_issue'); el('critical_issue').onclick=()=>mark('critical_issue');
el('clear').onclick=()=>{{const s=current(); if(s){{const prior=reviewFor(s);reviews[s.id]={{...prior,verdict:undefined,updatedAt:new Date().toISOString()}};save();refreshReviewUi();}}}};
filter.onchange=()=>rebuild(current()?.id);
el('random').onclick=()=>{{if(visible.length){{cursor=Math.floor(Math.random()*visible.length);load();}}}};
el('jump').onchange=e=>{{cursor=Math.max(0,Math.min(visible.length-1,Number(e.target.value)-1||0));load();}};
note.oninput=()=>{{const s=current();if(!s)return;const prior=reviewFor(s);if(prior.verdict||note.value){{reviews[s.id]={{...prior,note:note.value,updatedAt:new Date().toISOString()}};save();}}}};
el('frameSeek').oninput=e=>showFrame(Number(e.target.value));
el('speed').onchange=e=>{{video.playbackRate=Number(e.target.value);}};
el('export').onclick=()=>{{
  const blob=new Blob([JSON.stringify({{schema:'linear-corpus-audit-v1',corpus:DATA.corpus,exportedAt:new Date().toISOString(),reviews}},null,2)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='linear-corpus-audit-reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}};
video.addEventListener('loadedmetadata',()=>{{showFrame(selectedFrame);drawOverlay();}}); video.addEventListener('timeupdate',()=>{{if(!video.paused)selectedFrame=mediaFrameIndex();updateFrame();drawOverlay();}}); video.addEventListener('seeked',()=>drawOverlay());
video.addEventListener('play',()=>{{el('play').textContent='Pause';scheduleOverlayFrame();}}); video.addEventListener('pause',()=>{{el('play').textContent='Play';drawOverlay();}}); video.addEventListener('ended',()=>{{el('play').textContent='Play';selectedFrame=frameCount()-1;updateFrame();drawOverlay();}});
video.addEventListener('click',togglePlay);
new ResizeObserver(()=>drawOverlay()).observe(video);
document.addEventListener('keydown',e=>{{
  if (['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)){{if(e.key==='Escape'){{e.target.blur();document.body.focus();}}return;}}
  if(e.key===' '){{e.preventDefault();togglePlay();}}
  else if(e.key==='ArrowLeft'){{e.preventDefault();stepFrame(e.shiftKey?-5:-1);}} else if(e.key==='ArrowRight'){{e.preventDefault();stepFrame(e.shiftKey?5:1);}}
  else if(e.key==='[')move(-1); else if(e.key===']')move(1);
  else if(e.key.toLowerCase()==='g')mark('good'); else if(e.key.toLowerCase()==='m')mark('mild_issue'); else if(e.key.toLowerCase()==='c')mark('critical_issue');
}});
visible=DATA.samples.slice(); load(); updateOverlay();
</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Empty or new output directory for index.html")
    parser.add_argument("--title", default="Model 1 linear corpus audit")
    parser.add_argument("--link-assets", action="store_true",
                        help="Create an isolated, serveable asset directory using symlinks")
    args = parser.parse_args()
    corpus = args.corpus.resolve()
    output = args.out.resolve()
    if not corpus.is_dir():
        parser.error(f"corpus does not exist: {corpus}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records, stats = _records(corpus, output, link_assets=args.link_assets)
    if not records:
        parser.error("strict loader found no admitted samples")
    index = output / "index.html"
    index.write_text(_render(records=records, stats=stats, corpus=corpus, title=args.title))
    print(f"Wrote {index} with {len(records)} strict samples")


if __name__ == "__main__":
    main()
