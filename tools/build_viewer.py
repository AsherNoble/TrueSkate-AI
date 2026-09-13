#!/usr/bin/env python3
"""Build a frame-by-frame gesture-onset annotator for an XCTest segment recording.

Given a session directory containing ``segment_00000.mov`` (or an explicit
``--mov``), this:

  1. extracts every decoded frame to ``<dir>/frames/frame_NNNNNN.jpg`` (ffmpeg,
     original resolution, one jpg per decoded frame),
  2. writes ``<dir>/frame_index.json`` with each frame's presentation timestamp
     (``ffprobe -show_frames``), and
  3. renders ``<dir>/index.html`` — a self-contained annotator that shows every
     frame and lets a human mark the first visible frame of each touch, then
     download a ``timing-labels-*.json`` of ``{frame, time_s, kind, note}`` marks.

The annotator shows no predicted gestures or expected start times, so the marks
are an independent human label to compare against WDA's internal timing.

Frame numbers are zero-based; times are the recording's actual presentation
timestamps (no timing correction applied). Requires ffmpeg and ffprobe on PATH.

Example:
    python tools/build_viewer.py \
        --dir /Users/training-server/.../run05 \
        --session-id run05 \
        --recording xr2-duration-repeat-run05/segment_00000.mov \
        --title "Skateboard GB 2024 . XR2 . run05 (instrumented WDA)." \
        --footer "Expected: 1 calibration touch, then N gesture starts."
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

# Proven annotator template. Placeholders (@@...@@ and __TIMES__) are substituted
# below; the CSS/JS body is unchanged from the hand-built per-session viewers.
_TEMPLATE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gesture start annotation</title>
<style>body{margin:0;background:#14171c;color:#e9edf4;font:16px system-ui}main{max-width:1250px;margin:auto;padding:20px}h1{font-size:24px;margin:0 0 8px}p{color:#bfc8d6;line-height:1.5}.layout{display:grid;grid-template-columns:minmax(320px,1fr) 360px;gap:24px}.image{height:70vh;text-align:center;background:#080a0d;overflow:auto}#frame{height:100%;max-width:100%;object-fit:contain}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}button,select,input{font:inherit;padding:8px;border-radius:6px;border:1px solid #536074;background:#252e3b;color:white}button{cursor:pointer}button.primary{background:#226c58}button:disabled{opacity:.4;cursor:default}#seek{width:100%;padding:0}#number{width:95px}#note{width:95%}#status{font-variant-numeric:tabular-nums;font-size:19px;margin:12px 0}#marks{max-height:43vh;overflow:auto;padding-left:24px}li{margin:8px 0}li button{padding:3px 7px;font-size:13px}.strip{display:flex;gap:4px;overflow:auto}.strip button{padding:2px;min-width:70px}.strip img{width:68px;height:146px;object-fit:contain}.selected{outline:2px solid #62d8ac}small{color:#bfc8d6}textarea{width:95%;height:90px;background:#10151c;color:white}details{margin-top:16px}@media(max-width:800px){.layout{grid-template-columns:1fr}.image{height:60vh}}</style>
<main><h1>Mark the first visible frame of each gesture</h1><p>@@INTRO@@ Every decoded frame is included at its original resolution. No predicted gestures or expected start times are shown.</p>
<div class="layout"><section><div class="image"><img id="frame" alt="Original recording frame"></div><div id="status"></div><input id="seek" type="range" min="0" value="0" aria-label="Frame position"><div class="controls"><button id="back30">-30 frames</button><button id="back">&larr; Previous</button><button id="play">Play</button><button id="next">Next &rarr;</button><button id="next30">+30 frames</button></div><div class="controls"><label>Frame <input id="number" type="number" min="0" value="0"></label><button id="go">Go</button><label>Playback <select id="speed"><option value="100">10 fps</option><option value="200">5 fps</option><option value="33">30 fps</option></select></label></div><div class="strip" id="strip"></div></section>
<aside><p>Use <b>&larr; / &rarr;</b> for one frame, <b>Shift + arrows</b> for 30. Press <b>M</b> to mark the current frame. Space plays or pauses.</p><p>Mark the first frame where you can see a new touch begin. If its identity is unclear, choose &ldquo;Uncertain&rdquo; and leave a note.</p><label>Touch type <select id="kind"><option value="gesture">Gesture</option><option value="calibration">Calibration touch</option><option value="reset">Reset</option><option value="uncertain">Uncertain</option></select></label><p><input id="note" placeholder="Optional note" aria-label="Annotation note"></p><button class="primary" id="mark">Mark start (M)</button><h2 id="count">0 starts marked</h2><ol id="marks"></ol><button class="primary" id="export">Download labels.json</button><p><small id="saved">Labels save in this browser when available. Download a copy when finished, then attach it in our chat. You can also send just the frame numbers.</small></p><details><summary>Copyable labels / restore saved labels</summary><textarea id="text" readonly aria-label="Labels JSON"></textarea><p><label>Import labels <input id="import" type="file" accept="application/json"></label></p></details><p><small>Frame numbers start at 0. Times use the recording&rsquo;s actual presentation timestamps. No timing corrections have been applied. @@FOOTER@@</small></p></aside></div></main>
<script>const TIMES=__TIMES__;const KEY='@@KEY@@';let current=0,marks=[],timer=null;const $=id=>document.getElementById(id);const path=i=>'frames/frame_'+String(i).padStart(6,'0')+'.jpg';try{marks=JSON.parse(localStorage.getItem(KEY)||'[]')}catch(e){}$('seek').max=TIMES.length-1;$('number').max=TIMES.length-1;
function payload(){return {recording:'@@RECORDING@@',frame_numbering:'zero-based',frame_count:TIMES.length,labels:marks.slice().sort((a,b)=>a.frame-b.frame)}}
function list(){marks.sort((a,b)=>a.frame-b.frame);$('marks').replaceChildren();marks.forEach((m,n)=>{let li=document.createElement('li'),b=document.createElement('button');b.textContent='Frame '+m.frame+' · '+m.time_s.toFixed(3)+'s';b.onclick=()=>{stop();show(m.frame)};li.append(b,document.createTextNode(' '+m.kind+(m.note?' — '+m.note:'')));let d=document.createElement('button');d.textContent='Remove';d.onclick=()=>{marks.splice(n,1);save()};li.append(' ',d);$('marks').append(li)});$('count').textContent=marks.length+' starts marked';$('text').value=JSON.stringify(payload(),null,2)}
function save(){try{localStorage.setItem(KEY,JSON.stringify(marks))}catch(e){$('saved').textContent='Browser storage unavailable. Download labels before closing.'}list()}
function show(i){current=Math.max(0,Math.min(TIMES.length-1,Math.round(i)||0));$('frame').src=path(current);$('seek').value=current;$('number').value=current;$('status').textContent='Frame '+current+' / '+(TIMES.length-1)+' · '+TIMES[current].toFixed(3)+' seconds';$('back').disabled=current===0;$('next').disabled=current===TIMES.length-1;$('strip').replaceChildren();for(let k=Math.max(0,current-3);k<=Math.min(TIMES.length-1,current+3);k++){let b=document.createElement('button'),im=document.createElement('img');im.src=path(k);im.alt='Frame '+k;b.append(im,document.createElement('br'),document.createTextNode(k));if(k===current)b.className='selected';b.onclick=()=>{stop();show(k)};$('strip').append(b)}}
function stop(){clearInterval(timer);timer=null;$('play').textContent='Play'}function play(){if(timer){stop();return}if(current===TIMES.length-1)show(0);$('play').textContent='Pause';timer=setInterval(()=>{if(current===TIMES.length-1)stop();else show(current+1)},Number($('speed').value))}
function mark(){stop();if(!marks.some(m=>m.frame===current)){marks.push({frame:current,time_s:TIMES[current],kind:$('kind').value,note:$('note').value});$('note').value='';save()}}
$('back').onclick=()=>{stop();show(current-1)};$('next').onclick=()=>{stop();show(current+1)};$('back30').onclick=()=>{stop();show(current-30)};$('next30').onclick=()=>{stop();show(current+30)};$('seek').oninput=e=>{stop();show(Number(e.target.value))};$('go').onclick=()=>{stop();show(Number($('number').value))};$('play').onclick=play;$('speed').onchange=()=>{if(timer){stop();play()}};$('mark').onclick=mark;
$('export').onclick=()=>{let u=URL.createObjectURL(new Blob([JSON.stringify(payload(),null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=u;a.download='@@DOWNLOAD@@';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
$('import').onchange=async e=>{try{let p=JSON.parse(await e.target.files[0].text());if(p.recording!==payload().recording||!Array.isArray(p.labels))throw Error('Wrong recording or format');for(let m of p.labels){if(!Number.isInteger(m.frame)||m.frame<0||m.frame>=TIMES.length)throw Error('Invalid frame')}marks=p.labels.map(m=>({...m,time_s:TIMES[m.frame]}));save()}catch(err){alert(err.message)}};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();stop();show(current+(e.key==='ArrowRight'?1:-1)*(e.shiftKey?30:1))}else if(e.key.toLowerCase()==='m'){e.preventDefault();mark()}else if(e.code==='Space'){e.preventDefault();play()}});list();show(0);
</script></html>'''


def extract_frames(mov: Path, frames_dir: Path) -> None:
    """Extract every decoded frame at original resolution as frame_NNNNNN.jpg."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(mov),
         "-fps_mode", "passthrough", "-qscale:v", "3", "-start_number", "0",
         str(frames_dir / "frame_%06d.jpg")],
        check=True,
    )


def build_frame_index(mov: Path, index_path: Path) -> list[float]:
    """Write ffprobe per-frame PTS to index_path and return the times list."""
    raw = subprocess.run(
        ["ffprobe", "-loglevel", "error", "-select_streams", "v:0",
         "-show_frames", "-show_entries", "frame=best_effort_timestamp_time",
         "-print_format", "json", str(mov)],
        check=True, capture_output=True, text=True,
    ).stdout
    index_path.write_text(raw)
    data = json.loads(raw)
    return [float(f["best_effort_timestamp_time"]) for f in data["frames"]]


def render(index_html: Path, times: list[float], *, session_id: str,
           recording: str, intro: str, footer: str, download_name: str) -> None:
    html = (_TEMPLATE
            .replace("@@INTRO@@", intro)
            .replace("@@KEY@@", f"trueskate-timing-{session_id}-v1")
            .replace("@@RECORDING@@", recording)
            .replace("@@FOOTER@@", footer)
            .replace("@@DOWNLOAD@@", download_name)
            .replace("__TIMES__", json.dumps(times)))
    index_html.write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True,
                    help="Session directory (absolute). frames/, frame_index.json "
                         "and index.html are written here.")
    ap.add_argument("--mov", type=Path,
                    help="Recording path. Default: <dir>/segment_00000.mov")
    ap.add_argument("--session-id", required=True,
                    help="Short id used for the localStorage key and default "
                         "download filename, e.g. 'run05'.")
    ap.add_argument("--recording", required=True,
                    help="Recording identifier stored in the exported labels "
                         "(the import guard checks it must match).")
    ap.add_argument("--title", default="",
                    help="Intro sentence describing the recording (park/device/run).")
    ap.add_argument("--footer", default="",
                    help="Optional trailing note, e.g. the expected touch count.")
    ap.add_argument("--download-name",
                    help="Exported filename. Default: timing-labels-<session-id>.json")
    ap.add_argument("--skip-frames", action="store_true",
                    help="Reuse an existing frames/ dir and frame_index.json.")
    args = ap.parse_args()

    session_dir = args.dir.resolve()
    mov = (args.mov or session_dir / "segment_00000.mov").resolve()
    frames_dir = session_dir / "frames"
    index_path = session_dir / "frame_index.json"
    index_html = session_dir / "index.html"
    download_name = args.download_name or f"timing-labels-{args.session_id}.json"

    if not args.skip_frames:
        if not mov.is_file():
            raise SystemExit(f"Recording not found: {mov}")
        extract_frames(mov, frames_dir)
        times = build_frame_index(mov, index_path)
    else:
        times = [float(f["best_effort_timestamp_time"])
                 for f in json.loads(index_path.read_text())["frames"]]

    n_jpg = len(list(frames_dir.glob("frame_*.jpg")))
    if n_jpg != len(times):
        raise SystemExit(f"Frame/index mismatch: {n_jpg} jpgs vs {len(times)} index entries")

    render(index_html, times, session_id=args.session_id, recording=args.recording,
           intro=args.title, footer=args.footer, download_name=download_name)
    print(f"{index_html}  ({len(times)} frames, exports {download_name})")


if __name__ == "__main__":
    main()
