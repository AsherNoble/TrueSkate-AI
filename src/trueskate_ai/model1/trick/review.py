"""Operator spot check of admitted clips, verdict codes and the batch summary (laptop).

The page plays each picked clip with a ring at the commanded position of whichever
stroke is active, so gross faults show: a missing or extra touch, a trail far from its
ring, a menu. It is one static file; open it in Chrome, no server.
"""
from __future__ import annotations

from collections import Counter
import datetime as dt
import html
import json
from pathlib import Path
import re
import statistics

import numpy as np

from trueskate_ai.model1.curved.contract import digest, save_json
from .contract import EXPERIMENT, HEIGHT, SEED, WIDTH

DEFAULT_COUNT = 20
RING_PT = 10
VERDICTS = dict(K="ok", F="fault", U="unsure")
TOKEN = re.compile(r"(\d{1,2})\s*([KFU])", re.IGNORECASE)


def _read(path):
    return json.loads(Path(path).read_text())


def admitted_clips(root) -> list[dict]:
    """Clips whose recording admission lists them, in recording and run order."""
    clips = []
    for admission in sorted(Path(root).glob("recordings/*/admission.json")):
        result = _read(admission)
        if not result.get("accepted"):
            continue
        for number in sorted(result["admitted"]):
            folder = admission.parent/f"run-{number:05d}"
            meta = _read(folder/"meta.json")
            clips.append(dict(recording=admission.parent.name, run=number, folder=folder, meta=meta))
    return clips


def select(root, count: int = DEFAULT_COUNT) -> dict:
    """A seeded random pick of admitted clips with two-digit codes, kept in review/selection.json."""
    if not 1 <= count <= 99:
        raise ValueError("pick 1–99 clips")
    clips = admitted_clips(root)
    if not clips:
        raise ValueError("no admitted clips to review")
    picked = sorted(np.random.default_rng([SEED, 1]).choice(len(clips), min(count, len(clips)), replace=False))
    rows = [dict(code=f"{i:02d}", recording=clips[j]["recording"], run=clips[j]["run"],
                 clip_sha256=clips[j]["meta"]["clip_sha256"]) for i, j in enumerate(picked, 1)]
    selection = dict(experiment=EXPERIMENT, seed=[SEED, 1], admitted=len(clips), clips=rows)
    selection["sha256"] = digest(selection)
    return selection


def load_selection(root) -> dict:
    selection = _read(Path(root)/"review"/"selection.json")
    if digest({k: v for k, v in selection.items() if k != "sha256"}) != selection.get("sha256"):
        raise ValueError("review selection changed after it was written")
    return selection


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trick-speed spot check</title>
<style>
:root { --bg: #fff; --fg: #111; --muted: #666; --line: #ddd; }
@media (prefers-color-scheme: dark) { :root { --bg: #111; --fg: #eee; --muted: #aaa; --line: #333; } }
body { background: var(--bg); color: var(--fg); font: 15px/1.4 -apple-system, system-ui, sans-serif;
       margin: 0 auto; max-width: 1120px; padding: 16px 16px 150px; }
section { border-top: 1px solid var(--line); padding: 12px 0; display: flex; gap: 16px; flex-wrap: wrap; }
.stage { position: relative; line-height: 0; }
video { height: 70vh; max-width: 100%; background: #000; }
canvas { position: absolute; left: 0; top: 0; pointer-events: none; }
.side { flex: 1; min-width: 240px; }
.muted { color: var(--muted); }
label { display: block; margin: 4px 0; cursor: pointer; }
button { font: inherit; padding: 3px 10px; margin: 2px 2px 2px 0; }
footer { position: fixed; left: 0; right: 0; bottom: 0; background: var(--bg); border-top: 1px solid var(--line);
         padding: 10px 16px; }
textarea { width: 100%; box-sizing: border-box; font: 13px ui-monospace, monospace; background: var(--bg);
           color: var(--fg); border: 1px solid var(--line); }
</style></head><body>
<h1>Trick-speed spot check</h1>
<p>Each clip holds one run of 1–4 fast touches. The cyan ring marks where each touch was commanded, at the
time it was commanded. Choose <b>OK</b> when every touch leaves its trail with its ring and nothing else
touches the screen. Choose <b>Fault</b> for a missing or extra touch, a trail far from its ring, or a menu.
Choose <b>Unsure</b> if you can't tell. Say what you saw for any Fault or Unsure. When done, press
<b>Copy</b> and paste the codes into the chat.</p>
<p><label style="display:inline"><input type="checkbox" id="paths"> Show full paths</label></p>
__SECTIONS__
<footer><div><span id="count">0</span> of __TOTAL__ judged <button id="copy" type="button">Copy</button></div>
<textarea id="codes" rows="2" readonly></textarea></footer>
<script>
const CLIPS = __CLIPS__;
const W = __WIDTH__, H = __HEIGHT__, RING = __RING__;
const paths = document.getElementById('paths');
function position(s, t) {
  const e = t - s.onset_s, ts = s.times_s;
  if (e < 0 || e > ts[ts.length - 1]) return null;
  for (let i = 1; i < ts.length; i++) if (e <= ts[i]) {
    const u = (e - ts[i - 1]) / (ts[i] - ts[i - 1]), a = s.points[i - 1], b = s.points[i];
    return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];
  }
  return s.points[s.points.length - 1];
}
function draw(clip, t) {
  const v = clip.video, c = clip.canvas, ctx = c.getContext('2d');
  const w = v.clientWidth, h = v.clientHeight;
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = 2;
  if (paths.checked) {
    ctx.strokeStyle = 'rgba(255,255,255,0.55)';
    for (const s of clip.strokes) {
      ctx.beginPath();
      s.points.forEach((p, i) => i ? ctx.lineTo(p[0] * w, p[1] * h) : ctx.moveTo(p[0] * w, p[1] * h));
      ctx.stroke();
    }
  }
  ctx.strokeStyle = '#00e5ff';
  for (const s of clip.strokes) {
    const p = position(s, t);
    if (p) { ctx.beginPath(); ctx.arc(p[0] * w, p[1] * h, RING * w / W, 0, 2 * Math.PI); ctx.stroke(); }
  }
  clip.time.textContent = t.toFixed(3) + ' s';
}
function frame(clip, step) {
  const ft = clip.frame_times_s, t = clip.video.currentTime;
  let i = 0;
  while (i + 1 < ft.length && ft[i + 1] <= t + 1e-4) i++;
  i = Math.max(0, Math.min(ft.length - 1, i + step));
  clip.video.pause();
  clip.video.currentTime = ft[i] + 1e-3;
}
for (const clip of CLIPS) {
  clip.video = document.getElementById('v' + clip.code);
  clip.canvas = document.getElementById('c' + clip.code);
  clip.time = document.getElementById('t' + clip.code);
  const redraw = () => draw(clip, clip.video.currentTime);
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    const tick = (now, meta) => { draw(clip, meta.mediaTime); clip.video.requestVideoFrameCallback(tick); };
    clip.video.requestVideoFrameCallback(tick);
  } else {
    const tick = () => { redraw(); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
  for (const name of ['loadedmetadata', 'seeked', 'pause']) clip.video.addEventListener(name, redraw);
  window.addEventListener('resize', redraw);
  paths.addEventListener('change', redraw);
  document.getElementById('b' + clip.code).addEventListener('click', () => frame(clip, -1));
  document.getElementById('f' + clip.code).addEventListener('click', () => frame(clip, 1));
  for (const rate of [0.1, 0.25, 1]) document.getElementById('r' + clip.code + '_' + rate)
    .addEventListener('click', () => { clip.video.playbackRate = rate; clip.video.play(); });
}
const codes = document.getElementById('codes');
document.addEventListener('change', () => {
  const picked = [...document.querySelectorAll('input[type=radio]:checked')]
    .map(r => r.name.slice(1) + r.value).sort();
  codes.value = picked.join(' ');
  document.getElementById('count').textContent = picked.length;
});
document.getElementById('copy').addEventListener('click', () => {
  codes.select(); navigator.clipboard ? navigator.clipboard.writeText(codes.value) : document.execCommand('copy');
});
</script></body></html>
"""


def _section(row, meta) -> tuple[str, dict]:
    code = row["code"]
    strokes = [dict(family=s["family"], points=s["command_points"], onset_s=s["onset_s"],
                    times_s=[t*meta["clock_rate"]/1000 for t in s["times_ms"]]) for s in meta["strokes"]]
    touches = ", ".join(f'{html.escape(s["family"])} {s["times_ms"][-1]} ms' for s in meta["strokes"])
    src = f'../recordings/{row["recording"]}/run-{row["run"]:05d}/clip.mp4'
    choices = "".join(f'<label><input type="radio" name="c{code}" value="{k}"> {v.capitalize()}</label>'
                      for k, v in VERDICTS.items())
    rates = "".join(f'<button type="button" id="r{code}_{r}">{r}×</button>' for r in ("0.1", "0.25", "1"))
    section = (f'<section><div class="stage"><video id="v{code}" src="{src}" controls muted playsinline preload="auto">'
               f'</video><canvas id="c{code}"></canvas></div><div class="side"><h2>Clip {code}</h2>'
               f'<p class="muted">Recording {html.escape(row["recording"])}, run {row["run"]}. '
               f'{len(meta["strokes"])} touch{"es" if len(meta["strokes"]) != 1 else ""}, in order: {touches}.</p>'
               f'<p><button type="button" id="b{code}">◀ frame</button><button type="button" id="f{code}">frame ▶</button>'
               f'{rates} <span class="muted" id="t{code}">0.000 s</span></p>{choices}</div></section>')
    return section, dict(code=code, strokes=strokes, frame_times_s=meta["frame_times_s"])


def review_page(root, *, count: int = DEFAULT_COUNT, reselect: bool = False) -> Path:
    """Write review/index.html; the pick is kept so codes stay valid when the page is rebuilt."""
    root = Path(root)
    out = root/"review"
    out.mkdir(exist_ok=True)
    path = out/"selection.json"
    if path.exists() and not reselect:
        selection = load_selection(root)
    else:
        if (out/"verdicts.json").exists():
            raise ValueError("verdicts already recorded for the current pick; keep it")
        selection = select(root, count)
        save_json(path, selection, exclusive=False)
    sections, clips = [], []
    for row in selection["clips"]:
        meta = _read(root/"recordings"/row["recording"]/f'run-{row["run"]:05d}'/"meta.json")
        if meta["clip_sha256"] != row["clip_sha256"]:
            raise ValueError(f'clip {row["code"]} changed after it was picked')
        section, data = _section(row, meta)
        sections.append(section)
        clips.append(data)
    page = PAGE
    for token, value in (("__SECTIONS__", "\n".join(sections)), ("__TOTAL__", str(len(clips))),
                         ("__CLIPS__", json.dumps(clips).replace("</", "<\\/")), ("__WIDTH__", str(WIDTH)),
                         ("__HEIGHT__", str(HEIGHT)), ("__RING__", str(RING_PT))):
        page = page.replace(token, value)
    (out/"index.html").write_text(page)
    return out/"index.html"


def record_verdicts(root, codes: str, *, replace: bool = False) -> int:
    """Operator verdicts as `01K 02F 03U` (OK, Fault, Unsure); every token is checked before any is written."""
    root = Path(root)
    selection = load_selection(root)
    rows = {row["code"]: row for row in selection["clips"]}
    tokens = TOKEN.findall(codes)
    if not tokens or TOKEN.sub("", codes).replace(",", " ").split():
        raise ValueError('codes look like "01K 02F 03U" (K OK, F fault, U unsure)')
    path = root/"review"/"verdicts.json"
    recorded = _read(path) if path.exists() else dict(selection_sha256=selection["sha256"], verdicts={})
    if recorded["selection_sha256"] != selection["sha256"]:
        raise ValueError("recorded verdicts belong to another pick")
    checked, seen = [], set()
    for number, code in tokens:
        key = f"{int(number):02d}"
        if key not in rows:
            raise ValueError(f"clip {key} is not in the pick")
        if key in seen:
            raise ValueError(f"clip {key} is listed twice")
        seen.add(key)
        if key in recorded["verdicts"] and not replace:
            raise ValueError(f"clip {key} already has a verdict; pass --replace to change it")
        checked.append((key, code.upper()))
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for key, code in checked:
        row = rows[key]
        recorded["verdicts"][key] = dict(code=code, verdict=VERDICTS[code], recording=row["recording"],
                                         run=row["run"], clip_sha256=row["clip_sha256"], recorded_at=now)
    save_json(path, recorded, exclusive=False)
    return len(checked)


def summarize(root) -> dict:
    """Recordings, runs, admission, stroke mix and spot-check counts for one run directory."""
    root = Path(root)
    recordings, reasons, families, durations, touches = {}, Counter(), Counter(), [], Counter()
    totals, states, lags = Counter(), Counter(), []
    for execution_path in sorted(root.glob("recordings/*/execution.json")):
        folder = execution_path.parent
        planned, execution = _read(folder/"planned.json"), _read(execution_path)
        sent = [e for e in execution["events"] if e["kind"] == "stroke" and not e.get("skipped")]
        row = dict(runs_planned=len(planned["runs"]), runs_sent=len({e["run"] for e in sent}),
                   strokes_sent=len(sent), strokes_dropped=sum(bool(e.get("skipped")) for e in execution["events"]),
                   clips_admitted=0, error=execution["error"])
        if execution["error"]:
            row["state"] = "recording failed"
        elif not (folder/"admission.json").exists():
            row["state"] = "awaiting admission"
        else:
            admission = _read(folder/"admission.json")
            row["state"] = "admitted" if admission["accepted"] else "admission failed"
            row["clips_admitted"] = len(admission.get("admitted", []))
            row["error"] = admission.get("error")
            reasons.update(admission.get("rejected", {}).values())
            # Visible minus estimated stroke start (diagnostic; positive means the trail shows later).
            lags += (admission.get("calibration") or {}).get("stroke_visible_lags_s", {}).values()
        recordings[folder.name] = row
        totals.update({k: row[k] for k in ("runs_planned", "runs_sent", "strokes_sent", "strokes_dropped",
                                           "clips_admitted")})
        totals["recordings"] += 1
        states[row["state"]] += 1
    for clip in admitted_clips(root):
        touches[len(clip["meta"]["strokes"])] += 1
        for stroke in clip["meta"]["strokes"]:
            families[stroke["family"]] += 1
            durations.append(stroke["times_ms"][-1])
    spot = Counter()
    if (root/"review"/"verdicts.json").exists():
        spot.update(v["code"] for v in _read(root/"review"/"verdicts.json")["verdicts"].values())
    picked = len(load_selection(root)["clips"]) if (root/"review"/"selection.json").exists() else 0
    sent = totals["runs_sent"]
    return dict(experiment=EXPERIMENT, totals=dict(totals), recordings_by_state=dict(states),
                admitted_share_of_sent_runs=totals["clips_admitted"]/sent if sent else None,
                rejection_reasons=dict(reasons.most_common()), stroke_families=dict(families),
                touches_per_clip={str(k): v for k, v in sorted(touches.items())},
                stroke_ms=dict(min=min(durations), median=statistics.median(durations), max=max(durations),
                               share_under_200=sum(d < 200 for d in durations)/len(durations)) if durations else None,
                spot_check=dict(picked=picked, ok=spot["K"], fault=spot["F"], unsure=spot["U"],
                                unjudged=picked-sum(spot.values())),
                stroke_visible_lag_s=dict(measured=len(lags), median=statistics.median(lags),
                                          p10=float(np.percentile(lags, 10)), p90=float(np.percentile(lags, 90)))
                if lags else None,
                recordings=recordings)


def summary_markdown(summary: dict) -> str:
    t, spot = summary["totals"], summary["spot_check"]
    share = summary["admitted_share_of_sent_runs"]
    states = ", ".join(f"{state} {n}" for state, n in sorted(summary["recordings_by_state"].items())) or "none"
    lines = [f"# {summary['experiment']}", "",
             f"- Recordings: {t.get('recordings', 0)} ({states}).",
             f"- Clips admitted: {t.get('clips_admitted', 0)}"
             f"{'' if share is None else f' ({share:.0%} of runs sent)'}.",
             f"- Runs planned {t.get('runs_planned', 0)}, sent {t.get('runs_sent', 0)}; strokes sent "
             f"{t.get('strokes_sent', 0)}, dropped at run time {t.get('strokes_dropped', 0)}.",
             f"- Spot check: {spot['picked']} picked, {spot['ok']} OK, {spot['fault']} fault, "
             f"{spot['unsure']} unsure, {spot['unjudged']} unjudged."]
    if summary["stroke_ms"]:
        s = summary["stroke_ms"]
        lines.append(f"- Stroke durations: {s['min']}–{s['max']} ms, median {s['median']:g} ms, "
                     f"{s['share_under_200']:.0%} under 200 ms.")
    if summary.get("stroke_visible_lag_s"):
        g = summary["stroke_visible_lag_s"]
        lines.append(f"- Trail shows {g['median']*1000:+.0f} ms after the estimated stroke start (median of "
                     f"{g['measured']}; 10–90%: {g['p10']*1000:+.0f} to {g['p90']*1000:+.0f} ms). Labels unchanged.")
    if summary["rejection_reasons"]:
        lines += ["", "| Rejection reason | Runs |", "|---|---:|"]
        lines += [f"| {reason} | {n} |" for reason, n in summary["rejection_reasons"].items()]
    return "\n".join(lines) + "\n"
