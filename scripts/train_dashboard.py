"""Canonical training dashboard: phone-screen preview + training log + Mode A heartbeat.

Serves a single page — the only rig dashboard — with a per-device screen
preview and a distilled per-device log built from the newest run JSONL
(current trick, eval counts, throughput, rolling land rate, latest landed
tricks), plus a top heartbeat bar for Mode A (CMA-ES) runs sourced from
``logs/status.json`` (written by ``trueskate_ai.monitoring.status.StatusTracker``).
Absorbs what used to be the separate ``status_server.py`` — that script is
retired; two dashboards on two ports was one too many.

The screen preview is NOT the old view_device.py HLS stream — that path is the
AVFoundation/CoreMediaIO "DAL" screen-mirror capture, which is wedged at the OS
level with no third-party (headless) fix (see memory
ios-dal-screen-capture-wedge; re-confirmed dead 2026-07-13). Instead this
serves the newest clip the Mode B XCTest collector has already aligned to disk
(`data/<corpus>/<device>_*/.../sample_NNNNNN/frames.mp4`, or `frame_NNN.png`
for corpora aligned before the video packing) — real gameplay footage, zero
extra device/WDA traffic, but refreshed only on the collector's segment cadence
(~60-75s), not true live video.

Meant to run continuously (launchd, RunAtLoad+KeepAlive) rather than be
spawned per training run — Mode A's heartbeat bar just shows "idle" when
``logs/status.json`` is absent or stale, which is the common case since Mode B
collection is the rig's current default activity.

Usage:
    python scripts/train_dashboard.py [--port 8400] [--host 0.0.0.0]
        [--log-root logs/overnight] [--corpus-root data]
        [--status-path logs/status.json]

Open http://127.0.0.1:8400/ (or the tailnet IP from another device).
"""
import argparse
import functools
import itertools
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

DEVICES = ["iPhone_XR", "iPhone_XR2"]


def _park_tag(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _sessions(corpus_root: Path, device: str) -> list[Path]:
    """`device` session dirs newest-first, whether `corpus_root` IS one corpus
    (data/sls_xctest) or the parent of several (data/).

    Each collection experiment gets its own `--out-dir`, so pinning a single
    corpus goes blind the moment the rig moves to the next experiment — which
    is exactly how the preview went permanently STALE while collection was
    healthy. Session dirs are named `<device>_YYYYMMDD_HHMMSS`, so sorting by
    name is sorting by time, and it stays correct across sibling corpora.
    """
    candidates = itertools.chain(
            corpus_root.glob(f"{device}_*"),
            corpus_root.glob(f"*/{device}_*"),
            # A device-provenanced staged corpus adds one directory between the
            # corpus root and its session: <corpus>/<device_bucket>/<session>.
            # Without this level, the dashboard falls back to an old sibling
            # corpus and reports it stale while a current staged run is healthy.
            corpus_root.glob(f"*/*/{device}_*"),
    )
    # Device buckets such as ``iPhone_XR_sls_park`` share the device prefix but
    # are not sessions. Collector sessions have the timestamped suffix below.
    session_name = re.compile(rf"^{re.escape(device)}_\d{{8}}_\d{{6}}$")
    return sorted(
        (path for path in candidates if path.is_dir() and session_name.match(path.name)),
        key=lambda p: p.name, reverse=True,
    )


def _sample_media(sample_dir: Path) -> tuple[Path, str, str] | None:
    """(path, media, content-type) for a sample's footage, or None if it has none.

    Prefers the packed `frames.mp4` the aligner now writes (`--align-video` /
    `--direct-video`, which delete the PNGs), falling back to a `frame_NNN.png`
    sequence for corpora aligned before that change.
    """
    clip = sample_dir / "frames.mp4"
    if clip.exists():
        return clip, "video", "video/mp4"
    frames = sorted(sample_dir.glob("frame_*.png"))
    if frames:
        return frames[-1], "image", "image/png"
    return None


def _latest_preview_clip(corpus_root: Path, device: str) -> dict | None:
    """Newest aligned collector footage for `device` — stand-in for live video.

    Walks sessions newest-first, then that session's most-recently-aligned
    segment marker, then that manifest's gestures newest-first, to find the
    latest sample dir that has footage and isn't flagged `.menu` (replay/menu
    contamination — see flag_menu_samples.py; a flagged or footage-less gesture
    is skipped in favor of an earlier one in the same manifest before falling
    back to an older marker/session) — no directory-tree glob over the whole
    (potentially huge) corpus. Returns the media path plus the gesture's own
    call-end timestamp (rig clock) and park, so the UI can show how stale the
    footage is without trusting the viewer's own clock.
    """
    for session in _sessions(corpus_root, device):
        markers = sorted(session.glob("segment_*.aligned"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        for marker in markers:
            manifest_path = session / (marker.stem + ".json")
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for gesture in reversed(manifest.get("gestures", [])):
                park = gesture.get("park") or "?"
                sample_dir = session / _park_tag(park) / f"sample_{gesture['gesture_index']:06d}"
                if (sample_dir / ".menu").exists():
                    continue
                media = _sample_media(sample_dir)
                if media:
                    path, kind, ctype = media
                    return {
                        "path": path,
                        "media": kind,
                        "ctype": ctype,
                        "captured_at": gesture.get("t_call_end_epoch_s"),
                        "park": park,
                        "session": session.name,
                        "corpus": session.parent.name,
                    }
    return None


def _newest_jsonl(log_root: Path, device: str) -> Path | None:
    runs = sorted(log_root.glob(f"{device}/*/runs/cmaes_run_*/cmaes_run_*.jsonl"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


# Bounds on the collection-stats walk: the corpus holds thousands of sessions,
# and /data is polled every 5s, so the walk stops as soon as it has seen a
# session older than the throughput window rather than scanning everything.
_COLLECT_WINDOW_S = 3600.0
_COLLECT_MAX_SESSIONS = 40
# Only the segment manifest itself — a session also holds sidecars named off the
# same stem (segment_00000.calibration_rejected.json), which carry no gestures
# and would sort ahead of the real manifest.
_MANIFEST_RE = re.compile(r"^segment_\d+\.json$")


def _collection_status(corpus_root: Path, device: str) -> dict:
    """Mode B throughput for `device`, read from the collector's own manifests.

    The panel this feeds is otherwise dead whenever no CMA-ES run exists — which
    is the rig's usual state, since frame->gesture collection is the current
    default activity. Timestamps are the gestures' `t_call_end_epoch_s` (rig
    clock, same source as the preview's age) so both halves of the page agree.
    """
    now = time.time()
    cutoff = now - _COLLECT_WINDOW_S
    samples_1h = 0
    latest: float | None = None
    session_samples = 0
    current: dict = {}
    for i, session in enumerate(_sessions(corpus_root, device)):
        if i >= _COLLECT_MAX_SESSIONS:
            break
        newest_here: float | None = None
        for manifest_path in sorted(p for p in session.glob("segment_*.json")
                                    if _MANIFEST_RE.match(p.name)):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not current:
                current = {"session": session.name, "corpus": session.parent.name,
                           "park": manifest.get("park"), "use_spin": manifest.get("use_spin")}
            for gesture in manifest.get("gestures", []):
                ts = gesture.get("t_call_end_epoch_s")
                if ts is None:
                    continue
                # Counted against `current`, not session 0: the newest session dir
                # is normally the in-flight one, still empty until its segment
                # finishes recording and aligning.
                if session.name == current["session"]:
                    session_samples += 1
                if ts >= cutoff:
                    samples_1h += 1
                if newest_here is None or ts > newest_here:
                    newest_here = ts
        if newest_here is not None:
            latest = newest_here if latest is None else max(latest, newest_here)
            if newest_here < cutoff:
                break  # sessions are time-ordered: everything older is out of window
    if not current:
        return {"device": device, "mode": "collect", "trick": "—", "evals": 0,
                "note": "no collection corpus found"}
    return {
        "device": device,
        "mode": "collect",
        **current,
        "samples_1h": samples_1h,
        "session_samples": session_samples,
        "last_sample_ts": time.strftime("%H:%M:%S", time.localtime(latest)) if latest else "—",
        "last_sample_age_s": round(now - latest) if latest else None,
    }


# Per-device tail-read cache: {device: {"path", "pos", "target", "rows"}}. A run
# JSONL only ever grows, and /data is polled every 5s for the run's whole
# multi-hour duration, so re-reading and re-parsing the entire file from byte 0
# on every poll gets slower as the run goes on. Instead each poll seeks to
# where the last poll stopped and parses only the newly appended lines.
_STATUS_CACHE: dict[str, dict] = {}
_STATUS_LOCK = threading.Lock()


def _tail_new_rows(cached: dict) -> None:
    """Parse whatever complete (newline-terminated) lines were appended to
    cached["path"] since cached["pos"], updating cached["rows"]/["target"] and
    advancing cached["pos"]. A torn trailing line (writer mid-flush) is left
    unconsumed so it's picked up whole on a later poll instead of being lost."""
    try:
        with cached["path"].open("rb") as f:
            f.seek(cached["pos"])
            chunk = f.read()
    except OSError:
        return
    last_nl = chunk.rfind(b"\n")
    if last_nl == -1:
        return
    for raw in chunk[:last_nl].split(b"\n"):
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "run_config":
            cached["target"] = r.get("target")
        elif "trick_name" in r or "eval_num" in r:
            cached["rows"].append(r)
    cached["pos"] += last_nl + 1


def _visible_rows(rows: list[dict], log_delay_s: float) -> list[dict]:
    """Hide rows newer than the stream latency so the log never references a
    trick the (HLS-delayed) video hasn't shown yet. Rows are append-ordered by
    time, so "too new" is always a suffix — trim from the tail, not a full scan.
    Always returns a fresh list: the caller reads it after releasing the cache
    lock, and `rows` here is the live cached list another thread may append to."""
    if log_delay_s <= 0:
        return list(rows)
    cutoff = time.time() - log_delay_s
    end = len(rows)
    while end > 0:
        try:
            ts = time.mktime(time.strptime(rows[end - 1]["timestamp"][:19], "%Y-%m-%dT%H:%M:%S"))
        except (KeyError, ValueError):
            break  # unparseable -> treat as visible, matching the original scan's behavior
        if ts <= cutoff:
            break
        end -= 1
    return rows[:end]


def _device_status(log_root: Path, device: str, corpus_root: Path,
                   log_delay_s: float = 0.0) -> dict:
    j = _newest_jsonl(log_root, device)
    if j is None:
        # No Mode A run — report what Mode B collection is doing instead.
        return _collection_status(corpus_root, device)
    with _STATUS_LOCK:
        cached = _STATUS_CACHE.get(device)
        if cached is None or cached["path"] != j:
            cached = {"path": j, "pos": 0, "target": None, "rows": []}
            _STATUS_CACHE[device] = cached
        _tail_new_rows(cached)
        target = cached["target"]
        rows = _visible_rows(cached["rows"], log_delay_s)
    evals = len(rows)
    lands = [r for r in rows if r.get("trick_status") == "landed"]
    target_lands = [
        r for r in lands
        if target and target in [c.strip().upper() for c in (r.get("trick_name") or "").split(" + ")]
    ]
    window = rows[-50:]
    window_lands = sum(
        1 for r in window
        if r.get("trick_status") == "landed" and target
        and target in [c.strip().upper() for c in (r.get("trick_name") or "").split(" + ")]
    )
    # evals/hr over the last 100 rows
    rate = None
    if len(rows) >= 2:
        sample = rows[-100:]
        try:
            t0 = time.mktime(time.strptime(sample[0]["timestamp"][:19], "%Y-%m-%dT%H:%M:%S"))
            t1 = time.mktime(time.strptime(sample[-1]["timestamp"][:19], "%Y-%m-%dT%H:%M:%S"))
            if t1 > t0:
                rate = round((len(sample) - 1) * 3600 / (t1 - t0))
        except (KeyError, ValueError):
            pass
    recent = [
        {"eval": r.get("eval_num"), "trick": r.get("trick_name"),
         "reward": r.get("reward"), "ts": (r.get("timestamp") or "")[11:19]}
        for r in lands[-10:]
    ]
    last_ts = (rows[-1].get("timestamp") or "")[11:19] if rows else "—"
    return {
        "device": device,
        "run": j.parent.name,
        "trick": target or "?",
        "generation": rows[-1].get("generation") if rows else None,
        "evals": evals,
        "evals_per_hr": rate,
        "target_lands": len(target_lands),
        "land_rate_50": round(window_lands / max(1, len(window)), 3),
        "last_eval_ts": last_ts,
        "recent_lands": list(reversed(recent)),
    }


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>TrueSkate training</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
html,body{margin:0;background:#0d1117;color:#c9d1d9;font:13px/1.5 ui-monospace,Menlo,monospace;height:100%}
.heartbeat{display:flex;align-items:center;gap:.6rem;padding:.5rem .8rem;background:#161b22;
           border-bottom:1px solid #30363d;font-size:.8rem;color:#8b949e}
.heartbeat b{color:#e6edf3}
.wrap{display:grid;grid-template-columns:280px 280px 1fr;gap:10px;padding:10px;height:calc(100vh - 40px)}
.cam{display:flex;flex-direction:column;min-height:0;border:1px solid #30363d;border-radius:8px;
     overflow:hidden;background:#0d1117;align-self:start}
.cam-hdr{display:flex;justify-content:space-between;align-items:center;padding:6px 10px;
         background:#161b22;border-bottom:1px solid #30363d;flex:0 0 auto}
.cam-hdr .cam-dev{color:#58a6ff;font-weight:700}
.screen{width:100%;flex:0 0 auto;aspect-ratio:414/896;object-fit:contain;background:#000}
.screen[hidden]{display:none}
.cam-footer{padding:8px 10px;font-size:11px;color:#8b949e;line-height:1.6}
.cam-footer b{color:#e6edf3}
.badge{font-size:10px;padding:2px 8px;border-radius:8px;font-weight:700;letter-spacing:.02em}
.badge.live{background:#1a7f37;color:#fff}
.badge.delayed{background:#9e6a03;color:#fff}
.badge.stale{background:#b62324;color:#fff}
.badge.none{background:#30363d;color:#8b949e}
.log{overflow-y:auto;border:1px solid #30363d;border-radius:8px;padding:12px}
h2{font-size:13px;margin:4px 0;color:#58a6ff}
.dev{margin-bottom:18px}
.stat{color:#8b949e}
.stat b{color:#e6edf3}
.land{color:#3fb950}
.hdr{position:sticky;top:0;background:#0d1117;padding-bottom:4px}
table{border-collapse:collapse;width:100%}
td{padding:1px 8px 1px 0;white-space:nowrap}
@media (max-width:900px){.wrap{grid-template-columns:1fr 1fr;grid-template-rows:300px 1fr}.log{grid-column:1/3}}
</style></head>
<body>
<div class="heartbeat"><span class="badge none" id="hb-pill">—</span><span id="hb-text">loading heartbeat…</span></div>
<div class="wrap">
<div class="cam">
  <div class="cam-hdr"><span class="cam-dev">XR1</span><span class="badge none" id="badge-iPhone_XR">—</span></div>
  <img class="screen" id="img-iPhone_XR" alt="iPhone_XR preview" hidden>
  <video class="screen" id="vid-iPhone_XR" autoplay loop muted playsinline hidden></video>
  <div class="cam-footer" id="foot-iPhone_XR">loading…</div>
</div>
<div class="cam">
  <div class="cam-hdr"><span class="cam-dev">XR2</span><span class="badge none" id="badge-iPhone_XR2">—</span></div>
  <img class="screen" id="img-iPhone_XR2" alt="iPhone_XR2 preview" hidden>
  <video class="screen" id="vid-iPhone_XR2" autoplay loop muted playsinline hidden></video>
  <div class="cam-footer" id="foot-iPhone_XR2">loading…</div>
</div>
<div class="log" id="log">loading…</div>
</div>
<script>
// Mode B (frame->gesture collection) has no run JSONL and no rewards, so it
// gets its own block rather than an all-dashes version of the CMA-ES one.
function collectBlock(s){
  if (s.note) return `<div class="dev"><h2 class="hdr">${s.device}</h2>
    <div class="stat">${s.note}</div></div>`;
  return `
      <div class="dev">
        <h2 class="hdr">${s.device} — collecting</h2>
        <div class="stat">corpus <b>${s.corpus||'—'}</b> · park <b>${s.park||'?'}</b> ·
          spin <b>${s.use_spin ? 'yes' : 'no'}</b></div>
        <div class="stat">session <b>${s.session||'—'}</b> (<b>${s.session_samples??0}</b> samples) ·
          last hour <b>${s.samples_1h??0}</b> ·
          last sample <b>${s.last_sample_ts}</b> (${fmtAge(s.last_sample_age_s)} ago)</div>
      </div>`;
}
async function tick(){
  try{
    const d = await (await fetch('/data')).json();
    document.getElementById('log').innerHTML = d.map(s => s.mode === 'collect' ? collectBlock(s) : `
      <div class="dev">
        <h2 class="hdr">${s.device} — ${s.trick}</h2>
        <div class="stat">run <b>${s.run||'—'}</b> · gen <b>${s.generation??'—'}</b> ·
          evals <b>${s.evals}</b> (${s.evals_per_hr??'?'}/hr) ·
          target lands <b>${s.target_lands??0}</b> ·
          land rate (last 50) <b>${((s.land_rate_50||0)*100).toFixed(0)}%</b> ·
          last eval <b>${s.last_eval_ts}</b></div>
        <table>${(s.recent_lands||[]).map(r =>
          `<tr><td class="stat">${r.ts}</td><td class="stat">#${r.eval}</td>
           <td class="land">${r.trick}</td><td class="stat">r=${r.reward}</td></tr>`).join('')}
        </table>
      </div>`).join('');
  }catch(e){ document.getElementById('log').textContent = 'fetch failed: '+e; }
}
// Age thresholds (seconds): the collector lands a new frame every ~60-75s, so
// "live" gives one missed cycle of slack before calling it delayed/stale.
const LIVE_MAX_S = 150, DELAYED_MAX_S = 900;
function fmtAge(s){
  if (s == null) return '?';
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${Math.round(s / 3600)}h`;
}
function badgeInfo(s){
  if (s == null) return ['none', '—'];
  if (s <= LIVE_MAX_S) return ['live', 'LIVE-ISH'];
  if (s <= DELAYED_MAX_S) return ['delayed', 'DELAYED'];
  return ['stale', 'STALE'];
}
async function refreshCams(){
  const t = Date.now();
  for (const dev of ['iPhone_XR', 'iPhone_XR2']) {
    const img = document.getElementById('img-' + dev);
    const vid = document.getElementById('vid-' + dev);
    const badge = document.getElementById('badge-' + dev);
    const foot = document.getElementById('foot-' + dev);
    try {
      const r = await fetch(`/preview/${dev}?t=${t}`);
      if (!r.ok) throw 0;
      const ageS = r.headers.get('X-Age-S');
      const capturedAt = r.headers.get('X-Captured-At');
      const park = r.headers.get('X-Park');
      const corpus = r.headers.get('X-Corpus');
      // Samples are packed as h264 clips (frames.mp4) since the aligner started
      // deleting the PNGs; older corpora still serve a single still frame.
      const el = r.headers.get('X-Media') === 'video' ? vid : img;
      const other = el === vid ? img : vid;
      const blob = await r.blob();
      const old = el.src;
      el.src = URL.createObjectURL(blob);
      if (old && old.startsWith('blob:')) URL.revokeObjectURL(old);
      el.hidden = false; other.hidden = true;
      const ageNum = ageS !== null ? parseFloat(ageS) : null;
      const [cls, label] = badgeInfo(ageNum);
      badge.textContent = label;
      badge.className = 'badge ' + cls;
      foot.innerHTML = `captured <b>${capturedAt || '?'}</b> (${fmtAge(ageNum)} ago)<br>` +
        `park: <b>${park || '?'}</b> · corpus: <b>${corpus || '?'}</b>`;
    } catch (e) {
      badge.textContent = '—'; badge.className = 'badge none';
      foot.textContent = 'no footage yet';
    }
  }
}
async function tickHeartbeat(){
  const pill = document.getElementById('hb-pill'), el = document.getElementById('hb-text');
  let s;
  try { s = await (await fetch('/status.json', {cache:'no-store'})).json(); }
  catch(e){ pill.textContent='—'; pill.className='badge none'; el.textContent='heartbeat unreachable'; return; }
  if (s.state === 'no-status-yet') {
    pill.textContent = 'IDLE'; pill.className = 'badge none';
    el.textContent = 'Mode A (CMA-ES): no training run active';
    return;
  }
  const ageS = (Date.now() - Date.parse(s.updated_at)) / 1000;
  const stale = ageS > 180;
  pill.textContent = stale ? `STALE ${Math.round(ageS)}s` : 'LIVE';
  pill.className = 'badge ' + (stale ? 'stale' : 'live');
  const deadTxt = (s.dead && s.dead.length) ? ` · dead: ${s.dead.join(', ')}` : '';
  el.innerHTML = `Mode A: <b>${s.target}</b> · run ${s.run_id} · gen ${s.generation} · ` +
    `evals ${s.total_evals}/${s.max_evals} · land rate ${((s.land_rate||0)*100).toFixed(1)}% · ` +
    `best ${s.best_reward} ${s.best_trick||''}${deadTxt}`;
}
tick(); setInterval(tick, 5000);
refreshCams(); setInterval(refreshCams, 10000);
tickHeartbeat(); setInterval(tickHeartbeat, 5000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, log_root: Path, corpus_root: Path, log_delay_s: float,
                 status_path: Path, **kwargs):
        self.log_root = log_root
        self.corpus_root = corpus_root
        self.log_delay_s = log_delay_s
        self.status_path = status_path
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/data":
            body = json.dumps([
                _device_status(self.log_root, d, self.corpus_root, self.log_delay_s)
                for d in DEVICES
            ]).encode()
            ctype = "application/json"
        elif self.path.startswith("/status.json"):
            # Mode A (CMA-ES) heartbeat, written by StatusTracker. Absent/stale
            # is the normal state whenever Mode B collection is what's running.
            if self.status_path.exists():
                body = self.status_path.read_bytes()
            else:
                body = json.dumps({"state": "no-status-yet"}).encode()
            ctype = "application/json"
        elif self.path.startswith("/preview/"):
            device = self.path[len("/preview/"):].split("?", 1)[0]
            device = device.rsplit(".", 1)[0] if "." in device else device
            info = _latest_preview_clip(self.corpus_root, device) if device in DEVICES else None
            if info is None:
                self._send(404, b"no preview footage yet", "text/plain")
                return
            body = info["path"].read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", info["ctype"])
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            captured_at = info.get("captured_at")
            if captured_at is not None:
                # Age computed with THIS process's clock (the rig's own), never the
                # viewer's — the rig clock has been known to drift from other
                # references, so cross-machine comparisons are unreliable.
                self.send_header("X-Age-S", str(round(time.time() - captured_at)))
                self.send_header("X-Captured-At",
                                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(captured_at)))
            self.send_header("X-Park", info.get("park") or "?")
            self.send_header("X-Media", info["media"])
            self.send_header("X-Corpus", info.get("corpus") or "?")
            self.send_header("X-Session", info.get("session") or "?")
            self.end_headers()
            self.wfile.write(body)
            return
        else:
            body = _PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined training dashboard.")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--log-root", type=Path, default=_REPO_ROOT / "logs" / "overnight")
    parser.add_argument("--corpus-root", type=Path, default=_REPO_ROOT / "data",
                        help="XCTest collector output — source of the screen-preview footage. "
                             "Either one corpus (data/sls_xctest) or their parent (data/), in "
                             "which case the newest session across all corpora wins.")
    parser.add_argument("--log-delay", type=float, default=8.0,
                        help="Seconds to lag the log so it doesn't outrun the preview frames.")
    parser.add_argument("--status-path", type=Path, default=_REPO_ROOT / "logs" / "status.json",
                        help="Mode A heartbeat file written by StatusTracker (default: logs/status.json).")
    args = parser.parse_args()

    handler = functools.partial(_Handler, log_root=args.log_root, corpus_root=args.corpus_root,
                                log_delay_s=args.log_delay, status_path=args.status_path.resolve())
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as e:
        sys.exit(f"Cannot bind {args.host}:{args.port} ({e.strerror}) — already running?")
    print(f"Dashboard: http://127.0.0.1:{args.port}/  (tailnet: http://<tailscale-ip>:{args.port}/)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
