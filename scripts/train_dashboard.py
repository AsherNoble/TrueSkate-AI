"""Combined training dashboard: phone-screen preview + distilled training log.

Serves a single page with a per-device screen preview and a distilled
per-device log built from the newest run JSONL: current trick, eval counts,
throughput, rolling land rate, and the latest landed tricks.

The screen preview is NOT the old view_device.py HLS stream — that path is the
AVFoundation/CoreMediaIO "DAL" screen-mirror capture, which is wedged at the OS
level with no third-party (headless) fix (see memory
ios-dal-screen-capture-wedge; re-confirmed dead 2026-07-13). Instead this
serves the newest frame the Mode B SLS/XCTest collector has already aligned to
disk (`data/sls_xctest/<device>_*/.../sample_NNNNNN/frame_NNN.png`) — real
gameplay frames, zero extra device/WDA traffic, but refreshed only on the
collector's segment cadence (~60-75s), not true live video.

Usage:
    python scripts/train_dashboard.py [--port 8400] [--host 0.0.0.0]
        [--log-root logs/overnight] [--corpus-root data/sls_xctest]

Open http://127.0.0.1:8400/ (or the tailnet IP from another device).
"""
import argparse
import functools
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

DEVICES = ["iPhone_XR", "iPhone_XR2"]


def _park_tag(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _latest_preview_frame(corpus_root: Path, device: str) -> Path | None:
    """Newest aligned SLS-collector frame for `device` — stand-in for live video.

    Walks sessions newest-first, then that session's most-recently-aligned
    segment marker, and reads its manifest for the last gesture's (park,
    gesture_index) to go straight to that sample dir — no directory-tree glob
    over the whole (potentially huge) corpus.
    """
    sessions = sorted(corpus_root.glob(f"{device}_*"), key=lambda p: p.name, reverse=True)
    for session in sessions:
        markers = sorted(session.glob("segment_*.aligned"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        for marker in markers:
            manifest_path = session / (marker.stem + ".json")
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            gestures = manifest.get("gestures", [])
            if not gestures:
                continue
            last = gestures[-1]
            sample_dir = session / _park_tag(last.get("park", "park")) / f"sample_{last['gesture_index']:06d}"
            frames = sorted(sample_dir.glob("frame_*.png"))
            if frames:
                return frames[-1]
    return None


def _newest_jsonl(log_root: Path, device: str) -> Path | None:
    runs = sorted(log_root.glob(f"{device}/*/runs/cmaes_run_*/cmaes_run_*.jsonl"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _device_status(log_root: Path, device: str, log_delay_s: float = 0.0) -> dict:
    j = _newest_jsonl(log_root, device)
    if j is None:
        return {"device": device, "trick": "—", "evals": 0, "note": "no run found"}
    target = None
    rows = []
    # Hide rows newer than the stream latency so the log never references a
    # trick the (HLS-delayed) video hasn't shown yet.
    cutoff = time.time() - log_delay_s
    with j.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "run_config":
                target = r.get("target")
            elif "trick_name" in r or "eval_num" in r:
                if log_delay_s > 0:
                    try:
                        ts = time.mktime(time.strptime(r["timestamp"][:19], "%Y-%m-%dT%H:%M:%S"))
                        if ts > cutoff:
                            continue
                    except (KeyError, ValueError):
                        pass
                rows.append(r)
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
.wrap{display:grid;grid-template-columns:280px 280px 1fr;gap:10px;padding:10px;height:calc(100vh - 20px)}
.cam{display:flex;flex-direction:column;min-height:0}
img.screen{width:100%;flex:1;min-height:0;border:1px solid #30363d;border-radius:8px;background:#000;object-fit:contain}
.lbl{text-align:center;color:#58a6ff;font-weight:600;padding:4px 0 0}
.lbl .stat{font-weight:400}
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
<body><div class="wrap">
<div class="cam"><img class="screen" id="cam-iPhone_XR" alt="iPhone_XR preview">
  <div class="lbl">XR1 <span class="stat">(SLS corpus, ~60-75s lag)</span></div></div>
<div class="cam"><img class="screen" id="cam-iPhone_XR2" alt="iPhone_XR2 preview">
  <div class="lbl">XR2 <span class="stat">(SLS corpus, ~60-75s lag)</span></div></div>
<div class="log" id="log">loading…</div>
</div>
<script>
async function tick(){
  try{
    const d = await (await fetch('/data')).json();
    document.getElementById('log').innerHTML = d.map(s => `
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
function refreshCams(){
  const t = Date.now();
  for (const dev of ['iPhone_XR', 'iPhone_XR2']) {
    const img = document.getElementById('cam-' + dev);
    fetch(`/preview/${dev}.jpg?t=${t}`).then(r => { if (r.ok) return r.blob(); throw 0; })
      .then(b => {
        const old = img.src;
        img.src = URL.createObjectURL(b);
        if (old.startsWith('blob:')) URL.revokeObjectURL(old);
      })
      .catch(() => {});
  }
}
tick(); setInterval(tick, 5000);
refreshCams(); setInterval(refreshCams, 10000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, log_root: Path, corpus_root: Path, log_delay_s: float, **kwargs):
        self.log_root = log_root
        self.corpus_root = corpus_root
        self.log_delay_s = log_delay_s
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/data":
            body = json.dumps([
                _device_status(self.log_root, d, self.log_delay_s) for d in DEVICES
            ]).encode()
            ctype = "application/json"
        elif self.path.startswith("/preview/"):
            device = self.path[len("/preview/"):].split("?", 1)[0]
            if device.endswith(".jpg"):
                device = device[:-len(".jpg")]
            frame = _latest_preview_frame(self.corpus_root, device) if device in DEVICES else None
            if frame is None:
                self._send(404, b"no preview frame yet", "text/plain")
                return
            self._send(200, frame.read_bytes(), "image/png")
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
    parser.add_argument("--corpus-root", type=Path, default=_REPO_ROOT / "data" / "sls_xctest",
                        help="SLS/XCTest collector output — source of the screen-preview frames.")
    parser.add_argument("--log-delay", type=float, default=8.0,
                        help="Seconds to lag the log so it doesn't outrun the preview frames.")
    args = parser.parse_args()

    handler = functools.partial(_Handler, log_root=args.log_root, corpus_root=args.corpus_root,
                                log_delay_s=args.log_delay)
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as e:
        sys.exit(f"Cannot bind {args.host}:{args.port} ({e.strerror}) — already running?")
    print(f"Dashboard: http://127.0.0.1:{args.port}/  (tailnet: http://<tailscale-ip>:{args.port}/)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
