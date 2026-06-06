"""Tiny status web server for the 24h training rig.

Serves the heartbeat written by ``trueskate_ai.monitoring.status.StatusTracker``
as a self-refreshing HTML dashboard plus a raw JSON endpoint. Designed to sit
behind Tailscale:

    python scripts/status_server.py            # binds 127.0.0.1:8200
    tailscale serve --bg 8200                  # expose over your tailnet (HTTPS)

then open  https://training-server.<tailnet>.ts.net/  from any device.

Endpoints:
    GET /             → HTML dashboard (polls /status.json every few seconds)
    GET /status.json  → raw status file (200), or {"state":"no-status-yet"} (200)
    GET /healthz      → "ok" (200) for uptime checks

Stdlib only — no Flask/dependency, so it runs anywhere the repo does.
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TrueSkate rig</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, system-ui, sans-serif; background:#0d1117;
         color:#e6edf3; margin:0; padding:1.2rem; }
  h1 { font-size:1.1rem; margin:0 0 .2rem; }
  .sub { color:#8b949e; font-size:.8rem; margin-bottom:1rem; }
  .grid { display:flex; flex-wrap:wrap; gap:.6rem; margin-bottom:1rem; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:.5rem;
          padding:.6rem .8rem; min-width:120px; }
  .card .k { color:#8b949e; font-size:.7rem; text-transform:uppercase; }
  .card .v { font-size:1.3rem; font-weight:600; }
  table { width:100%; border-collapse:collapse; font-size:.85rem; }
  th,td { text-align:left; padding:.45rem .6rem; border-bottom:1px solid #21262d; }
  th { color:#8b949e; font-weight:500; }
  .pill { padding:.1rem .5rem; border-radius:1rem; font-size:.72rem; font-weight:600; }
  .alive { background:#1a7f37; color:#fff; }
  .dead  { background:#b62324; color:#fff; }
  .stale { background:#9e6a03; color:#fff; }
  .ok    { background:#1a7f37; color:#fff; }
  #note { margin-top:1rem; color:#8b949e; font-size:.8rem; }
</style>
</head>
<body>
  <h1>TrueSkate training rig <span id="freshness" class="pill"></span></h1>
  <div class="sub" id="header">loading…</div>
  <div class="grid" id="cards"></div>
  <table>
    <thead><tr><th>Device</th><th>State</th><th>Evals</th><th>Lands</th><th>Last trick</th></tr></thead>
    <tbody id="devices"></tbody>
  </table>
  <div id="note"></div>
<script>
function card(k, v) { return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`; }
function fmtTrick(t) {
  if (!t) return "—";
  const name = t.trick || "—";
  return `${name} <span class="pill ${t.landed?'alive':'dead'}">${t.reward}</span>`;
}
async function tick() {
  let s;
  try { s = await (await fetch('/status.json', {cache:'no-store'})).json(); }
  catch (e) { document.getElementById('header').textContent = 'status server unreachable'; return; }
  const fresh = document.getElementById('freshness');
  if (s.state === 'no-status-yet') {
    document.getElementById('header').textContent = 'no run has started yet';
    fresh.textContent = 'idle'; fresh.className = 'pill stale';
    return;
  }
  const ageS = (Date.now() - Date.parse(s.updated_at)) / 1000;
  const stale = ageS > 180;
  fresh.textContent = stale ? `STALE ${Math.round(ageS)}s` : 'live';
  fresh.className = 'pill ' + (stale ? 'stale' : 'ok');
  document.getElementById('header').innerHTML =
    `target <b>${s.target}</b> · run ${s.run_id} · up ${s.uptime_hours}h · updated ${s.updated_at}`;
  document.getElementById('cards').innerHTML = [
    card('gen', s.generation),
    card('evals', `${s.total_evals}/${s.max_evals}`),
    card('evals/hr', s.recent_evals_per_hour),
    card('lands/hr', s.recent_lands_per_hour),
    card('land rate', (s.land_rate*100).toFixed(1)+'%'),
    card('best', `${s.best_reward} ${s.best_trick||''}`),
  ].join('');
  document.getElementById('devices').innerHTML = (s.devices||[]).map(d =>
    `<tr><td>${d.device_id}</td>
     <td><span class="pill ${d.alive?'alive':'dead'}">${d.alive?'alive':'DEAD'}</span></td>
     <td>${d.evals}</td><td>${d.lands}</td><td>${fmtTrick(d.last_trick)}</td></tr>`
  ).join('');
  document.getElementById('note').textContent = s.note ? ('note: ' + s.note) : '';
}
tick(); setInterval(tick, 5000);
</script>
</body>
</html>
"""


def _make_handler(status_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (stdlib naming)
            if self.path in ("/", "/index.html"):
                self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path.startswith("/status.json"):
                if status_path.exists():
                    self._send(200, status_path.read_bytes(), "application/json")
                else:
                    self._send(
                        200,
                        json.dumps({"state": "no-status-yet"}).encode("utf-8"),
                        "application/json",
                    )
            elif self.path.startswith("/healthz"):
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *args):  # silence per-request logging
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Status dashboard for the training rig.")
    parser.add_argument(
        "--status-path",
        type=Path,
        default=_REPO_ROOT / "logs" / "status.json",
        help="Path to status.json (default: logs/status.json).",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind host (default 127.0.0.1; tailscale serve proxies this).")
    parser.add_argument("--port", type=int, default=8200, help="Bind port (default 8200).")
    args = parser.parse_args()

    handler = _make_handler(args.status_path.resolve())
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Status server on http://{args.host}:{args.port}  (status: {args.status_path})")
    print(f"Expose over Tailscale:  tailscale serve --bg {args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down status server.")
        server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
