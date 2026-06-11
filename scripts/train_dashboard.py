"""Combined training dashboard: both phone streams + distilled training log.

Serves a single page with the two view_device.py HLS streams (iframed from
their own servers, so no CORS plumbing) and a distilled per-device log built
from the newest run JSONL: current trick, eval counts, throughput, rolling
land rate, and the latest landed tricks.

Usage:
    python scripts/train_dashboard.py [--port 8400] [--host 0.0.0.0]
        [--stream-host 100.83.159.74] [--log-root logs/overnight]

Open http://127.0.0.1:8400/ (or the tailnet IP from your phone).
"""
import argparse
import functools
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEVICES = ["iPhone_XR", "iPhone_XR2"]
STREAM_PORTS = {"iPhone_XR": 8300, "iPhone_XR2": 8301}


def _newest_jsonl(log_root: Path, device: str) -> Path | None:
    runs = sorted(log_root.glob(f"{device}/*/runs/cmaes_run_*/cmaes_run_*.jsonl"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _device_status(log_root: Path, device: str) -> dict:
    j = _newest_jsonl(log_root, device)
    if j is None:
        return {"device": device, "trick": "—", "evals": 0, "note": "no run found"}
    target = None
    rows = []
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
iframe{width:100%;height:100%;border:1px solid #30363d;border-radius:8px;background:#000}
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
<iframe src="http://__STREAM_HOST__:8300/" allow="autoplay"></iframe>
<iframe src="http://__STREAM_HOST__:8301/" allow="autoplay"></iframe>
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
tick(); setInterval(tick, 5000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, log_root: Path, stream_host: str, **kwargs):
        self.log_root = log_root
        self.stream_host = stream_host
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/data":
            body = json.dumps([_device_status(self.log_root, d) for d in DEVICES]).encode()
            ctype = "application/json"
        else:
            body = _PAGE.replace("__STREAM_HOST__", self.stream_host).encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined training dashboard.")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--stream-host", default="100.83.159.74",
                        help="Host the view_device streams are bound to")
    parser.add_argument("--log-root", type=Path, default=_REPO_ROOT / "logs" / "overnight")
    args = parser.parse_args()

    handler = functools.partial(_Handler, log_root=args.log_root, stream_host=args.stream_host)
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as e:
        sys.exit(f"Cannot bind {args.host}:{args.port} ({e.strerror}) — already running?")
    print(f"Dashboard: http://127.0.0.1:{args.port}/  (tailnet: http://{args.stream_host}:{args.port}/)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
