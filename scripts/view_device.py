"""Remote live-view of a connected iPhone over Tailscale.

Captures the iPhone's screen via macOS **AVFoundation** — the same hardware
H.264-over-USB path QuickTime uses — NOT WDA's MJPEG stream. This is deliberate:
WDA's MJPEG server is single-client and fragile, and the OCR FrameRecorder
(`device_worker.FrameRecorder`) needs it during every eval. A second MJPEG
consumer (the old `ffplay` --view) crashed/disconnected WDA. AVFoundation is a
completely separate pipe, so live-view here coexists with WDA + Appium + OCR.

Plain ffmpeg cannot see iOS devices at all: only processes that set the
CoreMediaIO `kCMIOHardwarePropertyAllowScreenCaptureDevices` flag (QuickTime
does; ffmpeg doesn't) get them in the AVFoundation device list. We inject a
tiny dylib (`tools/enable_dal.c`, built on first run) via DYLD_INSERT_LIBRARIES
that sets the flag and pumps the run loop until the device registers.

Pipeline:  AVFoundation capture → h264_videotoolbox (HW encode) → HLS segments
           → tiny HTTP server → `tailscale serve` → watch in any browser.

Usage:
    # 1. discover the device's AVFoundation name (iPhone must be plugged in,
    #    unlocked, trusted; QuickTime must NOT be holding the camera):
    python scripts/view_device.py --list

    # 2. start the stream (binds 127.0.0.1:8300 by default). Pass the device
    #    by [index] from --list, or by exact name — note Apple device names
    #    use a CURLY apostrophe (’ U+2019), not a straight one:
    python scripts/view_device.py --device 1
    python scripts/view_device.py --device "Asher’s iPhone"

    # 3. expose over your tailnet on a PATH so it doesn't clobber the
    #    dashboard's `tailscale serve --bg 8200` at the root:
    tailscale serve --bg --set-path /view 8300
    #    → watch at  https://<host>.<tailnet>.ts.net/view/
    #    undo with:  tailscale serve --set-path /view off

Caveats:
- The AVFoundation iPhone device is EXCLUSIVE — one consumer at a time. You
  cannot run this while QuickTime (or another ffmpeg) holds the same phone.
  It does NOT conflict with WDA/Appium/OCR, which use a different subsystem.
- The phone must be UNLOCKED with the screen on; if it locks, capture stalls
  at zero frames (ffmpeg hangs, no error). A watchdog detects the frozen
  playlist and relaunches ffmpeg, so the stream self-heals once the phone is
  unlocked again — no manual restart or page refresh needed.
- The hosting app (terminal/IDE) needs macOS Screen Recording permission, or
  capture opens but never delivers frames.
- macOS CoreMediaIO can wedge the iPhone's DAL screen-capture stream (one
  CMIO-level event froze both phones at the same instant). The device still
  enumerates and ffmpeg opens it, but zero frames arrive and the watchdog's
  relaunches can't help — the wedge is below ffmpeg. QuickTime's capture path
  is unaffected, and opening it RE-ACTIVATES the session: open QuickTime >
  New Movie Recording, pick the device once, then quit QuickTime and the
  ffmpeg capture works again. A USB replug re-registers the device but does
  NOT re-activate the stream; restarting CoreMediaIO/usbmuxd doesn't fix it.
- Run this on whichever Mac the phone is physically attached to (the Intel
  `training-server` for the rig's collection phones; this laptop for the 11).
- HLS latency is ~4-8s. If you need near-real-time later, swap the HLS sink for
  mediamtx (RTSP in → WebRTC out, sub-second) — capture command is identical.
"""
import argparse
import functools
import http.server
import os
import shutil
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

DEFAULT_PORT = 8300  # avoids 4723-5 (appium), 8100-2 (wda), 9100-2 (mjpeg), 8200 (dashboard)
PLAYLIST = "stream.m3u8"

# Watchdog: AVFoundation capture can stall at zero frames with no error (phone
# locks, USB hiccup, DAL glitch). ffmpeg stays alive but stops writing HLS
# segments, so the playlist freezes and the player spins after the stale tail.
# We treat "no new segment for STALL_TIMEOUT" as a hang and relaunch ffmpeg.
STALL_TIMEOUT = 12.0   # seconds without a fresh segment ⇒ encoder hung
START_GRACE = 15.0     # seconds to allow first segment after each (re)launch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SHIM_SRC = _REPO_ROOT / "tools" / "enable_dal.c"
_SHIM_DYLIB = _REPO_ROOT / "tools" / "libenable_dal.dylib"


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found. Install with: brew install ffmpeg")


def _shim_env() -> dict:
    """Environment for ffmpeg with the CoreMediaIO DAL shim injected.

    Builds tools/libenable_dal.dylib from tools/enable_dal.c on first run (or
    when the source is newer). Needs clang, which these Macs have via Xcode.
    """
    if not _SHIM_DYLIB.exists() or _SHIM_DYLIB.stat().st_mtime < _SHIM_SRC.stat().st_mtime:
        print(f"Building DAL shim → {_SHIM_DYLIB}")
        subprocess.run(
            ["clang", "-x", "c", "-dynamiclib", str(_SHIM_SRC),
             "-framework", "CoreMediaIO", "-framework", "CoreFoundation",
             "-o", str(_SHIM_DYLIB)],
            check=True,
        )
    env = os.environ.copy()
    env["DYLD_INSERT_LIBRARIES"] = str(_SHIM_DYLIB)
    return env


def list_devices() -> None:
    """Print AVFoundation capture devices. The iPhone shows up by its display
    name (e.g. 'Asher’s iPhone') once plugged in, unlocked, and trusted."""
    _require_ffmpeg()
    # ffmpeg prints the device table to stderr and exits non-zero by design.
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True, env=_shim_env(),
    )
    print(proc.stderr.rstrip())
    print("\nPass the iPhone's [index] or exact name to --device, e.g.:")
    print("    python scripts/view_device.py --device \"Asher’s iPhone\"   # curly ’, not '")


def _index_html() -> str:
    """Self-contained player page: hls.js for Chrome/Firefox, native HLS for Safari."""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TrueSkate live view</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>html,body{{margin:0;background:#111;height:100%}}
video{{width:100%;height:100%;object-fit:contain}}</style></head>
<body>
<video id="v" autoplay muted playsinline controls></video>
<script src="https://cdn.jsdelivr.net/npm/hls.js@1/dist/hls.min.js"></script>
<script>
const v = document.getElementById('v'), src = '{PLAYLIST}';
let hls = null;
// Not low-latency HLS (ffmpeg emits plain segments), so don't ride the very
// live edge — that stalls constantly. Keep a few segments of cushion and let
// hls.js retry playlist/segment loads for a long time so the player survives an
// encoder restart (the watchdog clears + rebuilds the playlist) and reconnects
// on its own instead of freezing on a spinner until a manual refresh.
function startHls() {{
  hls = new Hls({{
    liveSyncDurationCount: 4, liveMaxLatencyDurationCount: 12, lowLatencyMode: false,
    manifestLoadingMaxRetry: 1000, manifestLoadingRetryDelay: 1000, manifestLoadingMaxRetryTimeout: 8000,
    levelLoadingMaxRetry: 1000, levelLoadingRetryDelay: 1000, levelLoadingMaxRetryTimeout: 8000,
    fragLoadingMaxRetry: 8,
  }});
  hls.loadSource(src); hls.attachMedia(v);
  hls.on(Hls.Events.ERROR, (_, d) => {{
    if (!d.fatal) return;
    if (d.type === Hls.ErrorTypes.NETWORK_ERROR) {{ hls.startLoad(); }}        // playlist/segment gap (encoder restarting) → keep retrying
    else if (d.type === Hls.ErrorTypes.MEDIA_ERROR) {{ hls.recoverMediaError(); }}  // decode hiccup at the discontinuity
    else {{ try {{ hls.destroy(); }} catch (e) {{}} setTimeout(startHls, 2000); }}    // unrecoverable → rebuild from scratch
  }});
}}
if (v.canPlayType('application/vnd.apple.mpegurl')) {{          // Safari / iOS
  v.src = src;
  // Native HLS has no JS-level retry; re-point at the source if it errors out.
  v.addEventListener('error', () => setTimeout(() => {{ v.src = src; v.load(); v.play(); }}, 2000));
}} else if (window.Hls && Hls.isSupported()) {{                  // Chrome / Firefox
  startHls();
}}
// Live monitoring stream: never stay paused, never drift behind the live edge.
function liveEdge() {{
  if (v.seekable.length) v.currentTime = v.seekable.end(v.seekable.length - 1) - 0.5;
}}
v.addEventListener('pause', () => setTimeout(() => {{ liveEdge(); v.play(); }}, 300));
setInterval(() => {{
  if (v.paused) {{ liveEdge(); v.play(); }}
  else if (v.seekable.length && v.seekable.end(v.seekable.length - 1) - v.currentTime > 12) liveEdge();
}}, 5000);
document.addEventListener('visibilitychange', () => {{
  if (!document.hidden) {{ liveEdge(); v.play(); }}
}});
</script>
</body></html>"""


def build_ffmpeg_cmd(device: str, out_dir: Path, framerate: int, bitrate: str,
                     scale_height: int = 0) -> list[str]:
    scale = f",scale=-2:{scale_height}" if scale_height else ""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-f", "avfoundation", "-framerate", str(framerate),
        "-i", f"{device}:none",                  # ':none' = video only, no audio capture
        # iPhone DAL frames all arrive with pts=0; without real timestamps the
        # HLS muxer never accrues segment duration and writes no segments.
        # (-use_wallclock_as_timestamps does NOT fix this for avfoundation.)
        # The fps filter is the real framerate cap — the -framerate input
        # option doesn't constrain DAL devices (they deliver up to 60fps and
        # the encode falls behind realtime under load → player stalls).
        "-vf", f"settb=AVTB,setpts='(RTCTIME-RTCSTART)/(TB*1000000)',fps={framerate}{scale}",
        "-c:v", "h264_videotoolbox", "-realtime", "1",
        "-b:v", bitrate, "-g", str(framerate * 2),  # keyframe every 2s = one per segment
        "-f", "hls", "-hls_time", "2", "-hls_list_size", "6",
        "-hls_flags", "delete_segments+independent_segments+omit_endlist",
        str(out_dir / PLAYLIST),
    ]


def _spawn_encoder(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, env=_shim_env())


def _terminate_encoder(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def _clear_hls(out_dir: Path) -> None:
    """Drop the stale playlist + segments so a relaunched encoder serves a clean
    stream instead of one whose media sequence jumps backwards."""
    for seg in out_dir.glob("*.ts"):
        seg.unlink(missing_ok=True)
    (out_dir / PLAYLIST).unlink(missing_ok=True)


def _segment_age(out_dir: Path) -> float | None:
    """Seconds since the playlist was last rewritten (ffmpeg rewrites it on every
    new segment), or None if it doesn't exist yet."""
    try:
        return time.time() - (out_dir / PLAYLIST).stat().st_mtime
    except FileNotFoundError:
        return None


def run_with_watchdog(device: str, out_dir: Path, framerate: int, bitrate: str,
                      scale_height: int, stop: threading.Event, *,
                      cmd_builder=build_ffmpeg_cmd, spawn=_spawn_encoder,
                      stall_timeout: float = STALL_TIMEOUT,
                      start_grace: float = START_GRACE) -> None:
    """Keep the encoder alive: relaunch it if it exits AND restart it if it hangs
    (delivers no new segments). Runs until `stop` is set."""
    cmd = cmd_builder(device, out_dir, framerate, bitrate, scale_height)
    ff = spawn(cmd)
    launched = time.monotonic()
    restarts = 0
    fails_since_healthy = 0  # consecutive restarts without the stream coming back
    healthy = False
    try:
        while not stop.is_set():
            exited = ff.poll() is not None
            age = _segment_age(out_dir)
            stalled = (age is None or age > stall_timeout) and \
                      (time.monotonic() - launched) > start_grace
            if exited or stalled:
                reason = (f"encoder exited ({ff.returncode})" if exited
                          else f"no new segment for >{stall_timeout:.0f}s — encoder hung")
                restarts += 1
                fails_since_healthy += 1
                # Back off when restarts don't help (bad device, phone locked with
                # zero frames) so we don't spin; resets once the stream recovers.
                backoff = min(30.0, 2.0 ** min(fails_since_healthy, 5))
                print(f"[{device}] {reason}; restarting in {backoff:.0f}s (#{restarts}).",
                      flush=True)
                _terminate_encoder(ff)
                _clear_hls(out_dir)
                if stop.wait(backoff):
                    break
                ff = spawn(cmd)
                launched = time.monotonic()
                healthy = False
                continue
            if not healthy and age is not None and age <= stall_timeout:
                healthy = True
                fails_since_healthy = 0
                print(f"[{device}] stream live"
                      f"{f' (recovered after {restarts} restart(s))' if restarts else ''}.",
                      flush=True)
            stop.wait(2.0)
    finally:
        _terminate_encoder(ff)


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the HLS dir; correct MIME for .m3u8/.ts and no-cache on the playlist."""

    def end_headers(self):
        if self.path.endswith(".m3u8"):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def guess_type(self, path):
        if path.endswith(".m3u8"):
            return "application/vnd.apple.mpegurl"
        if path.endswith(".ts"):
            return "video/mp2t"
        return super().guess_type(path)

    def log_message(self, *args):
        pass  # quiet — ffmpeg warnings are the signal we care about


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        # HLS players abort in-flight segment fetches constantly; a client
        # hanging up mid-response is normal operation, not a server error.
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote live-view of a connected iPhone over Tailscale.")
    parser.add_argument("--list", action="store_true", help="List AVFoundation devices and exit.")
    parser.add_argument("--device", help="AVFoundation device name (e.g. \"Asher's iPhone\") or index.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"HTTP bind port (default {DEFAULT_PORT}).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1; tailscale serve proxies this).")
    parser.add_argument("--framerate", type=int, default=30, help="Capture framerate (default 30).")
    parser.add_argument("--bitrate", default="6M", help="H.264 bitrate (default 6M).")
    parser.add_argument("--scale-height", type=int, default=0,
                        help="Downscale video to this height (0 = native). Use ~640 for remote viewing.")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return
    if not args.device:
        parser.error("--device is required (use --list to discover it).")

    _require_ffmpeg()
    tmp = Path(tempfile.mkdtemp(prefix="trueskate_view_"))
    (tmp / "index.html").write_text(_index_html())
    try:
        handler = functools.partial(_Handler, directory=str(tmp))
        try:
            httpd = _Server((args.host, args.port), handler)
        except OSError as e:
            sys.exit(f"Cannot bind {args.host}:{args.port} ({e.strerror}) — is another "
                     "view_device.py already running? Stop it or pass --port.")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        print(f"Capturing '{args.device}' → HLS in {tmp}")
        print(f"Local:   http://{args.host}:{args.port}/")
        print(f"Tailnet: tailscale serve --bg --set-path /view {args.port}")
        print(f"         then watch https://<host>.<tailnet>.ts.net/view/")
        print("Ctrl+C to stop.")

        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        # Watchdog: relaunch ffmpeg if it exits (bad device, phone unplugged) AND
        # if it hangs delivering zero frames (phone locked, DAL stall) — the latter
        # froze the stream silently before, requiring a manual page refresh.
        run_with_watchdog(args.device, tmp, args.framerate, args.bitrate,
                          args.scale_height, stop)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print("Stopped, cleaned up.")


if __name__ == "__main__":
    main()
