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
  at zero frames (ffmpeg hangs, no error).
- The hosting app (terminal/IDE) needs macOS Screen Recording permission, or
  capture opens but never delivers frames.
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
from pathlib import Path

DEFAULT_PORT = 8300  # avoids 4723-5 (appium), 8100-2 (wda), 9100-2 (mjpeg), 8200 (dashboard)
PLAYLIST = "stream.m3u8"

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
if (v.canPlayType('application/vnd.apple.mpegurl')) {{          // Safari / iOS
  v.src = src;
}} else if (window.Hls && Hls.isSupported()) {{                  // Chrome / Firefox
  const hls = new Hls({{liveSyncDurationCount: 2, lowLatencyMode: true}});
  hls.loadSource(src); hls.attachMedia(v);
  hls.on(Hls.Events.ERROR, (_, d) => {{ if (d.fatal) setTimeout(() => {{
    hls.loadSource(src); hls.startLoad(); }}, 1500); }});         // retry while stream warms up
}}
</script>
</body></html>"""


def build_ffmpeg_cmd(device: str, out_dir: Path, framerate: int, bitrate: str) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-f", "avfoundation", "-framerate", str(framerate),
        "-i", f"{device}:none",                  # ':none' = video only, no audio capture
        # iPhone DAL frames all arrive with pts=0; without real timestamps the
        # HLS muxer never accrues segment duration and writes no segments.
        # (-use_wallclock_as_timestamps does NOT fix this for avfoundation.)
        "-vf", "settb=AVTB,setpts='(RTCTIME-RTCSTART)/(TB*1000000)'",
        "-c:v", "h264_videotoolbox", "-realtime", "1",
        "-b:v", bitrate, "-g", str(framerate * 2),  # keyframe every 2s = one per segment
        "-f", "hls", "-hls_time", "2", "-hls_list_size", "6",
        "-hls_flags", "delete_segments+independent_segments+omit_endlist",
        str(out_dir / PLAYLIST),
    ]


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
    args = parser.parse_args()

    if args.list:
        list_devices()
        return
    if not args.device:
        parser.error("--device is required (use --list to discover it).")

    _require_ffmpeg()
    tmp = Path(tempfile.mkdtemp(prefix="trueskate_view_"))
    (tmp / "index.html").write_text(_index_html())
    ff = None
    try:
        handler = functools.partial(_Handler, directory=str(tmp))
        try:
            httpd = _Server((args.host, args.port), handler)
        except OSError as e:
            sys.exit(f"Cannot bind {args.host}:{args.port} ({e.strerror}) — is another "
                     "view_device.py already running? Stop it or pass --port.")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        ff = subprocess.Popen(
            build_ffmpeg_cmd(args.device, tmp, args.framerate, args.bitrate),
            env=_shim_env(),
        )

        print(f"Capturing '{args.device}' → HLS in {tmp}")
        print(f"Local:   http://{args.host}:{args.port}/")
        print(f"Tailnet: tailscale serve --bg --set-path /view {args.port}")
        print(f"         then watch https://<host>.<tailnet>.ts.net/view/")
        print("Ctrl+C to stop.")

        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        # Exit if ffmpeg dies (bad device name, phone unplugged, QuickTime holding it).
        while not stop.is_set():
            if ff.poll() is not None:
                print(f"\nffmpeg exited ({ff.returncode}) — check the device name/connection "
                      "and that QuickTime isn't holding the camera.")
                break
            stop.wait(0.5)
    finally:
        if ff and ff.poll() is None:
            ff.terminate()
            try:
                ff.wait(timeout=3)
            except subprocess.TimeoutExpired:
                ff.kill()
        shutil.rmtree(tmp, ignore_errors=True)
        print("Stopped, cleaned up.")


if __name__ == "__main__":
    main()
