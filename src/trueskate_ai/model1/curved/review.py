"""Temporary native-frame human audit review server (presentation only).

This replaces viewer.py/viewer.html for reviewing. Those two files stay
byte-identical because qualification rounds are sealed under a pipeline
fingerprint that includes them, and production must reuse that exact
fingerprint. Nothing here affects collection, admission or scoring: exported
reports are imported and scored by ledger.import_review/audit.score_review.

Every clip and its metadata are verified against the sealed manifest before the
server starts, so a hash-identical local mirror of a rig run can be reviewed.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
from urllib.parse import urlparse

from .audit import verify_manifest
from .capture import file_sha256

FRAMES = 120


def verified_samples(root: Path, number: int):
    """Return the sealed manifest, sample folders and per-sample session context."""
    manifest = json.loads((root/f"audit-{number}"/"manifest.json").read_text())
    verify_manifest(manifest)
    if manifest.get("round") != number:
        raise ValueError("manifest belongs to a different round")
    paths, context = {}, {}
    for row in manifest["samples"]:
        path = (root/row["sample_dir"]).resolve()
        if (not path.is_relative_to(root) or file_sha256(path/"clip.mp4") != row["media_sha256"]
                or file_sha256(path/"meta.json") != row["meta_sha256"]):
            raise ValueError("audit sample changed")
        meta = json.loads((path/"meta.json").read_text())
        paths[row["id"]] = path
        context[row["id"]] = dict(session=meta.get("session_id"), settings_note=meta.get("settings_note"))
    return manifest, paths, context


class FrameCache:
    """Decode each clip once into native-frame JPEGs; clips decode independently."""

    def __init__(self, folder: Path, paths: dict):
        self.folder, self.paths = folder, paths
        self.locks = {identity: threading.Lock() for identity in paths}

    def frame(self, identity: int, index: int) -> bytes:
        if identity not in self.paths or not 0 <= index < FRAMES:
            raise ValueError("unknown frame")
        target = self.folder/str(identity)
        with self.locks[identity]:
            if not target.exists():
                building = self.folder/f"{identity}.building"
                shutil.rmtree(building, ignore_errors=True)
                building.mkdir()
                subprocess.run(["ffmpeg", "-v", "error", "-i", str(self.paths[identity]/"clip.mp4"),
                                "-fps_mode", "passthrough", "-q:v", "3", "-y", str(building/"%03d.jpg")],
                               check=True, capture_output=True)
                if len(list(building.glob("*.jpg"))) != FRAMES:
                    raise ValueError("native decode count mismatch")
                building.rename(target)
        return (target/f"{index+1:03}.jpg").read_bytes()


def serve(root: Path, number: int, *, port=8765, open_browser=False):
    """Serve until the review is imported (review.json exists) or SIGINT/SIGTERM."""
    root = Path(root).resolve()
    manifest, paths, context = verified_samples(root, number)
    page = Path(__file__).with_name("review.html").read_bytes()
    bodies = {"/manifest": json.dumps(manifest).encode(),
              "/context": json.dumps({str(k): v for k, v in context.items()}).encode()}
    with tempfile.TemporaryDirectory(prefix="audit-review-", dir=root) as temporary:
        cache = FrameCache(Path(temporary), paths)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                route = urlparse(self.path).path
                try:
                    if route == "/":
                        data, mime, caching = page, "text/html; charset=utf-8", "no-store"
                    elif route in bodies:
                        data, mime, caching = bodies[route], "application/json", "no-store"
                    elif route.startswith("/frame/"):
                        _, _, identity, index = route.split("/")
                        # URLs carry the manifest hash as a query, so frames are immutable.
                        data, mime = cache.frame(int(identity), int(index)), "image/jpeg"
                        caching = "private, max-age=86400, immutable"
                    else:
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", caching)
                    self.end_headers()
                    self.wfile.write(data)
                except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
                    self.send_error(400, str(exc))

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        server.daemon_threads = True
        server.timeout = .5
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Audit review: {url} (temporary; export report then import it)", flush=True)
        if open_browser:
            subprocess.run(["open", "-a", "Google Chrome", url], check=True)

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")

        # Background jobs inherit an ignored SIGINT; handle both so the frame cache is removed.
        previous = {}
        if threading.current_thread() is threading.main_thread():
            previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            while not (root/f"audit-{number}"/"review.json").exists():
                server.handle_request()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            for s, handler in previous.items():
                signal.signal(s, handler)


def main(argv=None):
    p = argparse.ArgumentParser(description="Serve a sealed curved audit round for human review.")
    p.add_argument("--root", type=Path, required=True, help="run directory or verified mirror of it")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="open the page in Google Chrome")
    a = p.parse_args(argv)
    serve(a.root, a.round, port=a.port, open_browser=a.open)


if __name__ == "__main__":
    main()
