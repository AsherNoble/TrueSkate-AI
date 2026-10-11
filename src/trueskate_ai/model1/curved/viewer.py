"""Temporary native-frame human audit server. Stops when an export is imported."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
from urllib.parse import urlparse

from .audit import verify_manifest
from .capture import file_sha256


def serve(root: Path, number: int, *, port=8765, open_browser=False):
    root = Path(root).resolve()
    manifest = json.loads((root/f"audit-{number}"/"manifest.json").read_text())
    verify_manifest(manifest)
    paths = {}
    for row in manifest["samples"]:
        path = (root/row["sample_dir"]).resolve()
        if (not path.is_relative_to(root) or file_sha256(path/"clip.mp4") != row["media_sha256"]
                or file_sha256(path/"meta.json") != row["meta_sha256"]):
            raise ValueError("audit sample changed")
        paths[row["id"]] = path
    cache_lock = threading.Lock()
    with tempfile.TemporaryDirectory(prefix="audit-viewer-", dir=root) as temporary:
        cache = Path(temporary)
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                route = urlparse(self.path).path
                try:
                    if route == "/":
                        data, mime = Path(__file__).with_name("viewer.html").read_bytes(), "text/html"
                    elif route == "/manifest":
                        data, mime = json.dumps(manifest).encode(), "application/json"
                    elif route.startswith("/frame/"):
                        _, _, identity, frame = route.split("/")
                        identity, frame = int(identity), int(frame)
                        if identity not in paths or not 0 <= frame < 120:
                            raise ValueError("unknown frame")
                        folder = cache/str(identity)
                        with cache_lock:
                            if not folder.exists():
                                building = folder.with_name(folder.name+".building")
                                building.mkdir(exist_ok=True)
                                subprocess.run(["ffmpeg", "-v", "error", "-i", str(paths[identity]/"clip.mp4"),
                                                "-fps_mode", "passthrough", "-q:v", "2", "-threads", "2",
                                                "-y", str(building/"%03d.jpg")], check=True, capture_output=True)
                                if len(list(building.glob("*.jpg"))) != 120:
                                    raise ValueError("native decode count mismatch")
                                building.rename(folder)
                        data, mime = (folder/f"{frame+1:03}.jpg").read_bytes(), "image/jpeg"
                    else:
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
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
        print(f"Audit viewer: {url} (temporary; export report then import it)", flush=True)
        if open_browser:
            subprocess.run(["open", "-a", "Google Chrome", url], check=True)
        try:
            # Importing the report writes this file and closes the viewer automatically.
            while not (root/f"audit-{number}"/"review.json").exists():
                server.handle_request()
        finally:
            server.server_close()
