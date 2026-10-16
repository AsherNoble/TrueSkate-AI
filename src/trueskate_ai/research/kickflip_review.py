"""Operator review of the kickflip tolerance attempts (laptop): did the skater land a kickflip?

Built on the curved audit review UI, as adapted for the trick-speed spot check
(model1/trick/review.py and review.html): a temporary 127.0.0.1 server shows each admitted
attempt's native frames, from just before the pop to 3 s after the flick, keyboard-first.
The page names attempts by number only; nothing in it shows or implies the distance.
Export POSTs the report; the server checks it as `verdicts --report` would, saves it in
review/ and stops, so an agent running it in the background learns where it is. Verdicts
stay bound to each attempt's video hash.
"""
from __future__ import annotations

import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
from urllib.parse import urlparse

import cv2
import numpy as np

from trueskate_ai.research import kickflip_repeatability as recipe
from trueskate_ai.research import kickflip_tolerance as tolerance
from trueskate_ai.research.curve_measurement import _decode_source_frames
from trueskate_ai.research.curve_protocol import digest, save_new

BEFORE_POP_S, AFTER_FLICK_S = .5, 3.
FRAME_SIZE = (414, 896)  # Half the 828x1792 native frame; the trick banner stays legible.
JPEG_QUALITY = 85
MAX_REPORT_BYTES = 2**20
PORT = 8767


def review_manifest(root) -> tuple[dict, dict]:
    """The page's blind manifest, and per attempt what the frame cache decodes.

    The manifest's hash binds every attempt's video, so a report from another set of
    videos is refused.
    """
    root = Path(root)
    plan = tolerance.load_plan(root)
    samples, sources, bound = [], {}, []
    for attempt in plan['attempts']:
        number = attempt['number']
        trial = tolerance.attempt_dir(root, number)
        if tolerance.attempt_state(trial) != 'admitted':
            continue
        admission = recipe.read_json(trial / 'admission.json')
        pts = np.asarray(admission['original_pts_s'], dtype=float)
        gestures = {g['role']: g for g in admission['gestures']}
        pop, flick = gestures['pop'], gestures['flick']
        flick_end = flick['video_start_s'] + flick['encoded_duration_s']
        window = np.flatnonzero((pts >= pop['video_start_s'] - BEFORE_POP_S) & (pts <= flick_end + AFTER_FLICK_S))
        if len(window) < 2:
            raise ValueError(f'attempt {number:02d} has no frames around its kickflip')
        first, last = int(window[0]), int(window[-1])
        start = float(pts[first])
        marks = [dict(role=role, start_s=g['video_start_s'] - start,
                      end_s=g['video_start_s'] + g['encoded_duration_s'] - start)
                 for role, g in (('pop', pop), ('flick', flick))]
        samples.append(dict(id=number, code=f'{number:02d}', frame_times_s=(pts[first:last + 1] - start).tolist(),
                            marks=marks))
        sources[number] = dict(video=trial / 'original.mov', pts=pts.tolist(), first=first, last=last)
        bound.append([number, admission['video_sha256'], first, last])
    if not samples:
        raise ValueError('no admitted attempts to review')
    manifest = dict(experiment=tolerance.EXPERIMENT,
                    sha256=digest(dict(experiment=tolerance.EXPERIMENT, attempts=bound)), samples=samples)
    return manifest, sources


class FrameCache:
    """Decode each attempt's window once into JPEGs; attempts decode independently."""

    def __init__(self, folder: Path, sources: dict):
        self.folder, self.sources = folder, sources
        self.locks = {number: threading.Lock() for number in sources}

    def frame(self, number: int, index: int) -> bytes:
        source = self.sources.get(number)
        if source is None or not 0 <= index <= source['last'] - source['first']:
            raise ValueError('unknown frame')
        target = self.folder / str(number)
        with self.locks[number]:
            if not target.exists():
                building = self.folder / f'{number}.building'
                shutil.rmtree(building, ignore_errors=True)
                building.mkdir()
                written = iter(range(source['last'] - source['first'] + 1))

                def keep(image):
                    small = cv2.resize(image, FRAME_SIZE, interpolation=cv2.INTER_AREA)
                    if not cv2.imwrite(str(building / f'{next(written):04d}.jpg'), small,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                        raise ValueError('frame write failed')

                pts = source['pts']
                # The exact-source-frame decoder: it refuses when decoded and selected counts differ.
                _decode_source_frames(source['video'], pts, start=pts[source['first']], end=pts[source['last']],
                                      keep=False, inspect=keep)
                building.rename(target)
        return (target / f'{index:04d}.jpg').read_bytes()


def check_report(manifest: dict, exported) -> list[tuple[int, str, str]]:
    """(attempt, code, note) for every reviewed attempt; Not landed and Unsure need a note."""
    if not isinstance(exported, dict) or not isinstance(exported.get('attempts'), list):
        raise ValueError('not an exported kickflip review')
    if exported.get('manifest_sha256') != manifest['sha256']:
        raise ValueError('the report belongs to another set of attempt videos')
    numbers = [sample['id'] for sample in manifest['samples']]
    judged = {}
    for row in exported['attempts']:
        if not isinstance(row, dict):
            raise ValueError('not an exported kickflip review')
        label, code, note = row.get('code'), row.get('verdict'), str(row.get('note') or '').strip()
        number = int(label) if isinstance(label, str) and label.isdigit() else None
        if number not in numbers or number in judged:
            raise ValueError(f'attempt {label} is not under review or is listed twice')
        if code not in tolerance.VERDICTS:
            raise ValueError(f'attempt {label} has no Landed, Not landed or Unsure verdict')
        if code != 'L' and not note:
            raise ValueError(f'attempt {label} needs a note saying what was seen')
        judged[number] = (number, code, note)
    missing = [f'{number:02d}' for number in numbers if number not in judged]
    if missing:
        raise ValueError(f"the report misses attempt(s) {' '.join(missing)}")
    return [judged[number] for number in numbers]


def save_report(root, manifest: dict, exported) -> Path:
    """Check an exported review against the attempt videos and keep it in the run's review folder."""
    check_report(manifest, exported)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = Path(root) / 'review' / f'report-{stamp}.json'
    save_new(path, exported)
    return path


def import_report(root, report, *, replace: bool = False) -> int:
    """Record the page's exported review once it passes check_report."""
    manifest, _ = review_manifest(root)
    checked = check_report(manifest, recipe.read_json(report))
    return tolerance.write_verdicts(root, checked, replace=replace, source=Path(report).name)


def serve(root, *, port: int = PORT, open_browser=False) -> Path | None:
    """Serve the review until it is exported (returns the saved report), imported, or SIGINT/SIGTERM."""
    root = Path(root).resolve()
    manifest, sources = review_manifest(root)
    numbers = [sample['id'] for sample in manifest['samples']]
    judged = [f'{n:02d}' for n in numbers if (tolerance.attempt_dir(root, n) / 'verdict.json').exists()]
    if judged:
        raise ValueError(f"attempt(s) {' '.join(judged)} already have verdicts; the report step shows them")
    for number, source in sources.items():
        if recipe.file_sha(source['video']) != recipe.read_json(source['video'].with_name('admission.json'))['video_sha256']:
            raise ValueError(f'attempt {number:02d} video changed after admission')
    page = Path(__file__).with_name('kickflip_review.html').read_bytes()
    body = json.dumps(manifest).encode()
    saved = []

    def imported():
        return all((tolerance.attempt_dir(root, n) / 'verdict.json').exists() for n in numbers)

    with tempfile.TemporaryDirectory(prefix='kickflip-review-', dir=root) as temporary:
        cache = FrameCache(Path(temporary), sources)

        class Handler(BaseHTTPRequestHandler):
            def reply(self, status: int, value: dict):
                data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if urlparse(self.path).path != '/export' or saved:
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get('Content-Length') or 0)
                    if not 0 < length <= MAX_REPORT_BYTES:
                        raise ValueError('the report is missing or too large')
                    path = save_report(root, manifest, json.loads(self.rfile.read(length)))
                except (ValueError, OSError) as exc:  # a malformed report is a ValueError too
                    self.reply(400, dict(error=str(exc)))
                    return
                self.reply(200, dict(path=str(path)))
                saved.append(path)  # after the reply, so the page hears before the server stops

            def do_GET(self):
                route = urlparse(self.path).path
                try:
                    if route == '/':
                        data, mime, caching = page, 'text/html; charset=utf-8', 'no-store'
                    elif route == '/manifest':
                        data, mime, caching = body, 'application/json', 'no-store'
                    elif route.startswith('/frame/'):
                        _, _, number, index = route.split('/')
                        # URLs carry the manifest hash as a query, so frames are immutable.
                        data, mime = cache.frame(int(number), int(index)), 'image/jpeg'
                        caching = 'private, max-age=86400, immutable'
                    else:
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', mime)
                    self.send_header('Content-Length', str(len(data)))
                    self.send_header('Cache-Control', caching)
                    self.end_headers()
                    self.wfile.write(data)
                except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
                    self.send_error(400, str(exc))

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        server.daemon_threads = True
        server.timeout = .5
        url = f'http://127.0.0.1:{server.server_port}'
        print(f'Kickflip review: {url} (temporary; it stops when the review is exported)', flush=True)
        if open_browser:
            subprocess.run(['open', '-a', 'Google Chrome', url], check=True)

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f'signal {signum}')

        # Background jobs inherit an ignored SIGINT; handle both so the frame cache is removed.
        previous = {}
        if threading.current_thread() is threading.main_thread():
            previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            while not (saved or imported()):
                server.handle_request()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            for s, handler in previous.items():
                signal.signal(s, handler)
    return saved[0] if saved else None
