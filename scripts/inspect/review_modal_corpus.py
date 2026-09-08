"""Browsable local dashboard for eyeballing the offloaded SLS corpus on Modal.

`offload_corpus_to_modal.sh` uploads frame->gesture samples to the
`trueskate-corpus` Modal Volume and deletes them locally once verified — so
after offload, the ONLY copy of most of the corpus lives on Modal. This script
lets you actually look at it: it lists sessions/parks/samples straight off the
Volume (cheap, non-recursive `listdir` calls — no bulk download, no full-tree
scan, so it starts instantly regardless of how many million frames are up
there) and fetches individual frames on demand, caching them to
`--cache-dir` so repeat views are instant and re-browsing doesn't re-pull from
Modal.

Nothing is downloaded until you actually view it. A "Random sample" link picks
a session/park/sample uniformly-ish at random for quick spot-checks across the
whole corpus without hand-picking paths.

Usage:
    python scripts/inspect/review_modal_corpus.py [--port 8410]

Open http://127.0.0.1:8410/
"""
from __future__ import annotations

import argparse
import functools
import html
import io
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

import modal
from modal.volume import FileEntryType

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_SESSION_RE = re.compile(r"^(iPhone_XR2?)_(\d{8})_(\d{6})$")
_PAGE_SIZE = 48
_THUMB_W = 220


class CorpusBrowser:
    """Thin lazy wrapper over the Modal Volume + a local on-disk frame cache."""

    # Measured against this volume: `listdir` takes ~20-23s FLAT — independent of
    # the target directory's size and NOT faster on a repeat call (no server-side
    # caching to lean on) — while a `listdir` root has 52 entries and a small park
    # has 12, both cost the same ~20s. So every listdir result is cached locally
    # for this long; a page is slow once per (session[, park]) and instant after.
    _LISTDIR_CACHE_TTL_S = 600.0

    def __init__(self, volume_name: str, cache_dir: Path):
        self.vol = modal.Volume.from_name(volume_name)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._listdir_cache: dict[str, tuple[float, list]] = {}
        # Grid pages fan out ~48 small per-sample Modal reads (meta.json / first
        # frame) — sequential round trips make a page take tens of seconds, so
        # every such fan-out goes through this pool instead.
        self._pool = ThreadPoolExecutor(max_workers=16)

    def _cached_listdir(self, path: str) -> list:
        now = time.time()
        hit = self._listdir_cache.get(path)
        if hit is not None and now - hit[0] < self._LISTDIR_CACHE_TTL_S:
            return hit[1]
        entries = self.vol.listdir(path)
        self._listdir_cache[path] = (now, entries)
        return entries

    # -- listing (cheap, non-recursive; never walks the full corpus) -------

    def sessions(self) -> list[str]:
        entries = self._cached_listdir("/")
        return sorted(
            (e.path for e in entries if e.type == FileEntryType.DIRECTORY),
            reverse=True,  # session names are DEVICE_YYYYMMDD_HHMMSS -> newest first
        )

    def parks(self, session: str) -> list[str]:
        entries = self._cached_listdir(session)
        return sorted(
            Path(e.path).name for e in entries if e.type == FileEntryType.DIRECTORY
        )

    def samples(self, session: str, park: str) -> list[str]:
        entries = self._cached_listdir(f"{session}/{park}")
        return sorted(
            Path(e.path).name for e in entries if e.type == FileEntryType.DIRECTORY
        )

    def frames(self, session: str, park: str, sample: str) -> list[str]:
        entries = self._cached_listdir(f"{session}/{park}/{sample}")
        return sorted(
            Path(e.path).name for e in entries
            if e.type == FileEntryType.FILE and Path(e.path).name.startswith("frame_")
        )

    # -- fetch + cache -------------------------------------------------------

    def _cached_bytes(self, remote_path: str) -> bytes:
        local = self.cache_dir / remote_path
        if local.exists():
            return local.read_bytes()
        local.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        self.vol.read_file_into_fileobj(remote_path, buf)
        data = buf.getvalue()
        local.write_bytes(data)
        return data

    def frame_bytes(self, session: str, park: str, sample: str, frame: str) -> bytes:
        return self._cached_bytes(f"{session}/{park}/{sample}/{frame}")

    def meta(self, session: str, park: str, sample: str) -> dict:
        try:
            raw = self._cached_bytes(f"{session}/{park}/{sample}/meta.json")
            return json.loads(raw)
        except Exception:  # noqa: BLE001 — missing/corrupt meta shouldn't break browsing
            return {}

    def metas_parallel(self, session: str, park: str, samples: list[str]) -> dict[str, dict]:
        """meta() for many samples at once, fanned out across the pool.

        Sequentially this is one Modal round trip per sample (~tens of seconds
        for a 48-sample grid page); in parallel it's bounded by the slowest one.
        """
        futures = {s: self._pool.submit(self.meta, session, park, s) for s in samples}
        return {s: f.result() for s, f in futures.items()}

    def thumb_bytes(self, session: str, park: str, sample: str, frame: str) -> bytes:
        local = self.cache_dir / ".thumbs" / session / park / sample / frame
        if local.exists():
            return local.read_bytes()
        from PIL import Image
        full = self.frame_bytes(session, park, sample, frame)
        img = Image.open(io.BytesIO(full)).convert("RGB")
        w, h = img.size
        new_h = round(h * (_THUMB_W / w))
        img = img.resize((_THUMB_W, new_h))
        local.parent.mkdir(parents=True, exist_ok=True)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=78)
        data = out.getvalue()
        local.write_bytes(data)
        return data

    def random_sample(self) -> tuple[str, str, str] | None:
        # Each candidate costs up to 2 uncached listdir calls (~20s flat each per
        # the benchmark above), so this tries very few sessions before giving up —
        # a wide search would compound into minutes on a cold cache. Sessions/parks
        # that DO get hit stay warm in _listdir_cache, so repeated /random clicks
        # get faster as the cache fills in.
        sessions = self.sessions()
        random.shuffle(sessions)
        for session in sessions[:3]:
            parks = self.parks(session)
            if not parks:
                continue
            park = random.choice(parks)
            samples = self.samples(session, park)
            if not samples:
                continue
            return session, park, random.choice(samples)
        return None


def _parse_session(name: str) -> tuple[str, str]:
    m = _SESSION_RE.match(name)
    if not m:
        return "?", name
    device, ymd, hms = m.groups()
    return device, f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]} {hms[:2]}:{hms[2:4]}:{hms[4:]}"


_STYLE = """
html,body{margin:0;background:#0d1117;color:#c9d1d9;font:13px/1.5 ui-monospace,Menlo,monospace}
a{color:#58a6ff;text-decoration:none} a:hover{text-decoration:underline}
.bar{display:flex;align-items:center;gap:1rem;padding:.6rem 1rem;background:#161b22;
     border-bottom:1px solid #30363d;position:sticky;top:0}
.bar b{color:#e6edf3}
.wrap{padding:1rem}
table{border-collapse:collapse;width:100%}
td,th{padding:4px 10px 4px 0;text-align:left;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}
.cell{border:1px solid #30363d;border-radius:8px;overflow:hidden;background:#161b22}
.cell img{width:100%;display:block;background:#000}
.cell .cap{padding:6px 8px;font-size:11px;color:#8b949e}
.cell .cap b{color:#e6edf3}
.filmstrip{display:flex;gap:8px;overflow-x:auto;padding:1rem;background:#161b22;border-radius:8px}
.filmstrip img{height:340px;border-radius:6px}
pre{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem;overflow-x:auto}
.pager{display:flex;gap:1rem;padding:1rem 0}
.badge{font-size:10px;padding:2px 8px;border-radius:8px;background:#30363d;color:#8b949e}
"""


def _page(title: str, body: str) -> bytes:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{_STYLE}</style></head><body>
<div class="bar"><b>SLS corpus review</b>
<a href="/">sessions</a><a href="/random">🎲 random sample</a>
<span style="margin-left:auto;color:#8b949e">{html.escape(title)}</span></div>
<div class="wrap">{body}</div></body></html>""".encode()


def _render_index(browser: CorpusBrowser, device_filter: str | None) -> bytes:
    rows = []
    for s in browser.sessions():
        device, when = _parse_session(s)
        if device_filter and device != device_filter:
            continue
        rows.append(
            f'<tr><td><a href="/session/{quote(s)}">{html.escape(s)}</a></td>'
            f'<td>{html.escape(device)}</td><td>{html.escape(when)}</td></tr>'
        )
    filt_links = (
        f'<a href="/">all</a> · <a href="/?device=iPhone_XR">iPhone_XR</a> · '
        f'<a href="/?device=iPhone_XR2">iPhone_XR2</a>'
    )
    body = (
        f"<p>{filt_links} &nbsp; ({len(rows)} session(s) shown)</p>"
        f"<table><tr><th>session</th><th>device</th><th>started</th></tr>{''.join(rows)}</table>"
    )
    return _page("sessions", body)


def _render_parks(browser: CorpusBrowser, session: str) -> bytes:
    rows = []
    for p in browser.parks(session):
        n = len(browser.samples(session, p))
        rows.append(
            f'<tr><td><a href="/session/{quote(session)}/{quote(p)}">{html.escape(p)}</a></td>'
            f'<td>{n} sample(s)</td></tr>'
        )
    body = f"<table><tr><th>park</th><th>samples</th></tr>{''.join(rows)}</table>"
    return _page(session, body)


def _render_samples(browser: CorpusBrowser, session: str, park: str, page: int) -> bytes:
    samples = browser.samples(session, park)
    total = len(samples)
    start = page * _PAGE_SIZE
    page_samples = samples[start:start + _PAGE_SIZE]
    # One meta.json fetch per sample is unavoidable for the caption, but fan them
    # out in parallel — sequential round trips make a 48-cell page take a full
    # minute+. The thumbnail always targets frame_000 (every sample has one,
    # whether it has 5 frames or 24) so no per-sample listdir is needed either.
    metas = browser.metas_parallel(session, park, page_samples)
    cells = []
    for s in page_samples:
        meta = metas.get(s, {})
        img_tag = f'<img src="/thumb/{quote(session)}/{quote(park)}/{quote(s)}/frame_000.png" loading="lazy">'
        kind = meta.get("gesture_distribution") or meta.get("kind") or "?"
        cells.append(
            f'<a class="cell" href="/sample/{quote(session)}/{quote(park)}/{quote(s)}">'
            f'{img_tag}<div class="cap"><b>{html.escape(s)}</b><br>{html.escape(str(kind))}</div></a>'
        )
    pages = (total + _PAGE_SIZE - 1) // _PAGE_SIZE
    prev_link = f'<a href="?page={page-1}">&larr; prev</a>' if page > 0 else ""
    next_link = f'<a href="?page={page+1}">next &rarr;</a>' if start + _PAGE_SIZE < total else ""
    pager = (
        f'<div class="pager">{prev_link}'
        f'<span class="badge">page {page+1}/{max(1,pages)} · {total} total</span>'
        f'{next_link}</div>'
    )
    body = pager + f'<div class="grid">{"".join(cells)}</div>' + pager
    return _page(f"{session} / {park}", body)


def _render_sample(browser: CorpusBrowser, session: str, park: str, sample: str) -> bytes:
    meta = browser.meta(session, park, sample)
    # meta.json (a single fast read) carries n_frames on newer manifests — use it
    # to build filenames directly and skip the ~20s-flat listdir tax. Fall back to
    # an actual listdir for older manifests that predate the field.
    n_frames = meta.get("n_frames")
    frames = [f"frame_{i:03d}.png" for i in range(n_frames)] if n_frames else browser.frames(session, park, sample)
    strip = "".join(
        f'<img src="/img/{quote(session)}/{quote(park)}/{quote(sample)}/{quote(f)}" alt="{html.escape(f)}">'
        for f in frames
    )
    body = (
        f'<p><a href="/session/{quote(session)}/{quote(park)}">&larr; back to {html.escape(park)}</a></p>'
        f'<div class="filmstrip">{strip}</div>'
        f'<h3>meta.json</h3><pre>{html.escape(json.dumps(meta, indent=2))}</pre>'
    )
    return _page(f"{session} / {park} / {sample}", body)


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, browser: CorpusBrowser, **kwargs):
        self.browser = browser
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa: N802 — stdlib override
        try:
            self._route()
        except Exception as exc:  # noqa: BLE001 — never let a bad path 500 silently
            self._send(500, f"error: {exc!r}".encode(), "text/plain")

    def _route(self):
        split = urlsplit(self.path)
        parts = [unquote(p) for p in split.path.split("/") if p]
        qs = parse_qs(split.query)
        b = self.browser

        if not parts:
            device = qs.get("device", [None])[0]
            self._send(200, _render_index(b, device), "text/html; charset=utf-8")
        elif parts[0] == "random":
            picked = b.random_sample()
            if picked is None:
                self._send(404, b"corpus is empty", "text/plain")
                return
            session, park, sample = picked
            self.send_response(302)
            self.send_header("Location", f"/sample/{quote(session)}/{quote(park)}/{quote(sample)}")
            self.end_headers()
        elif parts[0] == "session" and len(parts) == 2:
            self._send(200, _render_parks(b, parts[1]), "text/html; charset=utf-8")
        elif parts[0] == "session" and len(parts) == 3:
            page = int(qs.get("page", ["0"])[0] or 0)
            self._send(200, _render_samples(b, parts[1], parts[2], page), "text/html; charset=utf-8")
        elif parts[0] == "sample" and len(parts) == 4:
            self._send(200, _render_sample(b, *parts[1:]), "text/html; charset=utf-8")
        elif parts[0] == "img" and len(parts) == 5:
            data = b.frame_bytes(*parts[1:])
            self._send(200, data, "image/png", cache=True)
        elif parts[0] == "thumb" and len(parts) == 5:
            data = b.thumb_bytes(*parts[1:])
            self._send(200, data, "image/jpeg", cache=True)
        else:
            self._send(404, b"not found", "text/plain")

    def _send(self, code: int, body: bytes, content_type: str, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Browse the offloaded SLS corpus on Modal.")
    ap.add_argument("--volume", default="trueskate-corpus")
    ap.add_argument("--cache-dir", type=Path, default=_REPO_ROOT / "tmp" / "modal_corpus_cache")
    ap.add_argument("--port", type=int, default=8410)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    browser = CorpusBrowser(args.volume, args.cache_dir)
    handler = functools.partial(_Handler, browser=browser)
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as e:
        sys.exit(f"Cannot bind {args.host}:{args.port} ({e.strerror}) — already running?")
    print(f"Corpus review: http://127.0.0.1:{args.port}/")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
