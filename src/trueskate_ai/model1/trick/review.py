"""Operator spot check of admitted clips, verdicts and the batch summary (laptop).

The spot check reuses the curved audit review UI (curved/review.py and review.html): a
temporary 127.0.0.1 server shows each picked clip's native frames, keyboard-first, with a
ring at the commanded position of whichever stroke is active, so gross faults show: a
missing or extra touch, a trail far from its ring, a menu. The reviewer exports a JSON
report and sends its path; importing it records the verdicts and closes the server.
"""
from __future__ import annotations

from collections import Counter
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import signal
import statistics
import subprocess
import tempfile
import threading
from urllib.parse import urlparse

import numpy as np

from trueskate_ai.model1.curved.capture import file_sha256
from trueskate_ai.model1.curved.contract import digest, save_json
from trueskate_ai.model1.curved.review import FRAMES, FrameCache
from .contract import EXPERIMENT, SEED

DEFAULT_COUNT = 20
RING_PT = 10
VERDICTS = dict(K="ok", F="fault", U="unsure")
TOKEN = re.compile(r"(\d{1,2})\s*([KFU])", re.IGNORECASE)


def _read(path):
    return json.loads(Path(path).read_text())


def admitted_clips(root) -> list[dict]:
    """Clips whose recording admission lists them, in recording and run order."""
    clips = []
    for admission in sorted(Path(root).glob("recordings/*/admission.json")):
        result = _read(admission)
        if not result.get("accepted"):
            continue
        for number in sorted(result["admitted"]):
            folder = admission.parent/f"run-{number:05d}"
            meta = _read(folder/"meta.json")
            clips.append(dict(recording=admission.parent.name, run=number, folder=folder, meta=meta))
    return clips


def select(root, count: int = DEFAULT_COUNT) -> dict:
    """A seeded random pick of admitted clips with two-digit codes, kept in review/selection.json."""
    if not 1 <= count <= 99:
        raise ValueError("pick 1–99 clips")
    clips = admitted_clips(root)
    if not clips:
        raise ValueError("no admitted clips to review")
    picked = sorted(np.random.default_rng([SEED, 1]).choice(len(clips), min(count, len(clips)), replace=False))
    rows = [dict(code=f"{i:02d}", recording=clips[j]["recording"], run=clips[j]["run"],
                 clip_sha256=clips[j]["meta"]["clip_sha256"]) for i, j in enumerate(picked, 1)]
    selection = dict(experiment=EXPERIMENT, seed=[SEED, 1], admitted=len(clips), clips=rows)
    selection["sha256"] = digest(selection)
    return selection


def load_selection(root) -> dict:
    selection = _read(Path(root)/"review"/"selection.json")
    if digest({k: v for k, v in selection.items() if k != "sha256"}) != selection.get("sha256"):
        raise ValueError("review selection changed after it was written")
    return selection


def review_manifest(root, *, count: int = DEFAULT_COUNT, reselect: bool = False) -> tuple[dict, dict]:
    """The page's manifest and each picked clip's folder; the pick is kept so codes stay valid."""
    root = Path(root)
    out = root/"review"
    out.mkdir(exist_ok=True)
    path = out/"selection.json"
    if path.exists() and not reselect:
        selection = load_selection(root)
    else:
        if (out/"verdicts.json").exists():
            raise ValueError("verdicts already recorded for the current pick; keep it")
        selection = select(root, count)
        save_json(path, selection, exclusive=False)
    samples, folders = [], {}
    for row in selection["clips"]:
        folder = root/"recordings"/row["recording"]/f'run-{row["run"]:05d}'
        meta = _read(folder/"meta.json")
        if meta["clip_sha256"] != row["clip_sha256"] or file_sha256(folder/"clip.mp4") != row["clip_sha256"]:
            raise ValueError(f'clip {row["code"]} changed after it was picked')
        if len(meta["frame_times_s"]) != FRAMES:
            raise ValueError(f'clip {row["code"]} does not hold {FRAMES} native frames')
        identity = int(row["code"])
        folders[identity] = folder
        strokes = [{k: s[k] for k in ("family", "onset_s", "release_s", "times_ms", "command_points")}
                   for s in meta["strokes"]]
        samples.append(dict(id=identity, code=row["code"], recording=row["recording"], run=row["run"],
                            run_kind=meta.get("run_kind"), park=meta.get("park"), device=meta.get("device"),
                            clock_rate=meta["clock_rate"], frame_times_s=meta["frame_times_s"], strokes=strokes))
    manifest = dict(experiment=EXPERIMENT, sha256=selection["sha256"], ring_pt=RING_PT, samples=samples)
    return manifest, folders


def serve(root, *, count: int = DEFAULT_COUNT, reselect: bool = False, port: int = 8766, open_browser=False):
    """Serve the spot check until its report is imported (review/verdicts.json exists) or SIGINT/SIGTERM."""
    root = Path(root).resolve()
    done = root/"review"/"verdicts.json"
    if done.exists():
        raise ValueError("verdicts already recorded for this pick; the summary shows them")
    manifest, folders = review_manifest(root, count=count, reselect=reselect)
    page = Path(__file__).with_name("review.html").read_bytes()
    body = json.dumps(manifest).encode()
    with tempfile.TemporaryDirectory(prefix="spot-check-", dir=root) as temporary:
        cache = FrameCache(Path(temporary), folders)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                route = urlparse(self.path).path
                try:
                    if route == "/":
                        data, mime, caching = page, "text/html; charset=utf-8", "no-store"
                    elif route == "/manifest":
                        data, mime, caching = body, "application/json", "no-store"
                    elif route.startswith("/frame/"):
                        _, _, identity, index = route.split("/")
                        # URLs carry the pick's hash as a query, so frames are immutable.
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
        print(f"Spot check: {url} (temporary; export the review, then import it)", flush=True)
        if open_browser:
            subprocess.run(["open", "-a", "Google Chrome", url], check=True)

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")

        # Background jobs inherit an ignored SIGINT; handle both so the frame cache is removed.
        previous = {}
        if threading.current_thread() is threading.main_thread():
            previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            while not done.exists():
                server.handle_request()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            for s, handler in previous.items():
                signal.signal(s, handler)


def _write_verdicts(root: Path, selection: dict, checked: list, *, replace: bool, source: str) -> int:
    """Write (code, verdict, reason) rows once all are checked; never silently replace one."""
    path = root/"review"/"verdicts.json"
    recorded = _read(path) if path.exists() else dict(selection_sha256=selection["sha256"], verdicts={})
    if recorded["selection_sha256"] != selection["sha256"]:
        raise ValueError("recorded verdicts belong to another pick")
    for key, _, _ in checked:
        if key in recorded["verdicts"] and not replace:
            raise ValueError(f"clip {key} already has a verdict; pass --replace to change it")
    rows = {row["code"]: row for row in selection["clips"]}
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for key, code, reason in checked:
        row = rows[key]
        recorded["verdicts"][key] = dict(code=code, verdict=VERDICTS[code], reason=reason, recording=row["recording"],
                                         run=row["run"], clip_sha256=row["clip_sha256"], recorded_at=now,
                                         source=source)
    save_json(path, recorded, exclusive=False)
    return len(checked)


def record_verdicts(root, codes: str, *, replace: bool = False) -> int:
    """Operator verdicts as `01K 02F 03U` (OK, Fault, Unsure); every token is checked before any is written."""
    root = Path(root)
    selection = load_selection(root)
    rows = {row["code"] for row in selection["clips"]}
    tokens = TOKEN.findall(codes)
    if not tokens or TOKEN.sub("", codes).replace(",", " ").split():
        raise ValueError('codes look like "01K 02F 03U" (K OK, F fault, U unsure)')
    checked, seen = [], set()
    for number, code in tokens:
        key = f"{int(number):02d}"
        if key not in rows:
            raise ValueError(f"clip {key} is not in the pick")
        if key in seen:
            raise ValueError(f"clip {key} is listed twice")
        seen.add(key)
        checked.append((key, code.upper(), ""))
    return _write_verdicts(root, selection, checked, replace=replace, source="codes")


def import_report(root, report, *, replace: bool = False) -> int:
    """Record the page's exported review: the whole pick, with what was seen for every Fault or Unsure."""
    root = Path(root)
    selection = load_selection(root)
    exported = _read(report)
    if exported.get("selection_sha256") != selection["sha256"]:
        raise ValueError("the report belongs to another pick")
    rows = [row["code"] for row in selection["clips"]]
    judged = {}
    for clip in exported.get("clips", []):
        key, code, reason = clip.get("code"), clip.get("verdict"), str(clip.get("reason") or "").strip()
        if key not in rows or key in judged:
            raise ValueError(f"clip {key} is not in the pick or is listed twice")
        if code not in VERDICTS:
            raise ValueError(f"clip {key} has no OK, Fault or Unsure verdict")
        if code != "K" and not reason:
            raise ValueError(f"clip {key} needs what was seen")
        judged[key] = (key, code, reason)
    missing = [key for key in rows if key not in judged]
    if missing:
        raise ValueError(f"the report misses clip(s) {' '.join(missing)}")
    return _write_verdicts(root, selection, [judged[key] for key in rows], replace=replace,
                           source=Path(report).name)


def summarize(root) -> dict:
    """Recordings, runs, admission, stroke mix and spot-check counts for one run directory."""
    root = Path(root)
    recordings, reasons, families, durations, touches = {}, Counter(), Counter(), [], Counter()
    totals, states, lags = Counter(), Counter(), []
    for execution_path in sorted(root.glob("recordings/*/execution.json")):
        folder = execution_path.parent
        planned, execution = _read(folder/"planned.json"), _read(execution_path)
        sent = [e for e in execution["events"] if e["kind"] == "stroke" and not e.get("skipped")]
        row = dict(runs_planned=len(planned["runs"]), runs_sent=len({e["run"] for e in sent}),
                   strokes_sent=len(sent), strokes_dropped=sum(bool(e.get("skipped")) for e in execution["events"]),
                   clips_admitted=0, error=execution["error"])
        if execution["error"]:
            row["state"] = "recording failed"
        elif not (folder/"admission.json").exists():
            row["state"] = "awaiting admission"
        else:
            admission = _read(folder/"admission.json")
            row["state"] = "admitted" if admission["accepted"] else "admission failed"
            row["clips_admitted"] = len(admission.get("admitted", []))
            row["error"] = admission.get("error")
            reasons.update(admission.get("rejected", {}).values())
            # Visible minus estimated stroke start (diagnostic; positive means the trail shows later).
            lags += (admission.get("calibration") or {}).get("stroke_visible_lags_s", {}).values()
        recordings[folder.name] = row
        totals.update({k: row[k] for k in ("runs_planned", "runs_sent", "strokes_sent", "strokes_dropped",
                                           "clips_admitted")})
        totals["recordings"] += 1
        states[row["state"]] += 1
    for clip in admitted_clips(root):
        touches[len(clip["meta"]["strokes"])] += 1
        for stroke in clip["meta"]["strokes"]:
            families[stroke["family"]] += 1
            durations.append(stroke["times_ms"][-1])
    spot = Counter()
    if (root/"review"/"verdicts.json").exists():
        spot.update(v["code"] for v in _read(root/"review"/"verdicts.json")["verdicts"].values())
    picked = len(load_selection(root)["clips"]) if (root/"review"/"selection.json").exists() else 0
    sent = totals["runs_sent"]
    return dict(experiment=EXPERIMENT, totals=dict(totals), recordings_by_state=dict(states),
                admitted_share_of_sent_runs=totals["clips_admitted"]/sent if sent else None,
                rejection_reasons=dict(reasons.most_common()), stroke_families=dict(families),
                touches_per_clip={str(k): v for k, v in sorted(touches.items())},
                stroke_ms=dict(min=min(durations), median=statistics.median(durations), max=max(durations),
                               share_under_200=sum(d < 200 for d in durations)/len(durations)) if durations else None,
                spot_check=dict(picked=picked, ok=spot["K"], fault=spot["F"], unsure=spot["U"],
                                unjudged=picked-sum(spot.values())),
                stroke_visible_lag_s=dict(measured=len(lags), median=statistics.median(lags),
                                          p10=float(np.percentile(lags, 10)), p90=float(np.percentile(lags, 90)))
                if lags else None,
                recordings=recordings)


def summary_markdown(summary: dict) -> str:
    t, spot = summary["totals"], summary["spot_check"]
    share = summary["admitted_share_of_sent_runs"]
    states = ", ".join(f"{state} {n}" for state, n in sorted(summary["recordings_by_state"].items())) or "none"
    lines = [f"# {summary['experiment']}", "",
             f"- Recordings: {t.get('recordings', 0)} ({states}).",
             f"- Clips admitted: {t.get('clips_admitted', 0)}"
             f"{'' if share is None else f' ({share:.0%} of runs sent)'}.",
             f"- Runs planned {t.get('runs_planned', 0)}, sent {t.get('runs_sent', 0)}; strokes sent "
             f"{t.get('strokes_sent', 0)}, dropped at run time {t.get('strokes_dropped', 0)}.",
             f"- Spot check: {spot['picked']} picked, {spot['ok']} OK, {spot['fault']} fault, "
             f"{spot['unsure']} unsure, {spot['unjudged']} unjudged."]
    if summary["stroke_ms"]:
        s = summary["stroke_ms"]
        lines.append(f"- Stroke durations: {s['min']}–{s['max']} ms, median {s['median']:g} ms, "
                     f"{s['share_under_200']:.0%} under 200 ms.")
    if summary.get("stroke_visible_lag_s"):
        g = summary["stroke_visible_lag_s"]
        lines.append(f"- Trail shows {g['median']*1000:+.0f} ms after the estimated stroke start (median of "
                     f"{g['measured']}; 10–90%: {g['p10']*1000:+.0f} to {g['p90']*1000:+.0f} ms). Labels unchanged.")
    if summary["rejection_reasons"]:
        lines += ["", "| Rejection reason | Runs |", "|---|---:|"]
        lines += [f"| {reason} | {n} |" for reason, n in summary["rejection_reasons"].items()]
    return "\n".join(lines) + "\n"
