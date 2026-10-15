"""Kickflip position tolerance on XR1 (explore mode): prepare, run, admit, review, summarize.

Jittered copies of the approved kickflip recipe run through the repeatability runner's
execution path and live guards (`probe_kickflip_repeatability.live_batch`, tolerance mode).
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))

from trueskate_ai.research import kickflip_tolerance as tolerance

# The rig's shared "recording in progress" flag (RIG-BOARD): other sessions avoid heavy work while it exists.
BUSY_LOCK = Path('/Users/training-server/trueskate-ai-runtime/tmp/RIG-BUSY-curved-recording')


def load_runner():
    spec = importlib.util.spec_from_file_location('kickflip_runner', REPO / 'scripts/collection/probe_kickflip_repeatability.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def run(root, *, env_file, ready_note, settings_note, appium_port, after_failure_reason, lock):
    """Hold the busy lock for the recordings; refuse to start if anyone else holds it."""
    if 'training-server' not in socket.gethostname():
        raise ValueError('live runs belong on training-server, not laptop USB')
    stamp = dict(owner=f'{tolerance.EXPERIMENT} position-tolerance run', pid=os.getpid(),
                 started=dt.datetime.now(dt.timezone.utc).isoformat(), out=str(root))
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'{lock} exists: another recording holds the rig; no start attempted') from None
    with os.fdopen(fd, 'w') as fh:
        json.dump(stamp, fh)
    try:
        load_runner().live_batch(root, candidate_id=None, repeatability=False, tolerance=True, env_file=env_file,
                                 ready_note=ready_note, settings_note=settings_note, appium_port=appium_port,
                                 after_failure_reason=after_failure_reason)
    finally:
        try:
            if json.loads(Path(lock).read_text()).get('pid') == os.getpid():
                Path(lock).unlink()
        except (OSError, ValueError):
            pass


def status(root):
    plan = tolerance.load_plan(root)
    current = tolerance.states(root, plan)
    pending = [n for n, s in current.items() if s == 'pending']
    return dict(experiment=plan['experiment'], attempts=len(plan['attempts']), states=dict(Counter(current.values())),
                next_attempt=pending[0] if pending else None,
                not_runnable={n: s for n, s in current.items() if s in (*tolerance.FAILED_STATES, 'admission failed')},
                levels=plan['levels'], per_level=plan['per_level'], seed=plan['seed'],
                source_commit=plan['source_commit'], wda_revision=plan['wda_revision'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='absolute experiment directory')
    sub = parser.add_subparsers(dest='command', required=True)
    preparation = sub.add_parser('prepare', help='write the 40-attempt plan (offline, deterministic)')
    preparation.add_argument('--source-commit', help='commit this code was staged from (default: git HEAD, if any)')
    sub.add_parser('status')
    live = sub.add_parser('run', help='run every pending attempt in plan order on XR1 (rig only)')
    live.add_argument('--env-file', type=Path, required=True)
    live.add_argument('--operator-ready-note', required=True)
    live.add_argument('--settings-note', required=True, help='fixed waypoint, stance, camera and physics settings')
    live.add_argument('--appium-port', type=int, default=4725,
                      help='session-discovery-enabled Appium; the supervised 4723 lacks it, so use the temporary 4725')
    live.add_argument('--after-failure-reason', help='continue after a technical failure; failed attempts are never rerun')
    live.add_argument('--busy-lock', type=Path, default=BUSY_LOCK)
    admission = sub.add_parser('admit', help='admit every recorded attempt (laptop; same gates as the repeatability run)')
    admission.add_argument('--jobs', type=int, default=4)
    review = sub.add_parser('review', help='contact sheets and a static review page for admitted attempts')
    review.add_argument('--open', action='store_true', help='open the page in Google Chrome (laptop)')
    verdicts = sub.add_parser('verdicts', help='record operator verdicts, e.g. "01L 02F 03U"')
    verdicts.add_argument('--codes', required=True)
    verdicts.add_argument('--replace', action='store_true', help='change verdicts already recorded')
    sub.add_parser('report', help='per-distance landing counts (writes summary.json and summary.md)')
    paths = sub.add_parser('render-paths', help='draw the planned pop/flick paths per distance (offline check)')
    paths.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error('--root must be absolute')
    root = args.root
    if args.command == 'prepare':
        commit = args.source_commit
        if commit is None:
            head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True)
            commit = head.stdout.strip() or None
        plan = tolerance.prepare(root, REPO, source_commit=commit)
        redraws = [a['candidate']['jitter'][k]['redraws'] for a in plan['attempts'] for k in ('pop', 'flick')]
        print(json.dumps(dict(plan=str(root / 'plan.json'), attempts=len(plan['attempts']),
                              order=[a['level'] for a in plan['attempts']], redraws_total=sum(redraws),
                              redraws_max=max(redraws), source_commit=commit)))
    elif args.command == 'status':
        print(json.dumps(status(root), indent=2))
    elif args.command == 'run':
        if not args.env_file.is_absolute():
            parser.error('--env-file must be absolute')
        run(root, env_file=args.env_file, ready_note=args.operator_ready_note, settings_note=args.settings_note,
            appium_port=args.appium_port, after_failure_reason=args.after_failure_reason, lock=args.busy_lock)
    elif args.command == 'admit':
        results = tolerance.admit_all(root, jobs=args.jobs)
        print(json.dumps(results, indent=2))
        print(f"admitted {sum(r['admitted'] for r in results.values())} of {len(results)} recorded attempts")
    elif args.command == 'review':
        page = tolerance.review_page(root)
        print(page)
        if args.open:
            subprocess.run(['open', '-a', 'Google Chrome', str(page)], check=True)
    elif args.command == 'verdicts':
        print(f'recorded {tolerance.record_verdicts(root, args.codes, replace=args.replace)} verdict(s)')
    elif args.command == 'report':
        summary = tolerance.summarize(root)
        (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        markdown = tolerance.summary_markdown(summary)
        (root / 'summary.md').write_text(markdown)
        print(markdown)
    else:
        for path in tolerance.render_paths(tolerance.load_plan(root), args.out):
            print(path)


if __name__ == '__main__':
    main()
