"""Trick-speed Model 1 batch on XR1 (explore mode): record, admit, spot-check, summarize.

`run` records on the rig and holds the shared busy lock; every other step runs on the
laptop against the fetched run directory. See research/protocols/trick_speed_batch.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))

from trueskate_ai.model1.trick import admission, live, review
from trueskate_ai.model1.trick.contract import EXPERIMENT

# The rig's shared "recording in progress" flag (RIG-BOARD): other sessions avoid heavy work while it exists.
BUSY_LOCK = Path('/Users/training-server/trueskate-ai-runtime/tmp/RIG-BUSY-curved-recording')


def run(args):
    live.check_arguments(args.root, minutes=args.minutes, env_file=args.env_file, ready_note=args.operator_ready_note,
                         settings_note=args.settings_note, park=args.park, device=args.device)
    live.require_rig()  # before the lock: its default path exists only on the rig

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)  # stop and retrieve the current recording, then release
    with live.busy_lock(args.busy_lock, owner=f'{EXPERIMENT} trick-speed capture on {args.device}', out=args.root):
        live.collect(args.root, minutes=args.minutes, env_file=args.env_file, ready_note=args.operator_ready_note,
                     settings_note=args.settings_note, park=args.park, appium_port=args.appium_port,
                     device=args.device, source_commit=args.source_commit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='absolute run directory')
    sub = parser.add_subparsers(dest='command', required=True)
    live_run = sub.add_parser('run', help='record N one-minute recordings on XR1 (rig only)')
    live_run.add_argument('--env-file', type=Path, required=True)
    live_run.add_argument('--operator-ready-note', required=True)
    live_run.add_argument('--settings-note', required=True, help='fixed waypoint, stance, camera and physics settings')
    live_run.add_argument('--park', required=True, help="the park the operator confirmed, verbatim")
    live_run.add_argument('--minutes', type=int, required=True, help='one-minute recordings to make (1–60)')
    live_run.add_argument('--device', choices=live.PHONES, default='iPhone_XR',
                          help='XR2 runs alongside an XR1 capture only when its owner agrees; give it its own --busy-lock')
    live_run.add_argument('--appium-port', type=int, default=4725,
                          help='session-discovery-enabled Appium; the supervised 4723 lacks it, so use the temporary 4725')
    live_run.add_argument('--source-commit', help='commit this code was staged from')
    live_run.add_argument('--busy-lock', type=Path, default=BUSY_LOCK)
    sub.add_parser('status', help='recordings, runs and strokes so far (rig or laptop)')
    admit = sub.add_parser('admit', help='calibrate fetched recordings and cut one clip per run (laptop)')
    admit.add_argument('--jobs', type=int, default=4)
    page = sub.add_parser('review', help='serve the spot check of a seeded pick of admitted clips until it is imported')
    page.add_argument('--count', type=int, default=review.DEFAULT_COUNT)
    page.add_argument('--reselect', action='store_true', help='make a new pick (refused once verdicts exist)')
    page.add_argument('--port', type=int, default=8766)
    page.add_argument('--open', action='store_true', help='open the page in Google Chrome (laptop)')
    verdicts = sub.add_parser('verdicts', help="import the page's exported review, or record codes like \"01K 02F\"")
    source = verdicts.add_mutually_exclusive_group(required=True)
    source.add_argument('--report', type=Path, help='the JSON file the spot-check page exported')
    source.add_argument('--codes', help='OK/Fault/Unsure codes when the page is unavailable')
    verdicts.add_argument('--replace', action='store_true', help='change verdicts already recorded')
    sub.add_parser('summary', help='write summary.json and summary.md')
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error('--root must be absolute')
    root = args.root
    if args.command == 'run':
        if not args.env_file.is_absolute():
            parser.error('--env-file must be absolute')
        run(args)
    elif args.command == 'status':
        summary = review.summarize(root)
        print(json.dumps(dict(totals=summary['totals'], recordings_by_state=summary['recordings_by_state'],
                              failures={k: v['error'] for k, v in summary['recordings'].items() if v['error']}),
                         indent=2))
    elif args.command == 'admit':
        results = admission.admit_all(root, jobs=args.jobs)
        print(json.dumps(results, indent=2))
        print(f"admitted {sum(len(r['admitted']) for r in results.values())} clips from {len(results)} recording(s)")
    elif args.command == 'review':
        review.serve(root, count=args.count, reselect=args.reselect, port=args.port, open_browser=args.open)
    elif args.command == 'verdicts':
        if args.report:
            recorded = review.import_report(root, args.report, replace=args.replace)
        else:
            recorded = review.record_verdicts(root, args.codes, replace=args.replace)
        print(f'recorded {recorded} verdict(s)')
    else:
        summary = review.summarize(root)
        (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        markdown = review.summary_markdown(summary)
        (root / 'summary.md').write_text(markdown)
        print(markdown)


if __name__ == '__main__':
    main()
