"""Bounded curved Model 1 qualification/production coordinator (no background jobs)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
from contextlib import nullcontext, ExitStack

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"src"))
from trueskate_ai.model1.curved.ledger import Ledger, pipeline_fingerprint, operation_lock


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="isolated persistent run directory")
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--parks", type=Path, default=REPO/"config/parks.json")
    sub.add_parser("status")
    for name in ("start-audit", "prepare-production"):
        sub.add_parser(name).add_argument("--wda-revision", required=True)
    sub.add_parser("fail-audit").add_argument("--reason", required=True)
    seal = sub.add_parser("seal-audit")
    seal.add_argument("--round", type=int, required=True)
    sub.add_parser("import-review").add_argument("report", type=Path)
    sub.add_parser("freeze-dataset")
    sub.add_parser("freeze-human").add_argument("clips_json", type=Path)
    view = sub.add_parser("view-audit")
    view.add_argument("--round", type=int, required=True)
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--open", action="store_true")
    rec = sub.add_parser("reconcile")
    rec.add_argument("--abandon-unresolved", action="store_true")
    live = sub.add_parser("collect")
    live.add_argument("--session", required=True)
    live.add_argument("--park", required=True, help="verbatim current park, verified by operator")
    live.add_argument("--device", choices=("iPhone_XR", "iPhone_XR2"), required=True)
    live.add_argument("--wda-revision", required=True)
    live.add_argument("--max-minutes", type=int, required=True)
    live.add_argument("--env-file", type=Path, required=True)
    live.add_argument("--waypoint", type=int, choices=range(1, 6), required=True)
    live.add_argument("--settings-note", required=True, help="current waypoint and fixed game settings")
    a = p.parse_args(argv)
    if not a.root.is_absolute():
        p.error("--root must be absolute")
    if a.command == "init":
        ledger = Ledger.initialize(a.root, json.loads(a.parks.read_text())["parks"])
    else:
        if not (a.root/"ledger.sqlite").is_file():
            p.error("run does not exist; initialize once, reuse it for every round")
        ledger = Ledger(a.root)
    lock = (nullcontext() if a.command in ("init", "status", "view-audit") else operation_lock(a.root))
    stack = ExitStack()
    try:
        stack.enter_context(lock)
        if a.command in ("init", "status"):
            result = ledger.status()
        elif a.command == "start-audit":
            result = dict(round=ledger.start_round(pipeline_fingerprint(a.wda_revision)))
        elif a.command == "prepare-production":
            ledger.production_sessions(pipeline_fingerprint(a.wda_revision))
            result = ledger.sessions(phase="production")
        elif a.command == "fail-audit":
            ledger.fail_round(a.reason)
            result = ledger.status()
        elif a.command == "seal-audit":
            result = ledger.audit_manifest(a.round)
        elif a.command == "import-review":
            result = ledger.import_review(json.loads(a.report.read_text()))
        elif a.command == "freeze-dataset":
            from trueskate_ai.model1.curved.audit import freeze_dataset
            result = freeze_dataset(ledger)
            result = dict(path=str(a.root/"dataset.json"), sha256=result["sha256"])
        elif a.command == "freeze-human":
            from trueskate_ai.model1.curved.audit import freeze_human
            result = freeze_human(a.clips_json, a.root/"human.json", ledger.definition["parks"])
        elif a.command == "view-audit":
            # review.py replaces viewer.py; the latter stays byte-identical for the fingerprint.
            from trueskate_ai.model1.curved.review import serve
            serve(a.root, a.round, port=a.port, open_browser=a.open)
            result = dict(viewer="stopped")
        elif a.command == "reconcile":
            from trueskate_ai.model1.curved.collection import reconcile
            result = reconcile(ledger, abandon=a.abandon_unresolved)
        else:
            from trueskate_ai.model1.curved.collection import collect
            def interrupted(signum, frame):
                raise KeyboardInterrupt(f"signal {signum}")
            signal.signal(signal.SIGTERM, interrupted)
            collect(ledger, session_id=a.session, park=a.park, device=a.device,
                    revision=a.wda_revision, max_minutes=a.max_minutes, env_file=a.env_file,
                    waypoint=a.waypoint, settings_note=a.settings_note)
            result = ledger.status()
        print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        stack.close()
        ledger.close()


if __name__ == "__main__":
    main()
