"""Durable qualification state and a global, crash-safe dispatch budget."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sqlite3
from contextlib import contextmanager


@contextmanager
def operation_lock(root):
    """Serialize run mutations across processes, including the whole live invocation."""
    import fcntl
    with (Path(root)/".operation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another operation is already using this collection run") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

from .contract import SAMPLERS, SCHEMA, SEED, digest, save_json, select_parks, settings
from .sampling import sample_gesture


def pipeline_fingerprint(wda_revision: str) -> str:
    here = Path(__file__).parent
    package = here.parents[1]
    files = [here/name for name in ("contract.py", "sampling.py", "capture.py",
                                    "collection.py", "ledger.py", "audit.py", "viewer.py", "viewer.html")]
    files += [package/name for name in (
        "sim/timed_waypoints.py", "data/control_hitboxes.py",
        "collection/xctest_capture.py", "collection/wda_action_timing.py",
        "collection/die_five_calibration.py", "collection/tap_timing_calibration.py",
        "collection/gameplay_filter.py", "research/audit_execution.py",
        "research/audit_calibration.py", "research/audit_video.py", "research/curved_audit.py",
        "sim/device.py")]
    return digest(dict(settings=settings(), wda_revision=wda_revision,
                       sources={p.name: __import__("hashlib").sha256(p.read_bytes()).hexdigest()
                                for p in files}))


class Ledger:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.db = sqlite3.connect(self.root/"ledger.sqlite", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rounds(
            number INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL,
            status TEXT NOT NULL, manifest_hash TEXT, review_json TEXT);
        CREATE TABLE IF NOT EXISTS sessions(
            id TEXT PRIMARY KEY, park TEXT NOT NULL, phase TEXT NOT NULL,
            round INTEGER, split TEXT, target INTEGER NOT NULL,
            FOREIGN KEY(round) REFERENCES rounds(number));
        CREATE TABLE IF NOT EXISTS attempts(
            id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL,
            sampler TEXT NOT NULL, status TEXT NOT NULL, gesture_json TEXT NOT NULL,
            command_hash TEXT UNIQUE NOT NULL, sample_path TEXT, reason TEXT,
            FOREIGN KEY(session) REFERENCES sessions(id));
        """)
        self.db.commit()

    @classmethod
    def initialize(cls, root: Path, eligible: list[str], *, seed: int = SEED):
        root = Path(root)
        parks = select_parks(eligible, seed)
        root.mkdir(parents=True, exist_ok=False)
        ledger = cls(root)
        value = dict(settings(), seed=seed, eligible_parks=sorted(set(eligible)),
                     parks=parks)
        ledger.set("definition", value)
        save_json(root/"definition.json", value)
        return ledger

    def close(self):
        self.db.close()

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def set(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO config VALUES(?,?)",
                            (key, json.dumps(value, allow_nan=False)))

    @property
    def definition(self):
        value = self.get("definition")
        if value is None:
            raise ValueError("uninitialized curved run")
        return value

    @property
    def spent(self):
        return self.db.execute(
            "SELECT count(*) FROM attempts WHERE status IN ('dispatched','admitted','rejected')"
        ).fetchone()[0]

    def start_round(self, fingerprint: str) -> int:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.get("qualified"):
                raise ValueError("pipeline already qualified")
            row = self.db.execute("SELECT * FROM rounds ORDER BY number DESC LIMIT 1").fetchone()
            if row and row["status"] in ("collecting", "review"):
                if row["fingerprint"] != fingerprint:
                    raise ValueError("finish current audit before changing its pipeline")
                return row["number"]
            number = 1 if row is None else row["number"]+1
            if number > 5:
                raise ValueError("five audit rounds failed; redesign required")
            if row and row["fingerprint"] == fingerprint:
                raise ValueError("failed audit requires diagnosis and a revised pipeline")
            self.db.execute("INSERT INTO rounds VALUES(?,?,'collecting',NULL,NULL)",
                            (number, fingerprint))
            for park in self.definition["parks"]:
                session = f"audit-{number}-{digest(park)[:12]}"
                self.db.execute("INSERT INTO sessions VALUES(?,?,'audit',?,NULL,20)",
                                (session, park, number))
        return number

    def production_sessions(self, fingerprint: str):
        qualified = self.get("qualified")
        if not qualified or qualified["fingerprint"] != fingerprint:
            raise ValueError("bulk collection requires the exact qualified pipeline")
        with self.db:
            for i, park in enumerate(self.definition["parks"]):
                splits = ["train"]*8+["validation", "test"]
                random.Random(self.definition["seed"]+i).shuffle(splits)
                for index, split in enumerate(splits):
                    self.db.execute("INSERT OR IGNORE INTO sessions VALUES(?,?,'production',NULL,?,280)",
                                    (f"production-{digest(park)[:12]}-{index:02}", park, split))

    def sessions(self, *, phase=None, park=None):
        query, parameters = "SELECT * FROM sessions WHERE 1=1", []
        if phase:
            query += " AND phase=?"
            parameters.append(phase)
        if park:
            query += " AND park=?"
            parameters.append(park)
        query += " ORDER BY rowid"
        return [dict(r) for r in self.db.execute(query, parameters)]

    def admitted(self, session: str | None = None):
        query = "SELECT * FROM attempts WHERE status='admitted'"
        args = ()
        if session:
            query += " AND session=?"
            args = (session,)
        return [dict(row) for row in self.db.execute(query+" ORDER BY id", args)]

    def session(self, session_id):
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise ValueError("unknown collection session")
        return dict(row)

    def remaining(self, session_id, sampler, *, include_pending=True):
        session = self.session(session_id)
        statuses = ("admitted", "planned", "dispatched") if include_pending else ("admitted",)
        count = self.db.execute(
            f"SELECT count(*) FROM attempts WHERE session=? AND sampler=? AND status IN ({','.join('?'*len(statuses))})",
            (session_id, sampler, *statuses)).fetchone()[0]
        return session["target"]//2-count

    def plan_candidate(self, session_id: str, sampler: str):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            session = self.session(session_id)
            if sampler not in SAMPLERS or self.remaining(session_id, sampler) <= 0:
                raise ValueError("session sampler quota filled")
            if session["phase"] == "audit":
                status = self.db.execute("SELECT status FROM rounds WHERE number=?",
                                         (session["round"],)).fetchone()[0]
                if status != "collecting":
                    raise ValueError("audit is sealed")
            elif not self.get("qualified"):
                raise ValueError("unqualified production")
            if self.spent >= self.definition["attempt_limit"]:
                raise ValueError("20,000-attempt ceiling reached")
            ordinal = self.db.execute("SELECT coalesce(max(id),0)+1 FROM attempts").fetchone()[0]
            gesture = sample_gesture(self.definition["seed"]+ordinal*1009, sampler).to_dict()
            cursor = self.db.execute(
                "INSERT INTO attempts(session,sampler,status,gesture_json,command_hash) VALUES(?,?,'planned',?,?)",
                (session_id, sampler, json.dumps(gesture), gesture["payload_sha256"]))
            return dict(id=cursor.lastrowid, session=session_id, gesture=gesture)

    def dispatch(self, candidate_id: int):
        """Commit spending BEFORE the RPC. A timed-out dispatch is never replayed."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.spent >= self.definition["attempt_limit"]:
                raise ValueError("20,000-attempt ceiling reached")
            row = self.db.execute("SELECT * FROM attempts WHERE id=?", (candidate_id,)).fetchone()
            if row is None or row["status"] != "planned":
                raise ValueError("candidate already dispatched or cancelled")
            self.db.execute("UPDATE attempts SET status='dispatched' WHERE id=?", (candidate_id,))

    def settle(self, candidate_id: int, *, sample_path: Path | None = None, reason=None):
        status = "admitted" if sample_path is not None else "rejected"
        path = str(Path(sample_path).resolve()) if sample_path is not None else None
        with self.db:
            row = self.db.execute("SELECT * FROM attempts WHERE id=?", (candidate_id,)).fetchone()
            if row is None or row["status"] not in ("dispatched", status):
                raise ValueError("only dispatched candidates can be settled")
            if row["status"] == status:
                if row["sample_path"] != path:
                    raise ValueError("conflicting settlement")
                return
            if path is not None:
                meta = json.loads((Path(path)/"meta.json").read_text())
                if (meta.get("schema") != SCHEMA or not meta.get("admission", {}).get("accepted")
                        or meta.get("candidate_id") != candidate_id
                        or meta.get("gesture", {}).get("payload_sha256") != row["command_hash"]):
                    raise ValueError("sample does not prove admission of this command")
                from .capture import verify_sample
                verify_sample(Path(path))
                session = self.session(row["session"])
                fingerprint = (self.get("qualified")["fingerprint"] if session["phase"] == "production"
                               else self.db.execute("SELECT fingerprint FROM rounds WHERE number=?",
                                                    (session["round"],)).fetchone()[0])
                if (meta["session_id"] != session["id"] or meta["park"] != session["park"]
                        or meta["phase"] != session["phase"] or meta["split"] != session["split"]
                        or meta["fingerprint"] != fingerprint):
                    raise ValueError("sample provenance does not match ledger session")
            self.db.execute("UPDATE attempts SET status=?,sample_path=?,reason=? WHERE id=?",
                            (status, path, reason, candidate_id))

    def cancel_planned(self, ids):
        with self.db:
            self.db.executemany("UPDATE attempts SET status='cancelled' WHERE id=? AND status='planned'",
                                ((i,) for i in ids))

    def seal_interrupted(self, ids, reason="interrupted: unknown dispatch outcome"):
        """Reconcile preserved recording first; otherwise consume unknown attempts."""
        with self.db:
            self.db.executemany(
                "UPDATE attempts SET status='rejected',reason=? WHERE id=? AND status='dispatched'",
                ((reason, i) for i in ids))
        self.cancel_planned(ids)

    def audit_manifest(self, number: int):
        row = self.db.execute("SELECT * FROM rounds WHERE number=?", (number,)).fetchone()
        if row is None or row["status"] not in ("collecting", "review"):
            raise ValueError("audit round unavailable")
        entries = []
        for session in self.sessions(phase="audit"):
            if session["round"] != number:
                continue
            samples = self.admitted(session["id"])
            if len(samples) != 20 or any(self.remaining(session["id"], s, include_pending=False)
                                          for s in SAMPLERS):
                raise ValueError("audit requires 20 admitted clips per park, 10 per sampler")
            for sample in samples:
                path = Path(sample["sample_path"])
                from .capture import verify_sample
                meta = verify_sample(path)
                if meta["gesture"]["payload_sha256"] != sample["command_hash"]:
                    raise ValueError("admitted command changed")
                entries.append(dict(id=sample["id"], park=session["park"],
                                    sample_dir=str(path.relative_to(self.root)),
                                    meta_sha256=__import__("hashlib").sha256((path/"meta.json").read_bytes()).hexdigest(),
                                    media_sha256=__import__("hashlib").sha256((path/"clip.mp4").read_bytes()).hexdigest(),
                                    frame_times_s=meta["frame_times_s"],
                                    onset_s=meta["onset_s"], release_s=meta["release_s"],
                                    gesture=meta["gesture"], clock_rate=meta["clock_rate"]))
        if len(entries) != 100:
            raise ValueError("exactly 100 audit clips required")
        manifest = dict(schema=SCHEMA, round=number, fingerprint=row["fingerprint"],
                        samples=entries)
        manifest["sha256"] = digest(manifest)
        if row["manifest_hash"] and row["manifest_hash"] != manifest["sha256"]:
            raise ValueError("sealed audit data changed")
        with self.db:
            self.db.execute("UPDATE rounds SET status='review',manifest_hash=? WHERE number=?",
                            (manifest["sha256"], number))
        destination = self.root/f"audit-{number}"/"manifest.json"
        if destination.exists():
            if json.loads(destination.read_text()) != manifest:
                raise ValueError("existing audit manifest changed")
        else:
            save_json(destination, manifest)
        return manifest

    def import_review(self, report: dict):
        from .audit import score_review
        number = report.get("round")
        row = self.db.execute("SELECT * FROM rounds WHERE number=?", (number,)).fetchone()
        if row and row["status"] in ("passed", "failed") and row["review_json"]:
            previous = json.loads(row["review_json"])
            if previous.get("report") != report:
                raise ValueError("sealed review cannot be replaced")
            return previous["score"]
        if row is None or row["status"] != "review":
            raise ValueError("round is not awaiting review")
        manifest = json.loads((self.root/f"audit-{number}"/"manifest.json").read_text())
        if manifest["sha256"] != row["manifest_hash"]:
            raise ValueError("audit manifest differs from ledger")
        from .capture import file_sha256
        for item in manifest["samples"]:
            path = (self.root/item["sample_dir"]).resolve()
            if (not path.is_relative_to(self.root) or file_sha256(path/"meta.json") != item["meta_sha256"]
                    or file_sha256(path/"clip.mp4") != item["media_sha256"]):
                raise ValueError("reviewed audit evidence changed")
        result = score_review(manifest, report)
        # Files precede the DB transition; retry after interruption is idempotent.
        for name, value in (("review", report), ("score", result)):
            destination = self.root/f"audit-{number}"/f"{name}.json"
            if destination.exists():
                if json.loads(destination.read_text()) != value:
                    raise ValueError("existing review evidence conflicts")
            else:
                save_json(destination, value)
        with self.db:
            self.db.execute("UPDATE rounds SET status=?,review_json=? WHERE number=?",
                            ("passed" if result["passed"] else "failed",
                             json.dumps(dict(report=report, score=result)), number))
            if result["passed"]:
                self.db.execute("INSERT INTO config VALUES('qualified',?)",
                                (json.dumps(dict(round=number, fingerprint=row["fingerprint"],
                                                 audit_sha256=manifest["sha256"])),))
        return result

    def fail_round(self, reason: str):
        """A technical failure consumes a round too; no unlimited pilot restart loophole."""
        if not reason.strip():
            raise ValueError("diagnosis required")
        with self.db:
            row = self.db.execute("SELECT * FROM rounds ORDER BY number DESC LIMIT 1").fetchone()
            if row is None or row["status"] != "collecting":
                raise ValueError("only an active collection round can be failed")
            self.db.execute("UPDATE rounds SET status='failed',review_json=? WHERE number=?",
                            (json.dumps(dict(technical_failure=reason)), row["number"]))
            ids = [r[0] for r in self.db.execute(
                "SELECT a.id FROM attempts a JOIN sessions s ON a.session=s.id WHERE s.round=? AND a.status='planned'",
                (row["number"],))]
            self.db.executemany("UPDATE attempts SET status='cancelled' WHERE id=?", ((i,) for i in ids))

    def status(self):
        return dict(definition=self.definition, attempts_spent=self.spent,
                    attempts_remaining=self.definition["attempt_limit"]-self.spent,
                    qualified=self.get("qualified"),
                    rounds=[dict(r) for r in self.db.execute("SELECT number,status,fingerprint FROM rounds")],
                    admitted=len(self.admitted()),
                    sessions=[dict(s, admitted=len(self.admitted(s["id"]))) for s in self.sessions()])
