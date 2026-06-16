"""Run-scoped JSONL logging for RL training orchestrators.

A ``RunLogger`` owns one timestamped run folder and its line-buffered JSONL
log file. Both the CMA-ES optimizer and the PPO trainer write through it,
which gives a single logging sink that downstream tooling — notably remote
training monitors — can hook by wrapping or subclassing ``write``.

Layout created per run::

    <log_dir>/runs/<prefix>_run_<YYYYmmdd_HHMMSS_ffffff>/
        <prefix>_run_<YYYYmmdd_HHMMSS_ffffff>.jsonl
        <subdir>/                # optional, e.g. "frames" for CMA-ES

Public API:
    RunLogger — create the run folder, append JSONL records, manage subdirs.
"""
import json
from datetime import datetime
from pathlib import Path


class RunLogger:
    """Owns one run folder and its line-buffered JSONL log.

    Usable as a context manager; ``close()`` flushes and closes the handle.
    """

    def __init__(self, log_dir: Path, prefix: str, *, subdirs: tuple[str, ...] = ()) -> None:
        """Create the run folder and open its JSONL log.

        Args:
            log_dir: Root directory; the run folder is created under ``runs/``.
            prefix:  Run-folder / log-file prefix (e.g. ``"cmaes"``, ``"ppo"``).
            subdirs: Names of subdirectories to pre-create inside the run folder.
        """
        # Collision-proof run dir. A 1-second run_id + exist_ok mkdir + "w"-mode
        # JSONL let two runs starting in the same second (parallel spawns, fast
        # restarts) reuse the same folder and TRUNCATE each other's live log.
        # Append microseconds for ordering AND create with exist_ok=False,
        # retrying with a counter suffix, so a run dir is NEVER silently reused.
        # run_id stays timestamp-prefixed and lexically sortable.
        runs_root = Path(log_dir) / "runs"
        base_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_id: str = base_id
        self.run_dir: Path = runs_root / f"{prefix}_run_{base_id}"
        suffix = 0
        while True:
            try:
                self.run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
                self.run_id = f"{base_id}_{suffix}"
                self.run_dir = runs_root / f"{prefix}_run_{self.run_id}"
        for name in subdirs:
            (self.run_dir / name).mkdir(exist_ok=True)
        self.log_path: Path = self.run_dir / f"{prefix}_run_{self.run_id}.jsonl"
        # "x" (exclusive create) — refuse to clobber an existing JSONL even if
        # something raced a file into our fresh dir; never silently truncate.
        self._fh = self.log_path.open("x", buffering=1)  # line-buffered

    def write(self, record: dict) -> None:
        """Append one JSON record as a line to the run log."""
        self._fh.write(json.dumps(record) + "\n")

    def subdir(self, name: str) -> Path:
        """Return a subdirectory of the run folder, creating it if needed."""
        path = self.run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def close(self) -> None:
        """Flush and close the log file handle."""
        self._fh.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
