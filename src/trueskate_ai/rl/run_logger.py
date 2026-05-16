"""Run-scoped JSONL logging for RL training orchestrators.

A ``RunLogger`` owns one timestamped run folder and its line-buffered JSONL
log file. Both the CMA-ES optimizer and the PPO trainer write through it,
which gives a single logging sink that downstream tooling — notably remote
training monitors — can hook by wrapping or subclassing ``write``.

Layout created per run::

    <log_dir>/runs/<prefix>_run_<YYYYmmdd_HHMMSS>/
        <prefix>_run_<YYYYmmdd_HHMMSS>.jsonl
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
        self.run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir: Path = Path(log_dir) / "runs" / f"{prefix}_run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in subdirs:
            (self.run_dir / name).mkdir(exist_ok=True)
        self.log_path: Path = self.run_dir / f"{prefix}_run_{self.run_id}.jsonl"
        self._fh = self.log_path.open("w", buffering=1)  # line-buffered

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
