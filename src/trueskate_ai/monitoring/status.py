"""Heartbeat status for unattended training runs.

``StatusTracker`` accumulates live run metrics (throughput, per-device eval
counts, last trick, best, alive/dead) and writes them atomically to a JSON file
on every generation. ``scripts/status_server.py`` serves that file over
``tailscale serve`` so Asher can glance at the rig from his phone on the road.

The file is the single source of truth for "is the rig actually working?":
  * ``updated_at`` going stale  → training stalled / process died.
  * ``evals_per_hour`` near 0   → devices alive but producing nothing.
  * a device in ``dead``        → that phone dropped off.
"""
import json
import os
import tempfile
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

# Rolling window for "recent" throughput (wall-clock seconds).
_RECENT_WINDOW_S = 600.0


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON to ``path`` atomically (temp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class StatusTracker:
    """Accumulates run metrics and writes an atomic status.json heartbeat."""

    def __init__(
        self,
        status_path: Path,
        *,
        run_id: str,
        run_dir: str,
        target: str | None,
        max_evals: int,
        pop_size: int,
        device_ids: list[str],
    ) -> None:
        self.path = Path(status_path)
        self.run_id = run_id
        self.run_dir = run_dir
        self.target = target
        self.max_evals = max_evals
        self.pop_size = pop_size
        self.device_ids = list(device_ids)

        self._start = time.time()
        self.total_evals = 0
        self.total_lands = 0
        self.per_device_evals: dict[str, int] = defaultdict(int)
        self.per_device_lands: dict[str, int] = defaultdict(int)
        self.last_trick: dict[str, dict] = {}
        self.best_reward = 0.0
        self.best_trick: str | None = None
        self.generation = 0
        self.gen_best = 0.0
        self.gen_mean = 0.0
        self.alive: list[str] = list(device_ids)
        self.dead: list[str] = []
        self.note: str | None = None
        # (wall_time, reward) pairs for rolling-window throughput.
        self._recent: deque[tuple[float, float]] = deque()

    # -- ingest -------------------------------------------------------------

    def record_eval(
        self, device_id: str, reward: float, trick_name, trick_status
    ) -> None:
        self.total_evals += 1
        self.per_device_evals[device_id] += 1
        landed = reward > 0
        if landed:
            self.total_lands += 1
            self.per_device_lands[device_id] += 1
        self.last_trick[device_id] = {
            "trick": trick_name,
            "status": trick_status,
            "reward": round(float(reward), 3),
            "landed": landed,
        }
        now = time.time()
        self._recent.append((now, float(reward)))
        cutoff = now - _RECENT_WINDOW_S
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()

    def record_generation(
        self,
        *,
        generation: int,
        gen_best: float,
        gen_mean: float,
        best_reward: float,
        best_trick,
        alive: list[str],
        dead: list[str],
        note: str | None = None,
    ) -> None:
        self.generation = generation
        self.gen_best = gen_best
        self.gen_mean = gen_mean
        self.best_reward = best_reward
        self.best_trick = best_trick
        self.alive = list(alive)
        self.dead = list(dead)
        self.note = note
        self.write()

    def set_note(self, note: str | None) -> None:
        self.note = note
        self.write()

    # -- derived ------------------------------------------------------------

    def _rates(self) -> tuple[float, float, float]:
        """(overall evals/hr, recent evals/hr, recent lands/hr)."""
        elapsed_h = max(1e-9, (time.time() - self._start) / 3600.0)
        overall = self.total_evals / elapsed_h
        if self._recent:
            span_s = max(1.0, self._recent[-1][0] - self._recent[0][0])
            span_h = span_s / 3600.0
            recent_evals = len(self._recent) / span_h
            recent_lands = sum(1 for _, r in self._recent if r > 0) / span_h
        else:
            recent_evals = recent_lands = 0.0
        return overall, recent_evals, recent_lands

    def snapshot(self) -> dict:
        overall, recent_evals, recent_lands = self._rates()
        elapsed_s = time.time() - self._start
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "target": self.target,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "started_at": datetime.fromtimestamp(
                self._start, timezone.utc
            ).isoformat(timespec="seconds"),
            "uptime_hours": round(elapsed_s / 3600.0, 3),
            "generation": self.generation,
            "total_evals": self.total_evals,
            "max_evals": self.max_evals,
            "total_lands": self.total_lands,
            "land_rate": round(self.total_lands / self.total_evals, 4)
            if self.total_evals
            else 0.0,
            "evals_per_hour": round(overall, 1),
            "recent_evals_per_hour": round(recent_evals, 1),
            "recent_lands_per_hour": round(recent_lands, 1),
            "gen_best": round(self.gen_best, 3),
            "gen_mean": round(self.gen_mean, 4),
            "best_reward": round(self.best_reward, 3),
            "best_trick": self.best_trick,
            "devices": [
                {
                    "device_id": d,
                    "alive": d in self.alive,
                    "evals": self.per_device_evals.get(d, 0),
                    "lands": self.per_device_lands.get(d, 0),
                    "last_trick": self.last_trick.get(d),
                }
                for d in self.device_ids
            ],
            "alive": self.alive,
            "dead": self.dead,
            "note": self.note,
        }

    def write(self) -> None:
        try:
            _atomic_write_json(self.path, self.snapshot())
        except Exception:
            # Status is best-effort telemetry — never break training over it.
            pass
