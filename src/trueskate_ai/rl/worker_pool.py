"""Lifecycle management for a fleet of DeviceWorkers.

A ``WorkerPool`` owns the set of per-device workers and the connection
lifecycle around them — construction, connect/disconnect, scheduled revival
of dead workers, and the all-dead abort check. Previously this logic was
inlined separately in the CMA-ES optimizer, the PPO trainer, and the rollout
collector; centralizing it keeps those orchestrators focused on strategy.

The pool is iterable and indexable, so dispatch loops can treat it like the
plain ``list[DeviceWorker]`` it replaces.

Public API:
    WorkerPool          — owns workers + connect/disconnect/revive/abort.
    AllWorkersDeadError — raised when the whole fleet is unreachable too long.
"""
import time

from trueskate_ai.rl.device_worker import ALL_DEAD_TIMEOUT, DEVICES, DeviceWorker


class AllWorkersDeadError(RuntimeError):
    """Raised when every worker has been unreachable past ALL_DEAD_TIMEOUT."""


class WorkerPool:
    """Owns a fleet of DeviceWorkers and their connection lifecycle.

    Iterable and indexable. Usable as a context manager — ``__enter__``
    connects all workers, ``__exit__`` disconnects them.
    """

    def __init__(self, device_cfgs: list[dict] | None = None) -> None:
        """Build one DeviceWorker per config. Defaults to the module DEVICES list."""
        cfgs = DEVICES if device_cfgs is None else device_cfgs
        self._workers: list[DeviceWorker] = [DeviceWorker(cfg) for cfg in cfgs]

    def __len__(self) -> int:
        return len(self._workers)

    def __iter__(self):
        return iter(self._workers)

    def __getitem__(self, idx) -> DeviceWorker:
        return self._workers[idx]

    @property
    def workers(self) -> list[DeviceWorker]:
        """A shallow copy of the worker list."""
        return list(self._workers)

    @property
    def alive(self) -> list[DeviceWorker]:
        """Workers not currently in the dead state."""
        return [w for w in self._workers if w.alive]

    @property
    def device_ids(self) -> list[str]:
        return [w.device_id for w in self._workers]

    def connect_all(self) -> None:
        """Connect every worker. Raises on the first connection failure."""
        for w in self._workers:
            w.connect()

    def disconnect_all(self) -> None:
        """Disconnect every worker. Never raises."""
        for w in self._workers:
            w.disconnect()

    def revive_dead(self) -> None:
        """Attempt a scheduled reconnect of each dead worker (cooldown-gated)."""
        for w in self._workers:
            w.maybe_revive()

    def raise_if_all_dead(self) -> None:
        """Raise AllWorkersDeadError if the whole fleet has been dead too long.

        No-op while at least one worker is alive, or while the fleet has been
        fully dead for less than ALL_DEAD_TIMEOUT.
        """
        if any(w.alive for w in self._workers):
            return
        now = time.monotonic()
        oldest_death = min(
            (w.dead_since for w in self._workers if w.dead_since is not None),
            default=now,
        )
        if now - oldest_death > ALL_DEAD_TIMEOUT:
            raise AllWorkersDeadError(
                f"All devices have been unreachable for >"
                f"{ALL_DEAD_TIMEOUT / 60:.0f} minutes — aborting."
            )

    def __enter__(self) -> "WorkerPool":
        self.connect_all()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect_all()
