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
import logging
import time

from trueskate_ai.rl.device_worker import ALL_DEAD_TIMEOUT, DeviceWorker


class AllWorkersDeadError(RuntimeError):
    """Raised when every worker has been unreachable past ALL_DEAD_TIMEOUT."""


class WorkerPool:
    """Owns a fleet of DeviceWorkers and their connection lifecycle.

    Iterable and indexable. Usable as a context manager — ``__enter__``
    connects all workers, ``__exit__`` disconnects them.
    """

    def __init__(self, device_cfgs: list[dict]) -> None:
        """Build one DeviceWorker per config. Caller must pass explicit cfgs.

        No default: an implicit "all DEVICES" fallback would bypass role
        filtering and could silently grab the personal iPhone 11 (role=
        "personal") during an unattended run. Callers select the roster via
        device_worker.select_devices/resolve_devices. Both real callers
        (cmaes_optimizer, ppo.trainer) already pass an explicit list.
        """
        if not device_cfgs:
            raise ValueError(
                "WorkerPool requires a non-empty device_cfgs list; refusing to "
                "default to the full DEVICES fleet (includes the personal phone). "
                "Use select_devices()/resolve_devices() to pick the roster."
            )
        self._workers: list[DeviceWorker] = [DeviceWorker(cfg) for cfg in device_cfgs]

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
        """Connect to every device, dropping any whose services aren't running.

        ``launch_services.py`` can bring up only a subset of the configured
        devices (e.g. ``--device iPhone_XR``). A device whose WDA/Appium is
        unreachable is logged and excluded from the pool rather than aborting
        the whole run. Raises RuntimeError only when *no* device connects.
        """
        connected: list[DeviceWorker] = []
        for w in self._workers:
            try:
                w.connect()
                connected.append(w)
            except Exception as exc:
                logging.warning(
                    "[%s] connection failed — excluding from this run: %s",
                    w.device_id, exc,
                )
        if not connected:
            raise RuntimeError(
                "No devices could be connected. Start at least one with "
                "'python scripts/launch_services.py [--device NAME]'."
            )
        skipped = len(self._workers) - len(connected)
        if skipped:
            logging.warning(
                "Running on %d of %d configured device(s); %d skipped.",
                len(connected), len(self._workers), skipped,
            )
        self._workers = connected

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
