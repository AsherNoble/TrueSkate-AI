"""Self-healing supervisor for 24h unattended training.

Starts and babysits the whole rig as child processes, keeps the Mac awake with
``caffeinate``, and restarts any piece that dies:

    services  — scripts/launch_services.py (already self-heals per device)
    status    — scripts/status_server.py   (Tailscale dashboard)
    training  — scripts/train/train_cmaes.py (restarted with a bumped seed
                whenever it exits, so collection runs continuously for 24h)

Usage (typical home run, over `tailscale ssh` inside tmux):

    python scripts/run_training.py --curriculum curricula/360_flip.json \
        --initial-mean trick_libraries/<best>.json

Stop with Ctrl-C (or `launchctl` if installed as a service): all children and
caffeinate are torn down cleanly.

Device selection (--devices/--personal/--all-devices) is resolved once and
passed identically to both launch_services and train_cmaes so they always agree.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from trueskate_ai.rl.device_worker import (  # noqa: E402
    add_device_selection_args,
    resolve_devices,
)
from trueskate_ai.utils.notify import notify  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

_PY = sys.executable
_POLL_S = 5.0
_TRAINING_RESTART_BACKOFF_S = 15.0
_SERVICES_READY_TIMEOUT_S = 600.0


def _start_caffeinate() -> subprocess.Popen | None:
    """Keep the Mac fully awake while we run; auto-exits when this PID dies."""
    try:
        proc = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
        print(f"[supervisor] caffeinate active (PID {proc.pid}).")
        return proc
    except FileNotFoundError:
        print("[supervisor] caffeinate not found — Mac may sleep. (macOS only.)")
        return None


def _wait_for_services(devices: list[dict], timeout: float) -> bool:
    """Block until every selected device's WDA /status responds, or timeout."""
    deadline = time.time() + timeout
    pending = {d["name"]: d["wda_port"] for d in devices}
    while time.time() < deadline:
        for name in list(pending):
            port = pending[name]
            try:
                if requests.get(f"http://127.0.0.1:{port}/status", timeout=2).status_code == 200:
                    print(f"[supervisor] WDA ready: {name} (:{port})")
                    del pending[name]
            except requests.exceptions.RequestException:
                pass
        if not pending:
            return True
        time.sleep(5)
    print(f"[supervisor] Timed out waiting for WDA on: {', '.join(pending)}")
    return False


class Supervisor:
    def __init__(self, args, devices: list[dict]) -> None:
        self.args = args
        self.devices = devices
        self.device_arg = ",".join(d["name"] for d in devices)
        self.caffeinate: subprocess.Popen | None = None
        self.services: subprocess.Popen | None = None
        self.status: subprocess.Popen | None = None
        self.training: subprocess.Popen | None = None
        self.seed = args.seed
        self.training_restarts = 0
        self._stop = False

    # -- child command builders --------------------------------------------

    def _services_cmd(self) -> list[str]:
        return [_PY, str(_HERE / "launch_services.py"), "--devices", self.device_arg]

    def _status_cmd(self) -> list[str]:
        return [
            _PY, str(_HERE / "status_server.py"),
            "--port", str(self.args.status_port),
        ]

    def _training_cmd(self) -> list[str]:
        cmd = [
            _PY, str(_HERE / "train" / "train_cmaes.py"),
            "--curriculum", str(self.args.curriculum),
            "--devices", self.device_arg,
            "--max-evals", str(self.args.max_evals),
            "--pop-size", str(self.args.pop_size),
            "--seed", str(self.seed),
            "--wait-time", str(self.args.wait_time),
            "--settle-time", str(self.args.settle_time),
        ]
        if self.args.initial_mean:
            cmd += ["--initial-mean", str(self.args.initial_mean)]
        return cmd

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if not self.args.no_caffeinate:
            self.caffeinate = _start_caffeinate()

        print(f"[supervisor] Launching services for: {self.device_arg}")
        self.services = subprocess.Popen(self._services_cmd())

        if not self.args.no_status_server:
            self.status = subprocess.Popen(self._status_cmd())
            print(f"[supervisor] Status server on :{self.args.status_port} "
                  f"(expose: tailscale serve --bg {self.args.status_port})")

        if not _wait_for_services(self.devices, _SERVICES_READY_TIMEOUT_S):
            notify("Services failed to come up — no WDA. Supervisor still retrying.",
                   title="TrueSkate rig", priority="high", tags=["warning"])

        self._start_training()
        notify(
            f"Supervisor up: target={self.args.curriculum.stem}, devices={self.device_arg}.",
            title="TrueSkate rig", tags=["rocket"],
        )

    def _start_training(self) -> None:
        print(f"[supervisor] Starting training (seed={self.seed}).")
        self.training = subprocess.Popen(self._training_cmd())

    def _restart(self, what: str, proc_attr: str, builder, *, notify_it=True) -> None:
        proc = getattr(self, proc_attr)
        code = proc.poll() if proc else None
        print(f"[supervisor] {what} exited (code={code}) — restarting.")
        if notify_it:
            notify(f"{what} died (code={code}); restarting.",
                   title="TrueSkate rig", priority="high", tags=["warning"])
        setattr(self, proc_attr, subprocess.Popen(builder()))

    def run(self) -> None:
        self.start()
        try:
            while not self._stop:
                time.sleep(_POLL_S)

                if self.services and self.services.poll() is not None:
                    self._restart("services", "services", self._services_cmd)
                    _wait_for_services(self.devices, _SERVICES_READY_TIMEOUT_S)

                if self.status and self.status.poll() is not None:
                    self._restart("status server", "status", self._status_cmd,
                                  notify_it=False)

                if self.training and self.training.poll() is not None:
                    # Training finished or crashed — bump seed and relaunch so
                    # 24h collection never just stops.
                    self.training_restarts += 1
                    self.seed += 1
                    notify(
                        f"Training run ended (restart #{self.training_restarts}); "
                        f"relaunching with seed={self.seed}.",
                        title="TrueSkate training", tags=["arrows_counterclockwise"],
                    )
                    time.sleep(_TRAINING_RESTART_BACKOFF_S)
                    self._start_training()

        except KeyboardInterrupt:
            print("\n[supervisor] Interrupted — shutting down.")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop = True
        for name, proc in (
            ("training", self.training),
            ("status", self.status),
            ("services", self.services),
        ):
            if proc and proc.poll() is None:
                print(f"[supervisor] Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if self.caffeinate and self.caffeinate.poll() is None:
            self.caffeinate.terminate()
        notify("Supervisor stopped.", title="TrueSkate rig", tags=["octagonal_sign"],
               block=True)
        print("[supervisor] Shutdown complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="24h self-healing training supervisor.")
    parser.add_argument("--curriculum", type=Path, required=True,
                        help="Curriculum JSON (e.g. curricula/360_flip.json).")
    parser.add_argument("--initial-mean", type=Path, default=None,
                        help="Trick library JSON to warm-start CMA-ES.")
    parser.add_argument("--max-evals", type=int, default=100_000,
                        help="Per-training-run eval budget (default large; restarts loop).")
    parser.add_argument("--pop-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42,
                        help="Starting seed; bumped on each training restart.")
    parser.add_argument("--wait-time", type=float, default=0.0)
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--status-port", type=int, default=8200)
    parser.add_argument("--no-status-server", action="store_true")
    parser.add_argument("--no-caffeinate", action="store_true")
    add_device_selection_args(parser)
    args = parser.parse_args()

    devices = resolve_devices(
        devices_arg=args.devices, personal=args.personal, all_devices=args.all_devices,
    )
    if not devices:
        print("No devices selected.")
        sys.exit(1)

    sup = Supervisor(args, devices)

    # Translate SIGTERM (launchd / kill) into the same clean shutdown as Ctrl-C.
    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _on_sigterm)
    sup.run()


if __name__ == "__main__":
    main()
