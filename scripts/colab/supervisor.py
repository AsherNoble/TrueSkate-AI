"""Run in Colab's notebook kernel; training runs only in the pinned .venv.

CPU preparation mounts Drive, publishes the bundle, and records one attempt
before an operator selects a free T4. Every allocated GPU session is bounded
by the remaining ledger budget, including installation and staging.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


def load_harness(path):
    spec = importlib.util.spec_from_file_location("colab_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(bundle, drive_folder):
    """Execute on CPU before requesting a GPU; Drive authorization is interactive."""
    bundle, drive_folder = Path(bundle), Path(drive_folder)
    from google.colab import drive
    drive.mount("/content/drive")
    drive_folder.mkdir(parents=True, exist_ok=True)
    harness = load_harness(bundle / "benchmark.py")
    # Explicit allowlist. Never upload operator .venv, Google/Modal tokens, .env,
    # checkout Git metadata, or any other local files.
    lock = json.loads((bundle / "inputs/input-lock.json").read_text())
    for name in ["input-lock.json", *lock["files"]]:
        harness.copy_verified(bundle / "inputs" / name, drive_folder / "inputs" / name)
    for name in ("benchmark.py", "supervisor.py"):
        harness.copy_verified(bundle / name, drive_folder / name)
    ledger_path = drive_folder / "results/ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    harness.register_attempt(ledger)
    ledger["prepared_attempt_at"] = ledger["connection_attempts"][-1]["at"]
    harness.write_json(ledger_path, ledger)
    print("Private artifacts verified; one attempt recorded. Select free T4, normal RAM. No Pro or paid credits.")


def execute(drive_folder, phase):
    from google.colab import drive, runtime
    import psutil
    drive.mount("/content/drive")
    folder = Path(drive_folder)
    harness = load_harness(folder / "benchmark.py")
    ledger_path = folder / "results/ledger.json"
    ledger = json.loads(ledger_path.read_text())
    persistent = folder / "results"
    # Allocation is conservatively charged from VM boot, including time spent
    # waiting for Drive authorization. Hardware identity is captured before setup.
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True)
    if gpu_info.returncode:
        raise RuntimeError("Select a free GPU runtime before this cell")
    if any("released_at" not in s for s in ledger.get("sessions", [])):
        raise RuntimeError("Close the previous allocation in the ledger before reconnecting")
    if ledger.get("prepared_attempt_at") is None:
        raise RuntimeError("Run the CPU preparation/attempt cell before requesting a GPU")
    sessions = ledger.setdefault("sessions", [])
    if any(s["boot_id"] == boot_id for s in sessions):
        raise RuntimeError("This VM was already used; recovery requires deleting the runtime")
    if phase == "completion" and not any(s.get("outcome") == "awaiting_runtime_deletion"
                                         for s in sessions):
        raise RuntimeError("Epoch-5 deletion boundary must precede completion")
    session_id = f"session-{len(sessions)+1}"
    session = {"id": session_id, "boot_id": boot_id, "allocated_at": psutil.boot_time(),
               "allocation_time_basis": "conservative VM boot timestamp", "gpu_identity": gpu_info.stdout.strip(),
               "phase": phase, "attempt_at": ledger.pop("prepared_attempt_at")}
    sessions.append(session)
    harness.write_json(ledger_path, ledger)
    seconds = harness.remaining(ledger)
    deadline = time.time() + seconds
    if seconds <= 0:
        session["cause"] = "budget exhausted before setup"
        harness.write_json(ledger_path, ledger)
        runtime.unassign()
        return
    local = Path("/content/model1-benchmark")
    local.mkdir(exist_ok=True)
    log_dir = persistent / "sessions" / session_id
    log_dir.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    process_holder = []

    def expire():
        if stop_event.wait(max(0, deadline-time.time())):
            return
        for proc in process_holder:
            if proc.poll() is None:
                proc.kill()
        session["outcome"] = "budget_expired"
        session["release_requested_at"] = time.time()
        ledger["cause"] = "aggregate allocation or 72-hour calendar budget expired"
        harness.write_json(ledger_path, ledger)
        # No ping, reconnect, retry, or background keep-alive.
        runtime.unassign()
    threading.Thread(target=expire, daemon=True).start()

    def command(cmd, log):
        if time.time() >= deadline:
            raise TimeoutError("allocation budget expired")
        with log.open("a") as output:
            proc = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
            process_holder.append(proc)
            try:
                rc = proc.wait(timeout=max(0.01, deadline-time.time()))
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
                raise TimeoutError("allocation budget expired")
            if rc:
                raise RuntimeError(f"command failed rc={rc}: {cmd}")

    outcome = "incomplete"
    try:
        setup_started = time.monotonic()
        install_log = log_dir / "install.log"
        command([sys.executable, "-m", "pip", "install", "uv"], install_log)
        command(["uv", "python", "install", "3.11"], install_log)
        command(["uv", "venv", "--python", "3.11", str(local / ".venv")], install_log)
        python = str(local / ".venv/bin/python")
        command(["uv", "pip", "install", "--python", python,
                 *[f"{p}=={v}" for p,v in harness.PACKAGES.items()]], install_log)
        harness.write_json(log_dir / "installation.json", {"seconds":time.monotonic()-setup_started})
        command([python, str(folder / "benchmark.py"), "run", "--inputs", str(folder / "inputs"),
                 "--persistent", str(persistent), "--local", str(local), "--session-id", session_id,
                 "--phase", phase], log_dir / "training.log")
        outcome = json.loads((log_dir / "outcome.json").read_text())["status"]
        session["outcome"] = outcome
        if outcome == "completed":
            ledger["cause"] = "40 epochs completed with verified recovery"
        elif outcome == "awaiting_runtime_deletion":
            ledger["cause"] = "epoch 5 verified; awaiting recreated runtime"
    except Exception as exc:
        session["outcome"] = "incomplete"
        session["error"] = str(exc)
        failure_path = log_dir / "failure.json"
        ledger["cause"] = json.loads(failure_path.read_text())["error"] if failure_path.exists() else str(exc)
        print(f"Stopped: {exc}. Last verified epoch remains in private Drive.")
    finally:
        session["release_requested_at"] = time.time()
        # Keep the allocation open until operator/CLI verifies actual deletion.
        harness.write_json(ledger_path, ledger)
        reference=json.loads((folder / "inputs/reference/modal_seed0.json").read_text())
        harness.report(persistent, reference)
        stop_event.set()
        print(f"Session {session_id}: {session['outcome']}. Releasing runtime; verify deletion and close ledger.")
        runtime.unassign()


def confirm_release(folder, session_id, released_at):
    """Call on CPU only after the UI confirms deletion. Timestamp must be observed."""
    folder = Path(folder)
    h = load_harness(folder / "benchmark.py")
    ledger_path = folder / "results/ledger.json"
    ledger = json.loads(ledger_path.read_text())
    session = next(s for s in ledger["sessions"] if s["id"] == session_id)
    if "released_at" in session:
        raise ValueError("session release already recorded; preserve the original observation")
    if released_at < session["allocated_at"]:
        raise ValueError("release precedes allocation")
    session["released_at"] = released_at
    session["release_verified"] = True
    h.write_json(ledger_path, ledger)
    h.report(folder / "results", json.loads((folder / "inputs/reference/modal_seed0.json").read_text()))
