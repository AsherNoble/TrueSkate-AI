"""Run in Colab's notebook kernel; training runs only in the pinned .venv.

Storage is one Drive folder reached through a drive.file-scoped rclone remote
that the notebook bootstrap configures from a Colab secret. The credential
sees only files its own OAuth client created, never the rest of Drive.
Persistent state is mirrored under STORE and uploaded with remote checksums.

CPU preparation verifies the uploaded inputs by Drive's own checksums and
records one attempt before an operator selects a free T4. Every allocated GPU
session is bounded by the remaining ledger budget, including installation and
staging.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

STORE = Path("/content/bench-store")
LOG_PUSH_SECONDS = 600


def load_harness(store=STORE):
    spec = importlib.util.spec_from_file_location("colab_benchmark", Path(store) / "code/benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STORE_ROOT = Path(store)
    return module


def open_ledger(h, store=STORE):
    """Refresh the local results mirror, then read the ledger from it."""
    results = Path(store) / "results"
    results.mkdir(parents=True, exist_ok=True)
    try:
        h.rclone("copy", h.REMOTE + "results", results)
    except RuntimeError as exc:
        if "directory not found" not in str(exc):
            raise
    path = results / "ledger.json"
    return path, (json.loads(path.read_text()) if path.exists() else {})


def save_ledger(h, path, ledger):
    h.write_json(path, ledger)
    h.persist(path)


def load_reference(h, store=STORE):
    lock = json.loads((Path(store) / "inputs/input-lock.json").read_text())
    path = Path(store) / "inputs/reference/modal_seed0.json"
    h.rclone("copyto", h.REMOTE + "inputs/reference/modal_seed0.json", path)
    if h.sha256(path) != lock["files"]["reference/modal_seed0.json"]["sha256"]:
        raise RuntimeError("reference history does not match the input lock")
    return json.loads(path.read_text())


def record_attempt(h, store=STORE):
    path, ledger = open_ledger(h, store)
    h.register_attempt(ledger)
    ledger["prepared_attempt_at"] = ledger["connection_attempts"][-1]["at"]
    save_ledger(h, path, ledger)
    return ledger


def prepare(store=STORE):
    """CPU only, before requesting a GPU: verify inputs remotely, record one attempt."""
    h = load_harness(store)
    lock = json.loads((Path(store) / "inputs/input-lock.json").read_text())
    for name, record in lock["files"].items():
        if h.remote_sha256(f"inputs/{name}") != record["sha256"]:
            raise RuntimeError(f"remote input hash mismatch: {name}")
    record_attempt(h, store)
    print("Scoped inputs verified by Drive checksums; one attempt recorded. "
          "Select free T4, standard RAM. No paid compute.")


def next_attempt(store=STORE):
    """CPU only, after the previous allocation is closed in the ledger."""
    h = load_harness(store)
    ledger = record_attempt(h, store)
    print("Next attempt recorded; remaining seconds:", h.remaining(ledger))


def record_failed_connection(note, store=STORE):
    """CPU only: the prepared attempt never produced a GPU session."""
    h = load_harness(store)
    path, ledger = open_ledger(h, store)
    at = ledger.pop("prepared_attempt_at", None)
    if at is None:
        raise RuntimeError("no prepared attempt to mark as failed")
    next(a for a in ledger["connection_attempts"] if a["at"] == at).update(failed=True, note=note)
    save_ledger(h, path, ledger)


def execute(phase, store=STORE):
    from google.colab import runtime
    import psutil
    store = Path(store)
    h = load_harness(store)
    ledger_path, ledger = open_ledger(h, store)
    persistent = store / "results"
    rclone = os.environ.get("RCLONE_BIN", "rclone")
    # Allocation is conservatively charged from VM boot, including time spent
    # on bootstrap. Hardware identity is captured before setup.
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
    save_ledger(h, ledger_path, ledger)
    seconds = h.remaining(ledger)
    deadline = time.time() + seconds
    if seconds <= 0:
        session["cause"] = "budget exhausted before setup"
        save_ledger(h, ledger_path, ledger)
        runtime.unassign()
        return
    local = Path("/content/model1-benchmark")
    local.mkdir(exist_ok=True)
    log_dir = persistent / "sessions" / session_id
    log_dir.mkdir(parents=True, exist_ok=True)
    logs = (log_dir / "install.log", log_dir / "training.log")
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
        try:
            save_ledger(h, ledger_path, ledger)
        finally:
            # No ping, reconnect, retry, or background keep-alive.
            runtime.unassign()

    def push_logs():
        while not stop_event.wait(LOG_PUSH_SECONDS):
            try:
                h.persist(*(p for p in logs if p.exists()))
            except Exception as exc:
                print(f"log upload failed: {exc}", flush=True)
    threading.Thread(target=expire, daemon=True).start()
    threading.Thread(target=push_logs, daemon=True).start()

    def command(cmd, log):
        if time.time() >= deadline:
            raise TimeoutError("allocation budget expired")
        with log.open("a") as output:
            proc = subprocess.Popen([str(c) for c in cmd], stdout=output, stderr=subprocess.STDOUT)
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
        install_log = logs[0]
        command([rclone, "copy", h.REMOTE + "inputs", store / "inputs"], install_log)
        pulled = time.monotonic()
        command([sys.executable, "-m", "pip", "install", "uv"], install_log)
        command(["uv", "python", "install", "3.11"], install_log)
        command(["uv", "venv", "--python", "3.11", local / ".venv"], install_log)
        python = local / ".venv/bin/python"
        command(["uv", "pip", "install", "--python", python,
                 *[f"{p}=={v}" for p,v in h.PACKAGES.items()]], install_log)
        h.write_json(log_dir / "installation.json", {"inputs_pull_seconds": pulled - setup_started,
                                                     "seconds": time.monotonic() - pulled})
        h.persist(log_dir / "installation.json", install_log)
        command([python, store / "code/benchmark.py", "run", "--inputs", store / "inputs",
                 "--persistent", persistent, "--local", local, "--session-id", session_id,
                 "--phase", phase, "--store-root", store], logs[1])
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
        print(f"Stopped: {exc}. The last verified epoch remains in the scoped Drive folder.")
    finally:
        stop_event.set()
        session["release_requested_at"] = time.time()
        # Keep the allocation open until operator/CLI verifies actual deletion.
        try:
            save_ledger(h, ledger_path, ledger)
            h.report(persistent, json.loads((store / "inputs/reference/modal_seed0.json").read_text()))
            h.persist_tree(log_dir)
        except Exception as exc:
            print(f"Final upload failed: {exc}", flush=True)
        print(f"Session {session_id}: {session['outcome']}. Releasing runtime; verify deletion and close ledger.")
        runtime.unassign()


def confirm_release(session_id, released_at, store=STORE):
    """Call on CPU only after the UI confirms deletion. Timestamp must be observed."""
    h = load_harness(store)
    path, ledger = open_ledger(h, store)
    session = next(s for s in ledger["sessions"] if s["id"] == session_id)
    if "released_at" in session:
        raise ValueError("session release already recorded; preserve the original observation")
    if released_at < session["allocated_at"]:
        raise ValueError("release precedes allocation")
    session["released_at"] = released_at
    session["release_verified"] = True
    save_ledger(h, path, ledger)
    h.report(Path(store) / "results", load_reference(h, store))


def record_manual_effort(entries, complete, store=STORE):
    """CPU only: active operator time as [{"what": ..., "seconds": ...}]."""
    h = load_harness(store)
    path, ledger = open_ledger(h, store)
    ledger["manual_effort"] = entries
    ledger["manual_effort_complete"] = complete
    save_ledger(h, path, ledger)
    h.report(Path(store) / "results", load_reference(h, store))
