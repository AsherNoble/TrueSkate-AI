"""Isolated Colab harness. Imports only the checksum-locked e0424f9 trainer.

No Modal invocation, credentials, collection, decoded-frame cache, or holdout
evaluation. The notebook owns allocation/release and a wall-clock subprocess
deadline; the callback commits completed epochs before the deliberate stop.

Persistent state lives in a local mirror (``STORE_ROOT``) of one Drive folder,
reached through a drive.file-scoped rclone remote: the credential sees only
files its own OAuth client created. ``persist`` uploads mirror files and
checks Drive's own SHA-256 before anything that depends on them is written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

SOURCE_SHA = "e0424f99e3f668e1722e56fac4f17372cd60a62d"
EXPERIMENT = "sha256:82e323f94bd8dc4db2f223da9a563570bbd3d72169d7819579d0645add1268b4"
SHARDS = "sha256:85e6a84418fa1eab5c8b951fcc1e690f66cfcceee5fd1cc8200bb1123cd8da28"
PACKAGES = {"torch": "2.12.0", "opencv-python-headless": "4.13.0.92",
            "numpy": "2.4.6", "scipy": "1.17.1", "selenium": "4.43.0"}
LIMIT_SECONDS = 12 * 3600
CALENDAR_SECONDS = 72 * 3600
REMOTE = "bench:"  # drive.file scope; root_folder_id is the benchmark folder
STORE_ROOT = None  # local mirror of REMOTE; None keeps everything local (tests)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256(source)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as f:
        os.fsync(f.fileno())
    if sha256(temporary) != expected:
        raise RuntimeError(f"checkpoint copy failed checksum: {destination}")
    temporary.replace(destination)
    if sha256(destination) != expected:
        raise RuntimeError(f"checkpoint persisted checksum failed: {destination}")
    return {"path": str(destination), "sha256": expected,
            "bytes": destination.stat().st_size}


def safe_extract(path, destination):
    destination = Path(destination).resolve()
    with tarfile.open(path) as archive:
        for member in archive:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError(f"unsafe archive member {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)


def rclone(*args):
    proc = subprocess.run([os.environ.get("RCLONE_BIN", "rclone"), *map(str, args)],
                          capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"rclone {args[0]} failed rc={proc.returncode}: {proc.stderr.strip()[-500:]}")
    return proc.stdout


def remote_sha256(relative):
    """SHA-256 of one remote file as reported by Drive, not recomputed locally."""
    lines = [l.split(None, 1) for l in rclone("hashsum", "sha256", REMOTE + relative).splitlines() if l.strip()]
    if len(lines) != 1 or len(lines[0][0]) != 64:
        raise RuntimeError(f"no unique remote SHA-256 for {relative}: {lines}")
    return lines[0][0]


def persist(*paths):
    """Upload mirror files in order; each is verified remotely before the next starts."""
    if STORE_ROOT is None:
        return
    root = Path(STORE_ROOT).resolve()
    for path in paths:
        path = Path(path).resolve()
        relative = path.relative_to(root).as_posix()
        expected = sha256(path)
        rclone("copyto", path, REMOTE + relative)
        if remote_sha256(relative) != expected:
            raise RuntimeError(f"remote checksum mismatch after upload: {relative}")


def persist_tree(directory):
    persist(*sorted(p for p in Path(directory).rglob("*") if p.is_file() and not p.name.endswith(".tmp")))


def remaining(ledger, now=None):
    now = time.time() if now is None else now
    sessions = ledger.get("sessions", [])
    spent = sum(max(0, s.get("released_at", now) - s["allocated_at"]) for s in sessions)
    first = min((s["allocated_at"] for s in sessions), default=now)
    return max(0, min(LIMIT_SECONDS - spent, first + CALENDAR_SECONDS - now))


def register_attempt(ledger, now=None):
    now = time.time() if now is None else now
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    attempts = ledger.setdefault("connection_attempts", [])
    if sum(a["utc_day"] == day for a in attempts) >= 2:
        raise RuntimeError("two connection attempts already recorded today (UTC)")
    if any("released_at" not in s for s in ledger.get("sessions", [])):
        raise RuntimeError("previous allocation must be closed before another attempt")
    if remaining(ledger, now) <= 0:
        raise RuntimeError("aggregate allocation/calendar budget exhausted")
    attempts.append({"utc_day": day, "at": now})


def state_digest(value):
    """Stable byte digest of tensors, nested optimizer state and RNG states."""
    import numpy as np
    import torch
    h = hashlib.sha256()

    def add(v):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().contiguous()
            h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
            h.update(v.numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=lambda x: (type(x).__name__, str(x))):
                add(k); add(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for item in v:
                add(item)
        else:
            h.update((type(v).__name__ + ":" + repr(v) + ";").encode())
    add(value)
    return h.hexdigest()


def stage(inputs, local):
    started = time.monotonic()
    lock = json.loads((inputs / "input-lock.json").read_text())
    if lock["source_sha"] != SOURCE_SHA:
        raise RuntimeError("wrong frozen source")
    for name, record in lock["files"].items():
        path = inputs / name
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"input hash mismatch: {name}")
    local.mkdir(parents=True, exist_ok=True)
    # Inputs were pulled from the scoped Drive folder; decode from ephemeral disk.
    for name in ("source.tar.gz", "data.tar"):
        copy_verified(inputs / name, local / name)
    source = local / "source"
    safe_extract(local / "source.tar.gz", source)
    safe_extract(local / "data.tar", local)
    for name, expected in lock["source_files"].items():
        if sha256(source / name) != expected:
            raise RuntimeError(f"frozen source byte mismatch: {name}")
    sys.path.insert(0, str(source / "src"))
    from trueskate_ai.data.sequential_shards import read_shard_manifest, materialize_sequential_shards
    from trueskate_ai.data.cohort_manifest import read_manifest, manifest_entries
    shard_path = local / "data/shards.json"
    shards = read_shard_manifest(shard_path)
    experiment = read_manifest(local / "data/experiment.json")
    if shards["fingerprint"] != SHARDS or experiment["fingerprint"] != EXPERIMENT:
        raise RuntimeError("full manifest fingerprint mismatch")
    sizes = {k: len(manifest_entries(experiment, partition=k))
             for k in ("train", "validation", "test")}
    if sizes != {"train": 18394, "validation": 1965, "test": 1965}:
        raise RuntimeError(f"partition sizes changed: {sizes}")
    materialize_sequential_shards(shard_path, local / "corpus", verify_samples=True)
    spec = importlib.util.spec_from_file_location(
        "colab_frozen_trainer", source / "scripts/model1/train_basic_linear_regressor.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer, {"seconds": time.monotonic() - started,
                     "shard_fingerprint": SHARDS, "experiment_fingerprint": EXPERIMENT,
                     "split_sizes": sizes, "shard_count": len(shards["shards"]),
                     "content_verified": True, "cache_frames": False}


def check_environment(evidence_path=None):
    import torch
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    evidence = {"python": sys.version, "packages": versions, "platform": platform.platform(),
                "torch_build": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "torch_threads": torch.get_num_threads(),
                "cpu_count": os.cpu_count(), "source_sha": SOURCE_SHA,
                "cuda_available": torch.cuda.is_available()}
    if evidence["cuda_available"]:
        evidence.update(gpu=torch.cuda.get_device_name(0),
                        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    if evidence_path is not None:
        write_json(evidence_path, evidence)
    for name, expected in PACKAGES.items():
        if versions[name].split("+")[0] != expected:
            raise RuntimeError(f"environment-incompatible: {name}={versions[name]}")
    if sys.version_info[:2] != (3, 11) or torch.__version__ != "2.12.0+cu130":
        raise RuntimeError("environment-incompatible: requires Python 3.11 and torch 2.12.0+cu130")
    if not evidence["cuda_available"]:
        raise RuntimeError("environment-incompatible: CUDA unavailable with original torch 2.12.0+cu130")
    return evidence


def preflight(trainer, inputs, local):
    import torch
    from torch.utils.data import DataLoader
    kwargs = dict(manifest=local / "data/experiment.json", cache_frames=False,
                  verify_manifest_content=True, image_width=64, image_height=144,
                  decode_mode="sequential")
    train = trainer.BasicLinearClipDataset(local / "corpus", manifest_partition="train", **kwargs)
    val = trainer.BasicLinearClipDataset(local / "corpus", manifest_partition="validation", **kwargs)
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = trainer.BasicLinearRegressor(base_channels=16, temporal_mixer=True).to(device)
    batch = next(iter(DataLoader(train, batch_size=8)))
    if tuple(batch["frames"].shape) != (8, 32, 3, 144, 64):
        raise RuntimeError("real batch has wrong shape")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = trainer.basic_linear_loss(model(batch["frames"].to(device)), batch["target"].to(device))
    loss.backward(); optimizer.step()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                      for p in model.parameters()):
        raise RuntimeError("real training batch produced non-finite loss or gradients")
    reference = json.loads((inputs / "reference/modal_seed0.json").read_text())
    payload = torch.load(inputs / "reference/modal_seed0.pth", map_location="cpu", weights_only=False)
    if sha256(inputs / "reference/modal_seed0.pth") != reference["checkpoint_sha256"]:
        raise RuntimeError("reference checkpoint checksum disagrees with Modal history")
    if payload["experiment_manifest_fingerprint"] != EXPERIMENT:
        raise RuntimeError("reference checkpoint input mismatch")
    model.load_state_dict(payload["state_dict"])
    metrics = trainer.basic_linear_metrics(model, DataLoader(val, batch_size=8), device)
    difference_pp = 100 * (metrics["gesture_recovery_accuracy"] -
                           reference["validation"]["gesture_recovery_accuracy"])
    result = {"batch_shape": list(batch["frames"].shape), "loss": float(loss),
              "reference_validation": metrics, "difference_pp": difference_pp,
              "reference_epoch": reference["best_epoch"],
              "reference_sha256": reference["checkpoint_sha256"],
              "passed": abs(difference_pp) <= 0.5}
    del model, optimizer, batch, train, val, payload
    torch.cuda.empty_cache()
    return result


class RecoveryBoundary(Exception):
    pass


def recovery_trace(trainer, snapshot, evidence_path):
    import numpy as np
    import torch
    expected = {"model": state_digest(snapshot["model_state_dict"]),
                "optimizer": state_digest(snapshot["optimizer_state_dict"]),
                "shuffle": state_digest(snapshot["shuffle_generator_state"]),
                "torch": state_digest(snapshot["torch_rng_state"]),
                "numpy": state_digest(snapshot["numpy_rng_state"]),
                "cuda": state_digest(snapshot["cuda_rng_state_all"])}
    # At the first epoch loop entry, restoration has actually been executed.
    target_line = next(i for i, line in enumerate(Path(trainer.__file__).read_text().splitlines(), 1)
                       if line.startswith("    for epoch in range(start_epoch"))
    lr_line = next(i for i, line in enumerate(Path(trainer.__file__).read_text().splitlines(), 1)
                  if line.strip() == "model.train()")
    checked = None

    def trace(frame, event, arg):
        nonlocal checked
        if frame.f_code is not trainer.train.__code__:
            return None
        if event == "line" and frame.f_lineno == target_line:
            v = frame.f_locals
            actual = {"model": state_digest(v["model"].state_dict()),
                      "optimizer": state_digest(v["optimizer"].state_dict()),
                      "shuffle": state_digest(v["shuffle_generator"].get_state()),
                      "torch": state_digest(torch.get_rng_state()),
                      "numpy": state_digest(np.random.get_state()),
                      "cuda": state_digest(torch.cuda.get_rng_state_all() if v["device"].type == "cuda" else None)}
            epoch = snapshot["completed_epoch"]
            lr = snapshot["optimizer_state_dict"]["param_groups"][0]["lr"]
            passed = (actual == expected and v["start_epoch"] == epoch + 1 and
                      lr == trainer.lr_for_epoch(1e-3, epoch, 40, "cosine"))
            checked = {"passed": False, "restored_states_passed": passed, "expected": expected, "actual": actual,
                                      "resume_at_epoch": v["start_epoch"], "restored_lr": lr,
                                      "next_lr": trainer.lr_for_epoch(1e-3, epoch + 1, 40, "cosine")}
            write_json(evidence_path, checked)
            if not passed:
                sys.settrace(None)
                raise RuntimeError("resume state restoration mismatch")
        elif event == "line" and frame.f_lineno == lr_line and checked is not None:
            actual_lrs = [g["lr"] for g in frame.f_locals["optimizer"].param_groups]
            checked["actual_next_lrs"] = actual_lrs
            checked["passed"] = checked["restored_states_passed"] and all(lr == checked["next_lr"] for lr in actual_lrs)
            write_json(evidence_path, checked)
            sys.settrace(None)
            if not checked["passed"]:
                raise RuntimeError("resumed cosine learning-rate mismatch")
        return trace
    sys.settrace(trace)


def commit_checkpoint(resume, persistent):
    """Persist one completed epoch. The checkpoint is verified remotely before
    the pointer naming it is written, so the pointer never names a missing file."""
    import torch
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    epoch = payload["completed_epoch"]
    relative = f"checkpoints/epoch-{epoch:02d}.pth"
    record = copy_verified(resume, persistent / relative)
    persist(persistent / relative)
    record.update({"relative_path": relative, "epoch": epoch})
    write_json(persistent / "latest.json", record)
    write_json(persistent / "history.json", {"validation_curve": payload["validation_curve"],
               "epoch_history": payload["epoch_history"], "best_epoch": payload["best"]["epoch"]})
    persist(persistent / "latest.json", persistent / "history.json")
    return record


def run(inputs, persistent, local, session_id, phase):
    import torch
    persistent.mkdir(parents=True, exist_ok=True)
    session = persistent / "sessions" / session_id
    session.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    check_environment(session / "environment.json")
    trainer, staging = stage(inputs, local)
    write_json(session / "staging.json", staging)
    persist(session / "environment.json", session / "staging.json")
    admission_path = persistent / "preflight.json"
    if not admission_path.exists():
        admission = preflight(trainer, inputs, local)
        write_json(admission_path, admission)
        persist(admission_path)
        if not admission["passed"]:
            raise RuntimeError("environment-mismatch: reference differs by more than 0.5 percentage points")
    elif not json.loads(admission_path.read_text())["passed"]:
        raise RuntimeError("prior preflight failed")
    resume = local / "resume.pth"
    pointer = persistent / "latest.json"
    epochs_completed = 0
    if pointer.exists():
        last = json.loads(pointer.read_text())
        remote = persistent / last["relative_path"]
        if sha256(remote) != last["sha256"]:
            raise RuntimeError("last verified checkpoint changed")
        copy_verified(remote, resume)
        snapshot = torch.load(resume, map_location="cpu", weights_only=False)
        epochs_completed = snapshot["completed_epoch"]
        recovery_trace(trainer, snapshot, session / "restoration.json")
    if phase == "completion" and epochs_completed < 5:
        raise RuntimeError("epoch-5 deletion/recovery boundary has not been reached")
    if phase == "timing" and epochs_completed >= 5:
        raise RuntimeError("timing phase already finished; release this runtime before completion")
    torch.cuda.reset_peak_memory_stats()
    write_json(session / "ready.json", {"setup_and_preflight_seconds": time.monotonic() - started,
                                       "resume_at_epoch": epochs_completed + 1})
    persist(session / "ready.json")
    out = local / "model.pth"

    def checkpoint_callback():
        tick = time.monotonic()
        if not resume.exists():
            record = copy_verified(out, persistent / "final/model.pth")
            persist(persistent / "final/model.pth")
            record["relative_path"] = "final/model.pth"
            write_json(persistent / "final/verified.json", record)
            persist(persistent / "final/verified.json")
            return
        record = commit_checkpoint(resume, persistent)
        epoch = record["epoch"]
        if (session / "restoration.json").exists():
            persist(session / "restoration.json")
        record.update({"callback_seconds": time.monotonic() - tick,
                       "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
                       "session_id": session_id, "verified_at": time.time()})
        write_json(persistent / f"epochs/epoch-{epoch:02d}.json", record)
        persist(persistent / f"epochs/epoch-{epoch:02d}.json")
        print(f"DRIVE_VERIFIED epoch={epoch} sha256={record['sha256']}", flush=True)
        if phase == "timing" and epoch == 5:
            raise RecoveryBoundary("epoch 5 saved; delete runtime before resuming epoch 6")

    try:
        result = trainer.train(data=local / "corpus", out=out, epochs=40, batch_size=8,
            lr=1e-3, seed=0, split_seed=0, base_channels=16, temporal_mixer=True,
            image_width=64, image_height=144, cache_frames=False, evaluate_test=False,
            record_train_metrics=False, verify_manifest_content=True,
            experiment_manifest=local / "data/experiment.json", resume_path=resume,
            checkpoint_callback=checkpoint_callback, decode_mode="sequential", lr_schedule="cosine")
        write_json(persistent / "final/metrics.json", {k:v for k,v in result.items() if k != "state_dict"})
        persist(persistent / "final/metrics.json")
        return "completed"
    except RecoveryBoundary:
        return "awaiting_runtime_deletion"
    finally:
        sys.settrace(None)


def report(persistent, reference):
    ledger = json.loads((persistent / "ledger.json").read_text()) if (persistent / "ledger.json").exists() else {}
    history = json.loads((persistent / "history.json").read_text()) if (persistent / "history.json").exists() else {}
    curve = history.get("validation_curve", [])
    epochs = history.get("epoch_history", [])
    sessions = ledger.get("sessions", [])
    now = time.time()
    spent = sum(s.get("released_at", now) - s["allocated_at"] for s in sessions)
    last_release = max((s.get("released_at", now) for s in sessions), default=now)
    calendar = last_release - sessions[0]["allocated_at"] if sessions else 0
    complete = len(curve) == 40 and (persistent / "final/verified.json").exists()
    restorations = [json.loads(p.read_text()) for p in persistent.glob("sessions/*/restoration.json")]
    boundary_sessions = [s for s in sessions if s.get("outcome") == "awaiting_runtime_deletion" and s.get("release_verified")]
    recreated = any(s.get("phase") == "completion" and any(s["boot_id"] != old["boot_id"] and
                    s.get("attempt_at",s["allocated_at"]) >= old["released_at"] for old in boundary_sessions) for s in sessions)
    recovery = recreated and any(r.get("passed") and r.get("resume_at_epoch") == 6 for r in restorations)
    late = statistics.mean(curve[30:40]) if len(curve) == 40 else None
    decrease = 100 * (reference["validation_plateau_mean_last10"] - late) if late is not None else None
    quality_concern = decrease is not None and decrease > 1
    status = ("completed with a quality concern" if quality_concern else "completed and comparable") if complete and recovery else "incomplete"
    times = [e["seconds"] for e in epochs]
    callbacks = [json.loads(p.read_text()) for p in persistent.glob("epochs/*.json")]
    session_evidence = {p.parent.name: {q.stem: json.loads(q.read_text()) for q in p.parent.glob("*.json")}
                        for p in persistent.glob("sessions/*/environment.json")}
    projection = times[0] + statistics.median(times[1:] or times) * 39 if times else None
    result = {"classification": status, "epochs_finished": len(curve), "epochs_target": 40,
              "cause": ledger.get("cause", "recovery unverified" if complete and not recovery else "run in progress"),
              "gpu_allocation_seconds": spent, "calendar_seconds": calendar,
              "manual_effort_seconds": sum(e["seconds"] for e in ledger.get("manual_effort", [])),
              "manual_effort_complete": ledger.get("manual_effort_complete", False),
              "first_epoch_seconds": times[0] if times else None,
              "subsequent_epoch_median_seconds": statistics.median(times[1:]) if len(times)>1 else None,
              "uninterrupted_training_projection_seconds": projection,
              "actual_full_run_turnaround_seconds": calendar if complete else None,
              "validation_curve": curve, "final_epoch_recovery": curve[-1] if curve else None,
              "late_epoch_mean": late, "decrease_vs_modal_pp": decrease,
              "modal_late_epoch_mean": reference["validation_plateau_mean_last10"],
              "modal_epoch_seconds": [e["seconds"] for e in reference["epoch_history"]],
              "modal_same_completed_work_value_usd": 4.15 if complete else None,
              "modal_full_seed_value_usd": 4.15, "colab_cash_spend_usd": 0,
              "cash_savings_after_credits_usd": None, "cash_savings_reason": "Modal credit balance not established",
              "epoch5_runtime_recovery_verified": recovery,
              "interruptions": ledger.get("interruptions", []),
              "repeated_work_seconds": ledger.get("repeated_work_seconds"),
              "checkpoint_records": callbacks,
              "session_evidence": session_evidence,
              "checkpoint_callback_seconds": sum(c.get("callback_seconds",0) for c in callbacks),
              "peak_ram_bytes": max((c.get("peak_ram_bytes",0) for c in callbacks), default=None),
              "peak_vram_allocated_bytes": max((c.get("peak_vram_allocated_bytes",0) for c in callbacks), default=None),
              "peak_vram_reserved_bytes": max((c.get("peak_vram_reserved_bytes",0) for c in callbacks), default=None),
              "caching_difference": "Colab cache_frames=False; Modal seed 0 cache_frames=True",
              "test_evaluated": False, "new_park_holdout_evaluated": False,
              "parallel_or_monthly_capacity_conclusion": None}
    write_json(persistent / "evidence.json", result)
    text = (f"# Isolated Model 1 Colab benchmark\n\n**{status}** — {len(curve)}/40 epochs.\n\n"
            f"Cause: {result['cause']}. GPU allocation {spent/3600:.3f} h; calendar {calendar/3600:.3f} h. "
            f"Epoch-5 recovery verified: {recovery}.\n\n"
            f"First epoch: {result['first_epoch_seconds']} s; subsequent median: {result['subsequent_epoch_median_seconds']} s. "
            f"Uninterrupted training projection: {projection} s; actual full-run turnaround: {result['actual_full_run_turnaround_seconds']} s.\n\n"
            f"Final-epoch validation: {result['final_epoch_recovery']}; epochs 31–40 mean: {late}; "
            f"decrease against Modal: {decrease} percentage points (flag >1). Modal late mean: 90.92%.\n\n"
            "Colab disables decoded-frame caching; Modal seed 0 enabled it. Both use sequential decoding, full precision, "
            "the same sealed cohort and unchanged 40-epoch cosine schedule. Hardware may differ between allocations.\n\n"
            "The Modal value of a full seed is approximately US$4.15. Colab cash spend is zero; "
            "cash savings after Modal credits are unknown. Partial work has no full-seed saving assigned.\n\n"
            "Setup/recovery timings, hardware/software, checkpoint overhead, memory peaks, interruptions, repeated work "
            "and manual-effort completeness are in evidence.json and sessions/. Missing measurements remain unknown. "
            "No test or new-park holdout evaluation; no parallel reliability or monthly capacity conclusion.\n")
    (persistent / "report.md").write_text(text)
    persist(persistent / "evidence.json", persistent / "report.md")
    return result


def main():
    global STORE_ROOT
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("run", "report"))
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--persistent", type=Path, required=True)
    p.add_argument("--local", type=Path, default=Path("/content/model1-benchmark"))
    p.add_argument("--session-id", default="session-1")
    p.add_argument("--phase", choices=("timing", "completion"), default="timing")
    p.add_argument("--store-root", type=Path, help="local mirror root of the scoped remote")
    a = p.parse_args()
    STORE_ROOT = a.store_root
    if a.command == "report":
        report(a.persistent, json.loads((a.inputs / "reference/modal_seed0.json").read_text()))
        return
    session = a.persistent / "sessions" / a.session_id
    try:
        status = run(a.inputs, a.persistent, a.local, a.session_id, a.phase)
        write_json(session / "outcome.json", {"status": status})
        persist(session / "outcome.json")
    except Exception as exc:
        write_json(session / "failure.json",
                   {"error": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()})
        try:
            persist(*(p for p in (session / "environment.json", session / "staging.json",
                                  session / "restoration.json", session / "failure.json") if p.exists()))
        except Exception as upload_error:
            print(f"failure evidence upload failed: {upload_error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
