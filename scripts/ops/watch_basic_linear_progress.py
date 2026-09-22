#!/usr/bin/env python3
"""Stop one linear collector at a strict target and announce 10% milestones.

The collector writes complete one-minute segments, so this watcher only sees new
admissions after alignment has finished.  State is persisted before the next
poll so restarting the watcher does not repeat already announced milestones.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from trueskate_ai.model1.linear.dataset import discover_basic_linear_samples
from trueskate_ai.utils.notify import notify


def strict_device_count(root: Path, device: str) -> int:
    samples, _ = discover_basic_linear_samples(root)
    return sum(
        json.loads((sample / "meta.json").read_text()).get("device") == device
        for sample in samples
    )


def completed_milestone(total: int, target: int, step: int = 10) -> int:
    if target <= 0 or step <= 0 or 100 % step:
        raise ValueError("target and step must be positive, and step must divide 100")
    return min(100, (max(0, total) * 100 // target) // step * step)


def milestones_due(last_pct: int, total: int, target: int, step: int = 10) -> list[int]:
    reached = completed_milestone(total, target, step)
    return list(range(last_pct + step, reached + 1, step))


def read_pid(path: Path) -> int:
    value = path.read_text().strip()
    if not value.isdecimal() or int(value) <= 1:
        raise RuntimeError(f"invalid collector PID in {path}: {value!r}")
    return int(value)


def collector_command(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def collector_alive(pid: int, *, device: str, output: Path) -> bool:
    command = collector_command(pid)
    expected = f"mvp_collect_linear.sh {device} {output}"
    return bool(command and expected in command)


def stop_collector(pid: int, *, device: str, output: Path) -> None:
    if not collector_alive(pid, device=device, output=output):
        raise RuntimeError(f"PID {pid} does not identify the expected {device} collector")
    os.kill(pid, signal.SIGTERM)


def write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_or_initialize_state(
    path: Path, *, label: str, device: str, target: int, baseline: int, current: int
) -> dict:
    identity = {
        "label": label,
        "device": device,
        "target": target,
        "baseline": baseline,
    }
    if path.exists():
        state = json.loads(path.read_text())
        if any(state.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"milestone state identity differs from this run: {path}")
        return state
    state = {
        **identity,
        # Existing clips establish the starting percentage; do not send old
        # milestones retrospectively when a new bounded collection begins.
        "last_notified_pct": completed_milestone(baseline + current, target),
        "accepted_in_output": current,
        "total": baseline + current,
    }
    write_state(path, state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--baseline-count", type=int, default=0)
    parser.add_argument("--collector-pid-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target <= 0 or args.baseline_count < 0 or args.poll_seconds <= 0:
        raise SystemExit("target and poll-seconds must be positive; baseline must be non-negative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pid = read_pid(args.collector_pid_file)
    current = strict_device_count(output, args.device)
    state = load_or_initialize_state(
        args.state_file,
        label=args.label,
        device=args.device,
        target=args.target,
        baseline=args.baseline_count,
        current=current,
    )

    while True:
        current = strict_device_count(output, args.device)
        total = args.baseline_count + current
        for percentage in milestones_due(
            int(state["last_notified_pct"]), total, args.target
        ):
            notify(
                f"{args.label} is at {percentage}% completion",
                title="TrueSkate collection",
                block=True,
            )
            state["last_notified_pct"] = percentage
            state["accepted_in_output"] = current
            state["total"] = total
            write_state(args.state_file, state)

        state["accepted_in_output"] = current
        state["total"] = total
        write_state(args.state_file, state)
        print(
            f"[{time.strftime('%F %T')}] {args.label} accepted={current} "
            f"baseline={args.baseline_count} total={total} target={args.target}",
            flush=True,
        )
        if total >= args.target:
            stop_collector(pid, device=args.device, output=output)
            return 0
        if not collector_alive(pid, device=args.device, output=output):
            raise RuntimeError(
                f"{args.label} collector stopped before target: total={total}/{args.target}"
            )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
