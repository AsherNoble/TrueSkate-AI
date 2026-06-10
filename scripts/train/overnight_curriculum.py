"""Sequential multi-trick CMA-ES queue for one device — the overnight runner.

Runs an ordered queue of trick curricula on a single device, advancing when a
trick early-stops (rolling land-rate threshold), exhausts its eval budget, or
hits its wall-clock cap. Between tricks it mines the finished run's JSONL for
the finished trick AND the next trick (accidental-discovery harvest), and
chains the best available warm start into each run.

Run one instance per device (each device's WDA/Appium ports are isolated):

    # tmux pane 1 (once, all devices):  python scripts/launch_services.py --devices ...
    # tmux pane 2..N (one per device):
    python scripts/train/overnight_curriculum.py --config configs/overnight/xr1.json

Config schema (configs/overnight/<name>.json):
    {
      "device": "iPhone_XR",
      "seed": 42,
      "defaults": {"pop_size": 12, "max_evals": 1500, "max_hours": 3.0,
                   "stop_land_rate": 0.7, "stop_window": 50,
                   "wait_time": 0.0, "settle_time": 0.5},
      "queue": [
        {"curriculum": "curricula/360_flip.json", "warm_start": "trick_libraries/x.json"},
        {"curriculum": "curricula/hard_flip.json"}
      ]
    }
Per-item keys override defaults; "warm_start" is optional.

Warm-start priority per trick (each candidate must match the curriculum's
num_gestures or it is skipped with a warning):
    1. explicit "warm_start" in the queue item
    2. newest library for the target mined THIS session (<log-root>/<device>/mined/)
    3. newest trick_libraries/<target>_*.json (excluding pre_normalisation/)
    4. none — train_cmaes falls back to the curriculum's own warm_start / prior

Every queue item is wrapped in try/except: a single trick failing (or crashing
twice) never kills the night. Each trick gets its own --log-dir so concurrent
per-device runners never collide on status.json or run-dir names.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

from trueskate_ai.rl.device_worker import select_devices
from trueskate_ai.utils.notify import notify

_PY = sys.executable
_TRAIN_SCRIPT = _HERE / "train_cmaes.py"
_MINE_SCRIPT = _REPO_ROOT / "scripts" / "data" / "build_trick_library.py"
_LIBRARIES_DIR = _REPO_ROOT / "trick_libraries"
_POLL_S = 10.0
_SIGINT_GRACE_S = 60.0   # checkpoint + result.json need time to flush
_RETRY_MIN_BUDGET_S = 20 * 60
_MINE_MIN_SAMPLES = 5


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    for key in ("device", "queue"):
        if key not in cfg:
            sys.exit(f"ERROR: config {path} missing required key '{key}'")
    # Fail fast on unknown device names.
    select_devices(names=[cfg["device"]])
    if not isinstance(cfg["queue"], list) or not cfg["queue"]:
        sys.exit(f"ERROR: config {path} has an empty queue")
    for i, item in enumerate(cfg["queue"]):
        curr = _REPO_ROOT / item["curriculum"]
        if not curr.exists():
            sys.exit(f"ERROR: queue[{i}] curriculum not found: {curr}")
    return cfg


def _curriculum_meta(rel_path: str) -> tuple[str, int]:
    """Return (target, num_gestures) from a curriculum JSON."""
    data = json.loads((_REPO_ROOT / rel_path).read_text())
    return data["target"], int(data.get("num_gestures", 2))


def _library_num_gestures(path: Path) -> int | None:
    try:
        data = json.loads(path.read_text())
        return len(data["median_gestures"]["gestures"])
    except Exception:
        return None


def _newest_matching_library(target: str, search_dirs: list[Path], num_gestures: int) -> Path | None:
    """Newest library file for the target whose gesture count matches."""
    safe = _sanitize_filename(target)
    candidates: list[Path] = []
    for d in search_dirs:
        if d.exists():
            candidates.extend(d.glob(f"{safe}_*.json"))
    for lib in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        n = _library_num_gestures(lib)
        if n == num_gestures:
            return lib
        print(f"  [warm-start] skipping {lib.name}: {n} gestures, need {num_gestures}")
    return None


def _pick_warm_start(item: dict, target: str, num_gestures: int, mined_dir: Path) -> Path | None:
    explicit = item.get("warm_start")
    if explicit:
        path = _REPO_ROOT / explicit
        if not path.exists():
            print(f"  [warm-start] explicit {explicit} missing — falling through")
        elif _library_num_gestures(path) != num_gestures:
            print(f"  [warm-start] explicit {explicit} gesture-count mismatch — falling through")
        else:
            return path
    return _newest_matching_library(target, [mined_dir, _LIBRARIES_DIR], num_gestures)


def _train_cmd(item: dict, defaults: dict, device: str, seed: int,
               warm_start: Path | None, log_dir: Path) -> list[str]:
    def opt(key):
        return item.get(key, defaults.get(key))

    cmd = [
        _PY, str(_TRAIN_SCRIPT),
        "--curriculum", str(_REPO_ROOT / item["curriculum"]),
        "--devices", device,
        "--seed", str(seed),
        "--log-dir", str(log_dir),
    ]
    for key, flag in (("max_evals", "--max-evals"), ("pop_size", "--pop-size"),
                      ("stop_land_rate", "--stop-land-rate"), ("stop_window", "--stop-window"),
                      ("wait_time", "--wait-time"), ("settle_time", "--settle-time")):
        value = opt(key)
        if value is not None:
            cmd.extend([flag, str(value)])
    if warm_start is not None:
        cmd.extend(["--initial-mean", str(warm_start)])
    return cmd


def _run_trick(cmd: list[str], log_dir: Path, max_hours: float | None) -> dict:
    """Run one train_cmaes subprocess; enforce the wall-clock cap via SIGINT.

    Returns the run's result.json contents, or {"stop_reason": "crashed"/"time_cap"}.
    """
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, start_new_session=True)
    deadline = time.monotonic() + max_hours * 3600 if max_hours else None
    timed_out = False
    while proc.poll() is None:
        if deadline and time.monotonic() > deadline:
            timed_out = True
            print(f"  [time-cap] {max_hours}h reached — SIGINT for graceful checkpoint")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except ProcessLookupError:
                break
            try:
                proc.wait(timeout=_SIGINT_GRACE_S)
            except subprocess.TimeoutExpired:
                print("  [time-cap] no exit after SIGINT grace — terminating")
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
            break
        time.sleep(_POLL_S)

    run_dirs = sorted((log_dir / "runs").glob("cmaes_run_*")) if (log_dir / "runs").exists() else []
    result_path = run_dirs[-1] / "result.json" if run_dirs else None
    if result_path and result_path.exists():
        result = json.loads(result_path.read_text())
        if timed_out:
            result["stop_reason"] = "time_cap"
        return result
    return {"stop_reason": "time_cap" if timed_out else "crashed",
            "total_evals": 0, "best_reward": 0.0, "best_trick": None,
            "log_path": None}


def _mine(log_path: str | None, trick: str, out_dir: Path) -> Path | None:
    """Mine a finished JSONL for a trick's library. Never raises."""
    if not log_path or not Path(log_path).exists():
        return None
    before = set(out_dir.glob("*.json")) if out_dir.exists() else set()
    try:
        proc = subprocess.run(
            [_PY, str(_MINE_SCRIPT), "--log", log_path, "--trick", trick,
             "--landed-only", "--min-samples", str(_MINE_MIN_SAMPLES),
             "--out-dir", str(out_dir)],
            capture_output=True, text=True, timeout=300,
        )
    except Exception as exc:
        print(f"  [mine] {trick}: subprocess error: {exc}")
        return None
    if proc.returncode == 2:
        print(f"  [mine] {trick}: not enough landed samples — skipped")
        return None
    if proc.returncode != 0:
        print(f"  [mine] {trick}: failed: {proc.stderr.strip()[:200]}")
        return None
    new = set(out_dir.glob("*.json")) - before
    path = max(new, key=lambda p: p.stat().st_mtime) if new else None
    if path:
        print(f"  [mine] {trick}: → {path}")
    return path


def _start_caffeinate() -> subprocess.Popen | None:
    """Keep the Mac fully awake while we run; auto-exits when this PID dies."""
    try:
        proc = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
        print(f"[overnight] caffeinate active (PID {proc.pid}).")
        return proc
    except FileNotFoundError:
        print("[overnight] caffeinate not found — Mac may sleep. (macOS only.)")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential multi-trick CMA-ES queue for one device."
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Queue config JSON (configs/overnight/<name>.json)")
    parser.add_argument("--log-root", type=Path, default=_REPO_ROOT / "logs" / "overnight",
                        help="Root for per-trick log dirs (default: logs/overnight)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the per-trick commands and warm-start choices; run nothing")
    parser.add_argument("--no-caffeinate", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = cfg["device"]
    seed = int(cfg.get("seed", 42))
    defaults = cfg.get("defaults", {})
    queue = cfg["queue"]

    device_root = args.log_root / device
    mined_dir = device_root / "mined"

    caffeinate = None if (args.dry_run or args.no_caffeinate) else _start_caffeinate()

    summary: list[dict] = []
    if not args.dry_run:
        notify(f"Overnight queue starting on {device}: "
               f"{[Path(i['curriculum']).stem for i in queue]}",
               title="TrueSkate overnight", tags=["crescent_moon"])

    try:
        for idx, item in enumerate(queue):
            target, num_gestures = _curriculum_meta(item["curriculum"])
            trick_slug = _sanitize_filename(target)
            log_dir = device_root / f"{idx:02d}_{trick_slug}"
            print(f"\n=== [{device}] trick {idx + 1}/{len(queue)}: {target} "
                  f"(N={num_gestures}) ===")

            try:
                warm = _pick_warm_start(item, target, num_gestures, mined_dir)
                print(f"  warm start: {warm if warm else 'none (curriculum default / cold)'}")
                cmd = _train_cmd(item, defaults, device, seed + idx, warm, log_dir)

                if args.dry_run:
                    print(f"  $ {' '.join(cmd)}")
                    summary.append({"target": target, "dry_run": True})
                    continue

                notify(f"Trick {idx + 1}/{len(queue)} starting on {device}: {target}, "
                       f"warm={'yes' if warm else 'cold'}.",
                       title="TrueSkate overnight", tags=["hammer_and_wrench"])

                max_hours = item.get("max_hours", defaults.get("max_hours"))
                started = time.monotonic()
                result = _run_trick(cmd, log_dir, max_hours)

                # One retry on infrastructure-shaped failures with budget left.
                budget_left = (max_hours * 3600 - (time.monotonic() - started)) if max_hours else None
                if (result["stop_reason"] in ("crashed", "all_workers_dead")
                        and (budget_left is None or budget_left > _RETRY_MIN_BUDGET_S)):
                    retry_hours = budget_left / 3600 if budget_left else max_hours
                    print(f"  [retry] {result['stop_reason']} — one retry with seed+100")
                    cmd = _train_cmd(item, defaults, device, seed + idx + 100, warm, log_dir)
                    result = _run_trick(cmd, log_dir, retry_hours)

                # Harvest: the finished trick, and the next trick if its basin
                # was visited accidentally during this run.
                mined_own = _mine(result.get("log_path"), target, mined_dir)
                if idx + 1 < len(queue):
                    next_target, _ = _curriculum_meta(queue[idx + 1]["curriculum"])
                    _mine(result.get("log_path"), next_target, mined_dir)

                summary.append({
                    "target": target,
                    "stop_reason": result["stop_reason"],
                    "evals": result.get("total_evals", 0),
                    "land_rate_window": result.get("land_rate_window"),
                    "best_reward": result.get("best_reward"),
                    "library": str(mined_own) if mined_own else None,
                })
                notify(f"Trick done on {device}: {target} — {result['stop_reason']}, "
                       f"evals={result.get('total_evals', 0)}, "
                       f"land_rate={result.get('land_rate_window')}, "
                       f"lib={'yes' if mined_own else 'no'}.",
                       title="TrueSkate overnight", tags=["checkered_flag"])

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"  [queue] {target} failed: {exc} — continuing with next trick")
                summary.append({"target": target, "stop_reason": f"orchestrator_error: {exc}"})
                notify(f"Trick FAILED on {device}: {target} — {exc}. Queue continues.",
                       title="TrueSkate overnight", priority="high", tags=["warning"])

    except KeyboardInterrupt:
        print("\n[overnight] interrupted — exiting after current cleanup.")

    finally:
        if caffeinate and caffeinate.poll() is None:
            caffeinate.terminate()
        print(f"\n=== [{device}] overnight summary ===")
        for entry in summary:
            print(f"  {json.dumps(entry)}")
        if not args.dry_run:
            lines = "; ".join(
                f"{e['target']}: {e.get('stop_reason')} ({e.get('evals', '?')} evals)"
                for e in summary
            )
            notify(f"Overnight queue finished on {device} at "
                   f"{datetime.now().strftime('%H:%M')}: {lines or 'nothing ran'}",
                   title="TrueSkate overnight", tags=["sunrise"], block=True)


if __name__ == "__main__":
    main()
