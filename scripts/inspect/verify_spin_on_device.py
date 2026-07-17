"""On-device verification of the guaranteed-spin SLS samples + fixed PPO spin mech.

Run on the rig (training-server) against ONE phone while its collector is stopped.
Designed to import the RIG checkout's src (whatever branch it is on) so it tests the
sampler OUTPUT (pre-generated vectors) against the production execution path, without
requiring the sampler branch to be checked out on the rig.

Phases:
  1. spin      — fire pre-generated guaranteed-spin vectors (sample_spin output) via
                 execute_gesture_params (the exact collector call), counting WDA errors
                 and park-editor / replay-menu triggers with the production detectors.
  2. control   — a batch of no-spin nslot vectors (--control-fires) -> baseline
                 editor/menu rate.
  3. ppo       — load the FIXED trick_conditioned_action module from a file path and
                 fire spin-enabled 42-dim actions through execute_gesture_params_vector.
  4. visual    — fire a converged library recipe twice (control vs +spin block held
                 t=0.05..0.95) while saving MJPEG frames, so the view-rotation effect of
                 the held rotate button can be eyeballed frame-by-frame.

Outputs: <out>/summary.json, <out>/fires.jsonl, <out>/frames/{control,spin}/*.jpg.

Usage (on the rig):
    launchctl bootout gui/$UID/com.trueskate.collect.xr1   # stop XR1's collector first
    python verify_spin_on_device.py --devices iPhone_XR \
        --vectors spin_vectors.json --ppo-module trick_conditioned_action_fixed.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# Mirror the SLS collector's production env: touch_actions reads this at import time.
os.environ.setdefault("TRUESKATE_MIN_FINGER_STAGGER_S", "0.12")

# The RIG's checkout, deliberately NOT this file's repo — see module docstring.
_REPO = Path(os.environ.get("TRUESKATE_REPO", str(Path.home() / "trueskate-ai")))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402
import requests  # noqa: E402

try:  # UDID env vars help ensure_foreground/reconnect; live-port connect works without.
    from dotenv import load_dotenv
    load_dotenv(_REPO / ".env")
except Exception:  # noqa: BLE001
    pass

from trueskate_ai.rl.cmaes.action_param import execute_gesture_params  # noqa: E402
from trueskate_ai.rl.device_worker import (  # noqa: E402
    BUNDLE_ID, DeviceWorker, add_device_selection_args, resolve_devices,
)
from trueskate_ai.sim.touch_actions import reset_position, skip_loading_screen  # noqa: E402
from trueskate_ai.vision.gameplay_filter import is_menu_frame  # noqa: E402

# is_editor_frame exists on the rig branch (uncommitted) and on the sampler branch;
# degrade gracefully if this checkout predates it rather than failing the whole test.
try:
    from trueskate_ai.vision.gameplay_filter import is_editor_frame
except ImportError:  # pragma: no cover - old checkout
    def is_editor_frame(_img) -> bool:  # type: ignore[misc]
        return False

# Phase-4 recipe loading is optional capability on older checkouts too.
from trueskate_ai.data.gesture_sampling import load_recipe_vectors  # noqa: E402

_PPO_ACTION_DIM = 42     # 4 slots x 9 + 3 delays + 3 spin params
_RESET_SETTLE_S = 0.3    # board settle after reset_position (mirrors the collector)
_GUARD_SETTLE_S = 0.35   # editor-UI render wait before scoring (the collector's
                         # post-gesture guard uses the same 0.35s)
_VISUAL_TAIL_S = 2.0     # longer tail in the visual phase so the trick plays out
                         # fully into the MJPEG frame strip


class MjpegFrameSaver:
    """Save raw JPEG frames from WDA's MJPEG stream while a gesture executes."""

    def __init__(self, url: str, out_dir: Path, keep_every: int = 2) -> None:
        self.url = url
        self.out_dir = out_dir
        self.keep_every = keep_every
        self._stop = False
        self._thread: threading.Thread | None = None
        self.saved = 0

    def __enter__(self) -> "MjpegFrameSaver":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        buf, n, t0 = b"", 0, time.monotonic()
        resp = None
        try:
            resp = requests.get(self.url, stream=True, timeout=5)
            for chunk in resp.iter_content(chunk_size=4096):
                if self._stop:
                    break
                buf += chunk
                while True:
                    s = buf.find(b"\xff\xd8")
                    if s == -1:
                        buf = b""
                        break
                    e = buf.find(b"\xff\xd9", s + 2)
                    if e == -1:
                        buf = buf[s:]
                        break
                    jpeg, buf = buf[s : e + 2], buf[e + 2 :]
                    n += 1
                    if n % self.keep_every == 0:
                        t = time.monotonic() - t0
                        (self.out_dir / f"f_{self.saved:03d}_t{t:06.3f}.jpg").write_bytes(jpeg)
                        self.saved += 1
        except Exception as exc:  # noqa: BLE001 — frames are evidence, not a gate
            print(f"  [mjpeg] stream ended: {exc!r}")
        finally:
            if resp is not None:
                try:
                    resp.close()
                except Exception:  # noqa: BLE001
                    pass

    def __exit__(self, *exc) -> None:
        self._stop = True
        if self._thread is not None:
            # a hair over the stream's 5s read timeout, so a blocked read can
            # surface before the thread is abandoned
            self._thread.join(timeout=6.0)


def _relaunch(worker: DeviceWorker) -> None:
    """Coordinate-free return to live gameplay (mirrors the collector's _exit_replay)."""
    d = worker.driver
    try:
        d.terminate_app(BUNDLE_ID)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(1.0)
    d.activate_app(BUNDLE_ID)
    time.sleep(1.0)
    skip_loading_screen(d, worker.device_w, worker.device_h)
    time.sleep(1.0)


def _fire(worker: DeviceWorker, params: list[float], log, tag: str, tail_s: float) -> dict:
    """One collector-faithful fire: reset -> gesture -> tail -> editor/menu scoring."""
    rec: dict = {"tag": tag, "n_params": len(params), "ok": False,
                 "editor": False, "menu": False, "error": None, "exec_s": None}
    dw, dh = worker.device_w, worker.device_h
    t0 = time.monotonic()
    try:
        # Reset inside the try so one WDA blip records an error on this fire
        # instead of killing the whole run.
        reset_position(worker.driver, dw, dh)
        time.sleep(_RESET_SETTLE_S)
        t0 = time.monotonic()  # re-stamp: exec_s times the gesture, not the reset
        execute_gesture_params(
            worker.driver, np.asarray(params, dtype=np.float64), dw, dh,
            spin_button_xy=worker.spin_button_xy, timing_device_key=worker.device_id,
        )
        rec["ok"] = True
    except Exception as exc:  # noqa: BLE001 — THE mechanics signal; record verbatim
        rec["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    rec["exec_s"] = round(time.monotonic() - t0, 3)
    time.sleep(tail_s)
    # Mirror the collector's post-gesture guard (editor opens DURING the gesture).
    time.sleep(_GUARD_SETTLE_S)
    try:
        png = worker.driver.get_screenshot_as_png()
        rec["editor"] = bool(is_editor_frame(png))
        rec["menu"] = bool(is_menu_frame(png))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = rec["error"] or f"screenshot: {type(exc).__name__}: {exc}"
    log.write(json.dumps(rec) + "\n")
    log.flush()
    status = "OK " if rec["ok"] else "ERR"
    flags = ("+EDITOR" if rec["editor"] else "") + ("+MENU" if rec["menu"] else "")
    print(f"  [{tag}] {status} exec={rec['exec_s']}s {flags}")
    if rec["menu"]:
        _relaunch(worker)  # menu needs a relaunch; the editor closes on next reset
    return rec


def _phase_stats(records: list[dict]) -> dict:
    n = max(1, len(records))
    return {
        "fires": len(records),
        "errors": sum(1 for r in records if not r["ok"]),
        "editor": sum(1 for r in records if r["editor"]),
        "menu": sum(1 for r in records if r["menu"]),
        "editor_rate": round(sum(1 for r in records if r["editor"]) / n, 3),
    }


def _load_ppo_module(path: Path):
    spec = importlib.util.spec_from_file_location("tca_fixed", path)
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: dataclass processing looks the module up in
    # sys.modules (fails with AttributeError on an unregistered module).
    sys.modules["tca_fixed"] = mod
    spec.loader.exec_module(mod)  # its trueskate_ai.* imports resolve from rig src
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="On-device spin verification.")
    ap.add_argument("--vectors", type=Path, required=True,
                    help="JSON with spin_n2/spin_n3/nslot_n2 vector lists (pre-generated "
                         "by the sampler branch's sample_spin/sample_nslot).")
    ap.add_argument("--ppo-module", type=Path, default=None,
                    help="Path to the FIXED trick_conditioned_action.py (phase 3).")
    ap.add_argument("--out", type=Path, default=Path.home() / "spin_verify" / "results")
    ap.add_argument("--spin-fires", type=int, default=40)
    ap.add_argument("--control-fires", type=int, default=20)
    ap.add_argument("--tail-s", type=float, default=1.0)
    ap.add_argument("--skip-visual", action="store_true")
    add_device_selection_args(ap)
    args = ap.parse_args()

    devices = resolve_devices(devices_arg=args.devices, personal=args.personal,
                              all_devices=args.all_devices)
    if len(devices) != 1:
        raise SystemExit("verify_spin_on_device tests ONE phone; pass a single --devices NAME.")
    cfg = devices[0]

    # Never run against a phone whose collector is live (session conflict + wedge risk).
    # NB: end-anchor the device name — a bare substring match on "iPhone_XR" would
    # also match the iPhone_XR2 collector's command line.
    r = subprocess.run(["pgrep", "-f", rf"collect_sls_xctest\.py.*--devices[= ]{cfg['name']}($| )"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        raise SystemExit(f"A collector is RUNNING for {cfg['name']} (pids {r.stdout.split()}). "
                         f"Stop com.trueskate.collect.* for this phone first.")

    vectors = json.loads(args.vectors.read_text())
    spin_vecs = (vectors.get("spin_n2") or [])[: args.spin_fires]
    spin_vecs += (vectors.get("spin_n3") or [])[: max(0, args.spin_fires - len(spin_vecs))]
    ctrl_vecs = (vectors.get("nslot_n2") or [])[: args.control_fires]

    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "fires.jsonl").open("w")

    worker = DeviceWorker(cfg)
    print(f"Connecting to {cfg['name']} ...")
    worker.connect()
    worker.ensure_foreground()
    time.sleep(1.0)
    if is_menu_frame(worker.driver.get_screenshot_as_png()):
        print("Starting in replay/menu — relaunching True Skate first.")
        _relaunch(worker)

    summary: dict = {"device": cfg["name"], "stagger_env":
                     os.environ.get("TRUESKATE_MIN_FINGER_STAGGER_S")}
    try:
        print(f"\n=== Phase 1: {len(spin_vecs)} guaranteed-spin fires ===")
        spin_recs = [_fire(worker, v, log, f"spin{i:02d}", args.tail_s)
                     for i, v in enumerate(spin_vecs)]
        summary["spin"] = _phase_stats(spin_recs)

        print(f"\n=== Phase 2: {len(ctrl_vecs)} no-spin control fires ===")
        ctrl_recs = [_fire(worker, v, log, f"ctrl{i:02d}", args.tail_s)
                     for i, v in enumerate(ctrl_vecs)]
        summary["control"] = _phase_stats(ctrl_recs)

        if args.ppo_module is not None:
            print("\n=== Phase 3: fixed-PPO spin fires ===")
            tca = _load_ppo_module(args.ppo_module)
            act = np.full(_PPO_ACTION_DIM, -1.0)
            act[0:8] = [0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.0, 0.0]  # mid-screen drag slot
            act[8] = 1.0                                          # slot 0 enabled
            act[39], act[40], act[41] = 1.0, -1.0, 0.0            # spin ON, hold 0..0.5
            ppo_recs = []
            for i in range(2):
                rec = {"tag": f"ppo{i}", "ok": False, "error": None}
                try:
                    reset_position(worker.driver, worker.device_w, worker.device_h)
                    time.sleep(_RESET_SETTLE_S)
                    tca.execute_gesture_params_vector(
                        worker.driver, act, device_w=worker.device_w,
                        device_h=worker.device_h, spin_button_xy=worker.spin_button_xy)
                    rec["ok"] = True
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                time.sleep(args.tail_s)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"  [ppo{i}] {'OK' if rec['ok'] else 'ERR ' + str(rec['error'])}")
                ppo_recs.append(rec)
            summary["ppo"] = {"fires": len(ppo_recs),
                              "errors": sum(1 for r in ppo_recs if not r["ok"])}

        if not args.skip_visual:
            print("\n=== Phase 4: recipe control vs +spin, MJPEG frames ===")
            recipes = [rv for rv in load_recipe_vectors(_REPO / "trick_libraries")
                       if not rv[2]]  # no-spin recipes only
            if not recipes:
                print("  no packable no-spin recipe found — skipping visual phase")
            else:
                vec, n, _, name = recipes[0]
                print(f"  recipe: {name} (N={n})")
                spin_vec = list(vec) + [1.0, 0.05, 0.95]  # gate ON, hold ~whole gesture
                for tag, v in (("control", list(vec)), ("spin", spin_vec)):
                    reset_position(worker.driver, worker.device_w, worker.device_h)
                    time.sleep(_RESET_SETTLE_S)
                    with MjpegFrameSaver(worker.mjpeg_url, args.out / "frames" / tag) as sav:
                        _fire(worker, v, log, f"visual-{tag}", tail_s=_VISUAL_TAIL_S)
                        time.sleep(0.5)
                    print(f"  [{tag}] saved {sav.saved} frames -> {args.out / 'frames' / tag}")
                summary["visual_recipe"] = name
    finally:
        log.close()
        worker.disconnect()

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if summary.get("spin", {}).get("fires", 0) == 0:
        print("\nVERDICT: FAIL — no spin fires executed (bad --vectors?)")
        raise SystemExit(1)
    errs = (summary.get("spin", {}).get("errors", 0)
            + summary.get("control", {}).get("errors", 0)
            + summary.get("ppo", {}).get("errors", 0))
    if errs:
        print(f"\nVERDICT: FAIL — {errs} execution error(s); see fires.jsonl")
        raise SystemExit(1)
    print("\nVERDICT: mechanics PASS (no WDA errors). Compare editor rates + eyeball frames.")


if __name__ == "__main__":
    main()
