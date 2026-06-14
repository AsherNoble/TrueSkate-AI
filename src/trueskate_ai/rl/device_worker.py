"""Per-device worker for parallelized CMA-ES evaluation.

Each DeviceWorker encapsulates one iPhone's Appium connection and can
independently execute a candidate eval. The orchestrator dispatches
candidates to workers via ThreadPoolExecutor and handles JSONL logging,
resets, and CMA-ES bookkeeping.

Public API:
    DEVICES          — list of device config dicts for all active iPhones.
    FrameRecorder    — MJPEG frame capture (one instance per eval).
    DeviceWorker     — connects to one device, runs evaluate().
"""
import io
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import requests
from appium import webdriver
from appium.options.ios import XCUITestOptions
from dotenv import load_dotenv
from PIL import Image

from typing import Callable

from trueskate_ai.rl.cmaes.action_param import execute_gesture_params
from trueskate_ai.rl.reward import capture_and_detect_with_diagnostics
from trueskate_ai.sim.touch_actions import calibrate_touch_timing, reset_position, skip_loading_screen
from trueskate_ai.sim.trick_info_reader import TrickResult
from trueskate_ai.vision.color_recorder import TimestampedColorRecorder
from trueskate_ai.vision.scene_classifier import SceneGuard

# Set TRACE_COLLECT=1 to passively capture COLOR frames each eval (downscaled,
# trace-window subset saved by the optimizer) so CMA-ES runs double as
# self-labeled trace-extraction data — the executed gesture vector is the label.
_TRACE_COLLECT = bool(os.environ.get("TRACE_COLLECT"))
_TRACE_RESIZE_WIDTH = 512

# ---------------------------------------------------------------------------
# Device configurations
# ---------------------------------------------------------------------------

# Each device carries a "role":
#   "collection" — part of the 24h home data-collection fleet (default roster).
#   "personal"   — a phone reserved for ad-hoc testing (e.g. Asher's iPhone 11),
#                  excluded from the default roster so a training run never grabs
#                  it. Select it explicitly with --personal / --devices.
DEVICES: list[dict] = [
    {
        "env_key": "IPHONE_XR_UDID",
        "name": "iPhone_XR",
        "role": "collection",
        "wda_port": 8100,
        "mjpeg_port": 9100,
        "appium_port": 4723,
        "logical_w": 414,
        "logical_h": 896,
        "spin_button_xy": (0.0604, 0.4040),
    },
    {
        "env_key": "IPHONE_11_UDID",
        "name": "iPhone_11",
        "role": "personal",  # Asher's personal device — testing only, not 24h collection.
        "wda_port": 8101,
        "mjpeg_port": 9101,
        "appium_port": 4724,
        "logical_w": 375,  # Display Zoom always on; reduces logical_w 414 → 375
        "logical_h": 812,  # Display Zoom always on; reduces logical_h 896 → 812
        "spin_button_xy": (0.0604, 0.4040),
    },
    {
        "env_key": "IPHONE_XS_UDID",
        "name": "iPhone_XS",
        "role": "collection",
        "wda_port": 8102,
        "mjpeg_port": 9102,
        "appium_port": 4725,
        "logical_w": 375,
        "logical_h": 812,
        "spin_button_xy": (0.0604, 0.4040),
    },
    {
        "env_key": "IPHONE_XR2_UDID",
        "name": "iPhone_XR2",
        "role": "collection",
        "wda_port": 8103,
        "mjpeg_port": 9103,
        "appium_port": 4726,
        "logical_w": 414,  # Display Zoom must be OFF (dim guard kills services on mismatch)
        "logical_h": 896,
        "spin_button_xy": (0.0604, 0.4040),
    },
]

DEFAULT_ROLE = "collection"


def select_devices(
    *,
    names: list[str] | None = None,
    roles: list[str] | None = None,
    devices: list[dict] | None = None,
) -> list[dict]:
    """Resolve which device configs a run/launcher should use.

    Precedence:
        1. ``names`` — explicit device names (case-insensitive, order preserved
           as listed by the caller). Unknown names raise ValueError.
        2. ``roles`` — every device whose ``role`` is in this set.
        3. default — every ``collection`` device (the 24h home roster; excludes
           ``personal`` phones such as the iPhone 11).

    Devices missing a ``role`` key are treated as ``DEFAULT_ROLE``.
    """
    pool = DEVICES if devices is None else devices

    if names:
        by_name = {d["name"].lower(): d for d in pool}
        selected: list[dict] = []
        unknown: list[str] = []
        for name in names:
            cfg = by_name.get(name.strip().lower())
            if cfg is None:
                unknown.append(name)
            else:
                selected.append(cfg)
        if unknown:
            valid = ", ".join(d["name"] for d in pool)
            raise ValueError(
                f"Unknown device name(s): {', '.join(unknown)}. Valid: {valid}"
            )
        return selected

    if roles:
        wanted = {r.lower() for r in roles}
        return [d for d in pool if d.get("role", DEFAULT_ROLE).lower() in wanted]

    return [d for d in pool if d.get("role", DEFAULT_ROLE) == "collection"]


def resolve_devices(
    *,
    devices_arg: str | None = None,
    personal: bool = False,
    all_devices: bool = False,
) -> list[dict]:
    """Map common CLI flags to a device list, shared by the entry-point scripts.

    ``--devices "iPhone_XR,iPhone_XS"`` → explicit names.
    ``--personal``    → every device with role ``personal``.
    ``--all-devices`` → every configured device regardless of role.
    (none)            → the default ``collection`` roster.

    ``devices_arg`` wins if given; otherwise ``personal``; otherwise
    ``all_devices``; otherwise the default.
    """
    if devices_arg:
        names = [n for n in (s.strip() for s in devices_arg.split(",")) if n]
        return select_devices(names=names)
    if personal:
        return select_devices(roles=["personal"])
    if all_devices:
        return list(DEVICES)
    return select_devices()


def add_device_selection_args(parser) -> None:
    """Register the shared --devices / --personal / --all-devices flags.

    Mutually exclusive; resolve the result with ``resolve_devices``.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--devices",
        help="Comma-separated device names (e.g. 'iPhone_XR,iPhone_XS'). "
        "Defaults to the 'collection' roster (excludes personal phones).",
    )
    group.add_argument(
        "--personal",
        action="store_true",
        help="Use personal-role device(s) only — e.g. test on the iPhone 11 on this Mac.",
    )
    group.add_argument(
        "--all-devices",
        action="store_true",
        help="Use every configured device regardless of role.",
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUNDLE_ID = "com.trueaxis.skate"
_APP_STATE_FOREGROUND = 4
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEAD_THRESHOLD = 5      # consecutive failures before a worker is considered dead
_REVIVE_COOLDOWN = 60.0  # seconds between scheduled reconnect attempts for dead workers
ALL_DEAD_TIMEOUT = 300.0  # seconds before aborting when every worker is dead (public: read by orchestrators)

# ---------------------------------------------------------------------------
# Frame recording
# ---------------------------------------------------------------------------


class FrameRecorder:
    """Reads 210x455 grayscale frames from WDA's MJPEG stream during an eval.

    Connects to the MJPEG server started by WDA, extracts JPEG frames by
    scanning for SOI/EOI markers (0xFF 0xD8 / 0xFF 0xD9), and decodes each
    to a 210x455 grayscale numpy array. Typical throughput: 30-60 fps.
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_flag = False
        self._frames: list[np.ndarray] = []
        self._response: requests.Response | None = None

    def start(self, mjpeg_url: str) -> None:
        self._stop_flag = False
        self._frames = []
        self._response = None
        self._thread = threading.Thread(
            target=self._capture_loop, args=(mjpeg_url,), daemon=True
        )
        self._thread.start()

    def _capture_loop(self, mjpeg_url: str) -> None:
        buf = b""
        try:
            resp = requests.get(mjpeg_url, stream=True, timeout=5)
            self._response = resp
            for chunk in resp.iter_content(chunk_size=4096):
                if self._stop_flag:
                    break
                buf += chunk
                while True:
                    start_idx = buf.find(b"\xff\xd8")
                    if start_idx == -1:
                        buf = b""
                        break
                    end_idx = buf.find(b"\xff\xd9", start_idx + 2)
                    if end_idx == -1:
                        buf = buf[start_idx:]
                        break
                    jpeg_bytes = buf[start_idx : end_idx + 2]
                    buf = buf[end_idx + 2:]
                    try:
                        img = (
                            Image.open(io.BytesIO(jpeg_bytes))
                            .convert("L")
                            .resize((210, 455), Image.LANCZOS)
                        )
                        self._frames.append(np.array(img, dtype=np.uint8))
                    except Exception as e:
                        logging.debug(f"Failed to decode JPEG frame: {e}")
        except Exception as e:
            logging.warning(f"MJPEG capture failed: {e}")
        finally:
            self._response = None

    def stop(self) -> list[np.ndarray]:
        self._stop_flag = True
        resp = self._response
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self._frames


# ---------------------------------------------------------------------------
# DeviceWorker
# ---------------------------------------------------------------------------


class DeviceWorker:
    """Encapsulates one iPhone's Appium connection and eval logic.

    The orchestrator creates one DeviceWorker per physical device, calls
    connect() once, then dispatches evaluate() calls via ThreadPoolExecutor.
    """

    def __init__(
        self,
        device_cfg: dict,
        *,
        record_frames: bool = False,
        calibrate_touch_on_connect: bool = False,
    ) -> None:
        self.device_id: str = device_cfg["name"]
        self._cfg = device_cfg
        self.record_frames = record_frames
        self.calibrate_touch_on_connect = calibrate_touch_on_connect
        self.driver: webdriver.Remote | None = None
        self.mjpeg_url: str | None = None
        self._failure_streak: int = 0
        self._last_reconnect_time: float = 0.0
        self._dead_since: float | None = None
        # Optional visual "still in a skatepark?" guard. Disabled (no-op) unless
        # SCENE_GUARD_MODEL points at a trained checkpoint. See
        # experiments/scene_classifier_journal.md.
        self.scene_guard = SceneGuard.from_env()

    @property
    def alive(self) -> bool:
        return self._failure_streak < _DEAD_THRESHOLD

    @property
    def dead_since(self) -> float | None:
        """monotonic() timestamp at which this worker crossed into the dead state.

        None while the worker is alive. Read-only — mutated only via the
        record_success / record_failure lifecycle methods.
        """
        return self._dead_since

    @property
    def device_w(self) -> float:
        return float(self._cfg["logical_w"])

    @property
    def device_h(self) -> float:
        return float(self._cfg["logical_h"])

    @property
    def spin_button_xy(self) -> tuple[float, float]:
        # Normalised [0, 1] coords of True Skate's rotate button. Every DEVICES
        # entry sets this; the default mirrors them (was a stale logical-point
        # tuple that never matched a real device).
        value = self._cfg.get("spin_button_xy", (0.0604, 0.4040))
        return float(value[0]), float(value[1])

    # -- connection ---------------------------------------------------------

    def connect(self) -> None:
        """Create an Appium driver for this device's ports."""
        load_dotenv(_REPO_ROOT / ".env")
        udid = os.environ.get(self._cfg["env_key"])

        options = XCUITestOptions()
        options.platform_name = "iOS"
        options.automation_name = "XCUITest"
        options.bundle_id = _BUNDLE_ID
        if udid:
            options.udid = udid
        else:
            logging.warning(
                "[%s] %s not set in .env; connecting via live Appium/WDA ports.",
                self.device_id,
                self._cfg["env_key"],
            )
        options.wda_local_port = self._cfg["wda_port"]
        options.use_prebuilt_wda = True
        options.skip_log_capture = True
        options.no_reset = True
        options.set_capability("webDriverAgentUrl", f"http://127.0.0.1:{self._cfg['wda_port']}")
        appium_url = f"http://127.0.0.1:{self._cfg['appium_port']}"

        # Pre-flight: confirm WDA is reachable before telling Appium to connect.
        # If WDA just started it may need a moment, so we retry up to 3 times.
        wda_url = f"http://127.0.0.1:{self._cfg['wda_port']}/status"
        for attempt in range(1, 4):
            try:
                if requests.get(wda_url, timeout=2).status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            if attempt == 3:
                raise RuntimeError(
                    f"[{self.device_id}] WDA is not responding at "
                    f"http://127.0.0.1:{self._cfg['wda_port']} after 3 attempts. "
                    f"Run 'python scripts/launch_services.py' first."
                )
            time.sleep(2)

        self.driver = webdriver.Remote(appium_url, options=options)

        actual = self.driver.get_window_size()
        exp_w, exp_h = int(self._cfg["logical_w"]), int(self._cfg["logical_h"])
        if actual["width"] != exp_w or actual["height"] != exp_h:
            raise RuntimeError(
                f"[{self.device_id}] screen dimensions mismatch: "
                f"expected {exp_w}×{exp_h}, got {actual['width']}×{actual['height']}. "
                f"Check Display Zoom: Settings → Display & Brightness → Display Zoom → Default."
            )

        self.mjpeg_url = f"http://127.0.0.1:{self._cfg['mjpeg_port']}"

        state = self.driver.query_app_state(_BUNDLE_ID)
        if state == _APP_STATE_FOREGROUND:
            print(f"[{self.device_id}] True Skate already in foreground — reusing.")
        else:
            print(
                f"[{self.device_id}] True Skate not in foreground "
                f"(state={state}) — activating."
            )
            self.driver.activate_app(_BUNDLE_ID)
            time.sleep(1.0)
            try:
                skip_loading_screen(self.driver, self.device_w, self.device_h)
            except Exception as exc:
                logging.warning("[%s] skip loading screen failed: %s", self.device_id, exc)
            time.sleep(1.0)

        if self.calibrate_touch_on_connect:
            try:
                calibration = calibrate_touch_timing(
                    self.driver,
                    device_key=self.device_id,
                    device_w=self.device_w,
                    device_h=self.device_h,
                )
                logging.info(
                    "[%s] touch calibration source=%s seq_overhead=%.3fs threshold=%.3fs",
                    self.device_id,
                    calibration.source,
                    calibration.sequential_overhead_s,
                    calibration.combined_nonneg_threshold_s,
                )
                reset_position(self.driver, self._cfg["logical_w"], self._cfg["logical_h"])
            except Exception as exc:
                logging.warning("[%s] touch calibration skipped: %s", self.device_id, exc)

    # -- foreground check ---------------------------------------------------

    def ensure_foreground(self) -> bool:
        """Verify True Skate is in the foreground; relaunch if not.

        Returns True if the app had to be relaunched.
        """
        state = self.driver.query_app_state(_BUNDLE_ID)
        if state == _APP_STATE_FOREGROUND:
            return False
        print(
            f"[{self.device_id}] True Skate not in foreground "
            f"(state={state}) — relaunching."
        )
        self.driver.activate_app(_BUNDLE_ID)
        time.sleep(3.0)
        try:
            skip_loading_screen(self.driver, self.device_w, self.device_h)
        except Exception as exc:
            logging.warning("[%s] skip loading screen failed: %s", self.device_id, exc)
        time.sleep(1.0)
        return True

    # -- scene guard --------------------------------------------------------

    def check_scene(self) -> bool | None:
        """Verify we're still in an active skatepark; recover if not.

        Returns True/False when the (optional) scene guard has a verdict, else
        None (guard disabled or inference failed). On a False verdict it tries
        to recover the device back into the park so the next eval isn't wasted
        swiping at a menu/home screen.
        """
        if not self.scene_guard.enabled:
            return None
        try:
            png = self.driver.get_screenshot_as_png()
            verdict = self.scene_guard.in_skatepark(Image.open(io.BytesIO(png)))
        except Exception as exc:
            logging.warning("[%s] scene check failed: %s", self.device_id, exc)
            return None
        if verdict is False:
            logging.warning(
                "[%s] scene guard: not in skatepark — recovering.", self.device_id
            )
            self._recover_to_skatepark()
        return verdict

    def _recover_to_skatepark(self) -> None:
        """Bring the device back into a skating session: re-activate → skip → reset."""
        try:
            self.driver.activate_app(_BUNDLE_ID)
            time.sleep(1.0)
            skip_loading_screen(self.driver, self.device_w, self.device_h)
            time.sleep(0.5)
            reset_position(self.driver, self._cfg["logical_w"], self._cfg["logical_h"])
        except Exception as exc:
            logging.warning("[%s] scene recovery failed: %s", self.device_id, exc)

    # -- reset --------------------------------------------------------------

    def reset(self) -> None:
        """Reset the board to its starting position."""
        try:
            reset_position(self.driver, self._cfg["logical_w"], self._cfg["logical_h"])
        except Exception as exc:
            logging.warning("[%s] reset failed, attempting reconnect: %s", self.device_id, exc)
            if self._reconnect():
                reset_position(self.driver, self._cfg["logical_w"], self._cfg["logical_h"])
            else:
                logging.error("[%s] reset skipped — device unreachable.", self.device_id)

    def timed_reset(self) -> tuple[str, float, float]:
        """Reset the board and time it. Returns (device_id, monotonic_start, duration_s).

        Convenience for orchestrators that dispatch resets on a thread pool and
        need the start timestamp to compute post-eval wait time.
        """
        started_at = time.monotonic()
        self.reset()
        return self.device_id, started_at, time.monotonic() - started_at

    # -- reconnect ----------------------------------------------------------

    def _reconnect(self, max_attempts: int = 3) -> bool:
        """Quit the stale Appium session and re-establish a fresh one.

        Returns True if reconnect succeeded, False if all attempts failed.
        """
        self._last_reconnect_time = time.monotonic()
        for attempt in range(1, max_attempts + 1):
            try:
                if self.driver is not None:
                    try:
                        self.driver.quit()
                    except Exception:
                        pass
                    self.driver = None
                self.connect()
                self._failure_streak = 0
                self._dead_since = None
                print(f"[{self.device_id}] Reconnected (attempt {attempt}).")
                return True
            except Exception as exc:
                logging.warning(
                    "[%s] reconnect attempt %d/%d failed: %s",
                    self.device_id, attempt, max_attempts, exc,
                )
                time.sleep(5 * attempt)
        logging.error("[%s] all reconnect attempts failed.", self.device_id)
        return False

    def maybe_revive(self) -> None:
        """Attempt a reconnect if the worker is dead and the cooldown has elapsed.

        Public worker-lifecycle hook for orchestrators (CMA-ES loop, PPO
        collector) to call after a batch. No-op while the worker is alive.
        """
        if self.alive:
            return
        if time.monotonic() - self._last_reconnect_time < _REVIVE_COOLDOWN:
            return
        logging.info("[%s] attempting scheduled reconnect.", self.device_id)
        self._reconnect()

    # -- health tracking ----------------------------------------------------

    def record_success(self) -> None:
        """Mark the most recent eval as successful: clear the failure streak.

        The single entry point for both orchestrators to report success —
        replaces direct mutation of _failure_streak / _dead_since.
        """
        self._failure_streak = 0
        self._dead_since = None

    def record_failure(self) -> None:
        """Mark the most recent eval as failed: advance the failure streak.

        Stamps _dead_since on the transition from alive → dead. The single
        entry point for both orchestrators to report failure. Does not itself
        reconnect — callers decide whether to follow up with maybe_revive() or
        _reconnect().
        """
        was_alive = self.alive
        self._failure_streak += 1
        if was_alive and not self.alive:
            self._dead_since = time.monotonic()

    # -- evaluate -----------------------------------------------------------

    def evaluate(
        self,
        params: np.ndarray,
        wait_time: float,
        eval_num: int,
        generation: int,
        *,
        scorer: Callable[[TrickResult | None], float],
        run_dir: Path | None = None,
    ) -> dict:
        """Execute one candidate eval on this device.

        Runs: ensure_foreground -> record frames -> execute_gesture_params ->
        capture_and_detect -> score via injected scorer -> stop recording.
        Does NOT call reset_position (the orchestrator handles resets).

        Args:
            scorer: Callable mapping TrickResult | None → float. Typically
                ``Curriculum.score`` passed in from the CMA-ES optimizer.
            run_dir: Run folder; when set, OCR failures dump diagnostics to
                ``run_dir/ocr_failures/``.

        Returns a result dict; never raises — logs errors and returns
        reward=0.0 on failure.
        """
        relaunched = self.ensure_foreground()
        recorder = FrameRecorder() if self.record_frames else None
        trace_rec = TimestampedColorRecorder() if _TRACE_COLLECT else None
        eval_start_time = time.monotonic()
        try:
            if recorder is not None:
                recorder.start(self.mjpeg_url)
            if trace_rec is not None:
                trace_rec.start(self.mjpeg_url, resize_width=_TRACE_RESIZE_WIDTH)
            action_start_time = time.monotonic()
            execute_gesture_params(
                self.driver,
                np.array(params),
                device_w=self._cfg["logical_w"],
                device_h=self._cfg["logical_h"],
                spin_button_xy=self.spin_button_xy,
                timing_device_key=self.device_id,
            )
            action_end_time = time.monotonic()
            if wait_time > 0:
                time.sleep(wait_time)
            ocr_failure_dir = run_dir / "ocr_failures" if run_dir is not None else None
            eval_label = f"eval_{eval_num:05d}_{self.device_id}"
            trick_result, capture_diag = capture_and_detect_with_diagnostics(
                self.driver,
                capture_interval=0.15,
                action_start_time=action_end_time,
                max_window_s=3.5,
                ocr_failure_dir=ocr_failure_dir,
                eval_label=eval_label,
            )
            reward = scorer(trick_result)
            reward_end_time = time.monotonic()
        except Exception as exc:
            if recorder is not None:
                recorder.stop()
            if trace_rec is not None:
                trace_rec.stop()
            logging.warning("[%s] eval %d failed: %s", self.device_id, eval_num, exc)
            self.record_failure()
            if self._failure_streak % 5 == 0:
                self._reconnect()
            print(
                f"[eval {eval_num:05d} | gen {generation:04d}] "
                f"device={self.device_id} reward=0.00 status=error "
                f"trick=- frames=0 error={exc}"
            )
            return {
                "reward": 0.0,
                "trick_name": None,
                "trick_status": None,
                "device_id": self.device_id,
                "params": params,
                "raw_frames": [],
                "n_composites": 0,
                "app_relaunched": relaunched,
                "in_skatepark": None,
                "action_exec_s": 0.0,
                "reward_eval_s": 0.0,
                "eval_total_s": 0.0,
                "capture_attempts": 0,
                "skipped_captures": 0,
                "detection_capture_idx": None,
                "capture_elapsed_s": 0.0,
            }

        self.record_success()
        in_skatepark = self.check_scene()  # None unless the scene guard is enabled
        raw_frames = recorder.stop() if recorder is not None else []
        # Color trace frames + their times relative to gesture start (the executed
        # gesture vector in "params" is the label; offline self_label aligns them).
        trace_frames, trace_times = trace_rec.stop() if trace_rec is not None else ([], [])
        trace_frame_times = [round(t - action_start_time, 4) for t in trace_times]
        trick_name = trick_result.trick if trick_result else None
        trick_status = trick_result.status if trick_result else None
        trick_label = trick_name if trick_name else "-"
        status_label = trick_status if trick_status else "-"
        relaunched_label = "yes" if relaunched else "no"

        print(
            f"[eval {eval_num:05d} | gen {generation:04d}] "
            f"device={self.device_id} reward={reward:.2f} "
            f"status={status_label} trick={trick_label} "
            f"frames={len(raw_frames)} relaunched={relaunched_label}"
        )

        return {
            "reward": reward,
            "trick_name": trick_name,
            "trick_status": trick_status,
            "device_id": self.device_id,
            "params": params,
            "raw_frames": raw_frames,
            "trace_frames": trace_frames,
            "trace_frame_times": trace_frame_times,
            "n_composites": 0,
            "app_relaunched": relaunched,
            "in_skatepark": in_skatepark,
            "action_exec_s": action_end_time - action_start_time,
            "reward_eval_s": reward_end_time - action_end_time,
            "eval_total_s": reward_end_time - eval_start_time,
            "capture_attempts": int(capture_diag["captures_attempted"]),
            "skipped_captures": int(capture_diag["skipped_captures"]),
            "detection_capture_idx": (
                None
                if capture_diag["detection_capture_idx"] is None
                else int(capture_diag["detection_capture_idx"])
            ),
            "capture_elapsed_s": float(capture_diag["capture_elapsed_s"]),
            "anchor_candidates": int(capture_diag["anchor_candidates"]),
            "ocr_calls": int(capture_diag["ocr_calls"]),
        }

    # -- disconnect ---------------------------------------------------------

    def disconnect(self) -> None:
        """Quit the Appium driver."""
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception as exc:
                logging.warning("[%s] disconnect error: %s", self.device_id, exc)
            finally:
                self.driver = None
                self.mjpeg_url = None
            print(f"[{self.device_id}] Disconnected.")
