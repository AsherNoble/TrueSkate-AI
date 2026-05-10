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

from trueskate_ai.rl.cmaes.action_param import execute_gesture_params
from trueskate_ai.rl.reward import (
    ContinuousTrickMonitor,
    compute_reward,
    get_reward,
    merge_trick_results,
)
from trueskate_ai.sim.touch_actions import calibrate_touch_timing, reset_position, skip_loading_screen

# ---------------------------------------------------------------------------
# Device configurations
# ---------------------------------------------------------------------------

DEVICES: list[dict] = [
    {
        "env_key": "IPHONE_XR_UDID",
        "name": "iPhone_XR",
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
        "wda_port": 8101,
        "mjpeg_port": 9101,
        "appium_port": 4724,
        "logical_w": 414,
        "logical_h": 896,
        "spin_button_xy": (0.0604, 0.4040),
    },
    {
        "env_key": "IPHONE_XS_UDID",
        "name": "iPhone_XS",
        "wda_port": 8102,
        "mjpeg_port": 9102,
        "appium_port": 4725,
        "logical_w": 375,
        "logical_h": 812,
        "spin_button_xy": (0.0604, 0.4040),
    },
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUNDLE_ID = "com.trueaxis.skate"
_APP_STATE_FOREGROUND = 4
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEAD_THRESHOLD = 5      # consecutive failures before a worker is considered dead
_REVIVE_COOLDOWN = 60.0  # seconds between scheduled reconnect attempts for dead workers
_ALL_DEAD_TIMEOUT = 300.0  # seconds before aborting when every worker is dead

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
        calibrate_touch_on_connect: bool = True,
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

    @property
    def alive(self) -> bool:
        return self._failure_streak < _DEAD_THRESHOLD

    @property
    def device_w(self) -> float:
        return float(self._cfg["logical_w"])

    @property
    def device_h(self) -> float:
        return float(self._cfg["logical_h"])

    @property
    def spin_button_xy(self) -> tuple[float, float]:
        value = self._cfg.get("spin_button_xy", (25.0, 362.0))
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

    def _try_revive(self) -> None:
        """Attempt a reconnect if the cooldown has elapsed. Called by the collector."""
        if self.alive:
            return
        if time.monotonic() - self._last_reconnect_time < _REVIVE_COOLDOWN:
            return
        logging.info("[%s] attempting scheduled reconnect.", self.device_id)
        self._reconnect()

    # -- evaluate -----------------------------------------------------------

    def evaluate(
        self,
        params: np.ndarray,
        wait_time: float,
        eval_num: int,
        generation: int,
    ) -> dict:
        """Execute one candidate eval on this device.

        Runs: ensure_foreground -> record frames -> execute_gesture_params ->
        get_reward -> stop recording. Does NOT call reset_position
        (the orchestrator handles resets across all devices).

        Returns a result dict; never raises — logs errors and returns
        reward=0.0 on failure.
        """
        relaunched = self.ensure_foreground()
        recorder = FrameRecorder() if self.record_frames else None
        eval_start_time = time.monotonic()
        try:
            monitor = ContinuousTrickMonitor()

            def _on_post_push() -> None:
                monitor.start(self.mjpeg_url)

            if recorder is not None:
                recorder.start(self.mjpeg_url)
            action_start_time = time.monotonic()
            execute_gesture_params(
                self.driver,
                np.array(params),
                device_w=self._cfg["logical_w"],
                device_h=self._cfg["logical_h"],
                on_post_push=_on_post_push,
                timing_device_key=self.device_id,
            )
            action_end_time = time.monotonic()
            monitor_result, monitor_diag = monitor.stop()
            reward, trick_result, _, capture_diag = get_reward(
                self.driver,
                wait_time=wait_time,
                penalty=None,
                action_start_time=action_end_time,
                return_diagnostics=True,
            )
            combined_result = merge_trick_results(monitor_result, trick_result)
            reward = compute_reward(combined_result)
            trick_result = combined_result
            reward_end_time = time.monotonic()
        except Exception as exc:
            if recorder is not None:
                recorder.stop()
            logging.warning("[%s] eval %d failed: %s", self.device_id, eval_num, exc)
            was_alive = self.alive
            self._failure_streak += 1
            if was_alive and not self.alive:
                self._dead_since = time.monotonic()
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
                "action_exec_s": 0.0,
                "reward_eval_s": 0.0,
                "eval_total_s": 0.0,
                "capture_attempts": 0,
                "skipped_captures": 0,
                "detection_capture_idx": None,
                "capture_elapsed_s": 0.0,
                "monitor_frames_checked": 0,
                "monitor_elapsed_s": 0.0,
            }

        self._failure_streak = 0
        raw_frames = recorder.stop() if recorder is not None else []
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
            "n_composites": 0,
            "app_relaunched": relaunched,
            "action_exec_s": action_end_time - action_start_time,
            "reward_eval_s": reward_end_time - action_end_time,
            "eval_total_s": reward_end_time - eval_start_time,
            "capture_attempts": int(capture_diag["captures_attempted"]) + int(monitor_diag["frames_checked"]),
            "skipped_captures": int(capture_diag["skipped_captures"]),
            "detection_capture_idx": (
                None
                if capture_diag["detection_capture_idx"] is None
                else int(capture_diag["detection_capture_idx"])
            ),
            "capture_elapsed_s": float(capture_diag["capture_elapsed_s"]),
            "monitor_frames_checked": int(monitor_diag["frames_checked"]),
            "monitor_elapsed_s": float(monitor_diag["monitor_elapsed_s"]),
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
