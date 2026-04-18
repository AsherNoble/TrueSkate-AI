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

from trueskate_ai.rl.action_param import execute_action
from trueskate_ai.rl.reward import get_reward
from trueskate_ai.sim.touch_actions import reset_position

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
    },
    {
        "env_key": "IPHONE_11_UDID",
        "name": "iPhone_11",
        "wda_port": 8101,
        "mjpeg_port": 9101,
        "appium_port": 4724,
    },
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUNDLE_ID = "com.trueaxis.skate"
_APP_STATE_FOREGROUND = 4
_REPO_ROOT = Path(__file__).resolve().parents[3]

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

    def __init__(self, device_cfg: dict) -> None:
        self.device_id: str = device_cfg["name"]
        self._cfg = device_cfg
        self.driver: webdriver.Remote | None = None
        self.mjpeg_url: str | None = None

    # -- connection ---------------------------------------------------------

    def connect(self) -> None:
        """Create an Appium driver for this device's ports/UDID."""
        load_dotenv(_REPO_ROOT / ".env")
        udid = os.environ.get(self._cfg["env_key"])
        if not udid:
            raise RuntimeError(
                f"[{self.device_id}] {self._cfg['env_key']} not set in .env"
            )

        options = XCUITestOptions()
        options.platform_name = "iOS"
        options.automation_name = "XCUITest"
        options.bundle_id = _BUNDLE_ID
        options.udid = udid
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
            time.sleep(1.5)

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
        time.sleep(2.0)
        return True

    # -- reset --------------------------------------------------------------

    def reset(self) -> None:
        """Reset the board to its starting position."""
        reset_position(self.driver)

    # -- evaluate -----------------------------------------------------------

    def evaluate(
        self,
        params: np.ndarray,
        wait_time: float,
        eval_num: int,
        generation: int,
    ) -> dict:
        """Execute one candidate eval on this device.

        Runs: ensure_foreground -> record frames -> execute_action ->
        get_reward -> stop recording. Does NOT call reset_position
        (the orchestrator handles resets across all devices).

        Returns a result dict; never raises — logs errors and returns
        reward=0.0 on failure.
        """
        relaunched = self.ensure_foreground()
        recorder = FrameRecorder()
        try:
            recorder.start(self.mjpeg_url)
            execute_action(self.driver, np.array(params))
            reward, trick_result, _ = get_reward(
                self.driver, wait_time=wait_time, penalty=None
            )
        except Exception as exc:
            recorder.stop()
            logging.warning("[%s] eval %d failed: %s", self.device_id, eval_num, exc)
            print(
                f"[{self.device_id}] [eval {eval_num} | gen {generation}] "
                f"ERROR: {exc} — assigning reward=0.0"
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
            }

        raw_frames = recorder.stop()
        trick_name = trick_result.trick if trick_result else None
        trick_status = trick_result.status if trick_result else None

        print(
            f"[{self.device_id}] [eval {eval_num} | gen {generation}] "
            f"reward={reward:.2f}  trick={trick_name}  status={trick_status}  "
            f"raw_frames={len(raw_frames)}"
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
