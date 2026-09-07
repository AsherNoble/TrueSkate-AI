"""Timestamped COLOR MJPEG recorder for self-labeled trace data.

The RL FrameRecorder (device_worker) decodes grayscale 210x455 for OCR; trace
extraction needs COLOR (the orange finger-trace is a hue). This recorder keeps
full-res RGB frames + per-frame monotonic capture timestamps so a frame can be
aligned to the known gesture trajectory offline (see vision/self_label).

Used by the standalone collector AND, flag-gated, by the CMA-ES eval loop so
trick runs passively generate (frame -> known-touch) pairs.
"""
from __future__ import annotations

import io
import threading
import time

import numpy as np
import requests
from PIL import Image


class TimestampedColorRecorder:
    """Reads an MJPEG stream into COLOR frames + monotonic capture timestamps."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = False
        self._frames: list[np.ndarray] = []
        self._times: list[float] = []
        self._resp: requests.Response | None = None
        self._resize_width: int | None = None

    def start(self, mjpeg_url: str, resize_width: int | None = None) -> None:
        """Begin capture. resize_width downscales frames on decode (keeps aspect)
        to bound memory/disk — used by the CMA-ES eval loop where ~100 full-res
        frames × parallel workers would OOM."""
        self._stop = False
        self._frames, self._times = [], []
        self._resize_width = resize_width
        self._thread = threading.Thread(target=self._loop, args=(mjpeg_url,), daemon=True)
        self._thread.start()

    def _loop(self, mjpeg_url: str) -> None:
        buf = b""
        try:
            resp = requests.get(mjpeg_url, stream=True, timeout=5)
            self._resp = resp
            for chunk in resp.iter_content(chunk_size=8192):
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
                    jpeg = buf[s:e + 2]
                    buf = buf[e + 2:]
                    try:
                        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                        if self._resize_width and img.width > self._resize_width:
                            h = round(img.height * self._resize_width / img.width)
                            img = img.resize((self._resize_width, h), Image.BILINEAR)
                        self._times.append(time.monotonic())
                        self._frames.append(np.array(img, dtype=np.uint8))
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            self._resp = None

    def stop(self) -> tuple[list[np.ndarray], list[float]]:
        self._stop = True
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self._frames, self._times
