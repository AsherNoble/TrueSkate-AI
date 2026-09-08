"""Single source of truth for frame preprocessing across the BC pipeline.

Model 1 (trace extractor) and Model 2 (sequence policy) both consume frames as
resize -> BGR2RGB -> normalise -> CHW. Train and inference MUST transform
frames identically or the model sees a distribution shift it was never trained
on; this used to be reimplemented independently at each call site.
"""
from __future__ import annotations

import numpy as np


def prep_frame_rgb(frame_bgr: np.ndarray, h: int, w: int, *, normalize: bool = True) -> np.ndarray:
    """BGR uint8 HxWx3 (cv2 convention) -> resized RGB HxWxC array.

    normalize=True (default) returns float32 in [0, 1] — what every model
    forward pass expects. normalize=False returns uint8 [0, 255], for callers
    that cache many frames in memory and normalise lazily per-item.
    """
    import cv2

    img = cv2.resize(np.asarray(frame_bgr), (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / 255.0) if normalize else img.astype(np.uint8)
