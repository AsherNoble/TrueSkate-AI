"""Shared compact clip decoding and deterministic grouped splits."""
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_IMAGE_HEIGHT = 288
DEFAULT_IMAGE_WIDTH = 128

def _frame_paths(sample: Path) -> tuple[Path, ...]:
    return tuple(sorted(sample.glob("frame_*.png")))

def _has_frames(sample: Path) -> bool:
    return bool(_frame_paths(sample)) or (sample / "frames.mp4").is_file()

def _decode_frames(sample: Path) -> list[np.ndarray]:
    paths = _frame_paths(sample)
    if paths:
        decoded = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    else:
        capture = cv2.VideoCapture(str(sample / "frames.mp4"))
        decoded = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded.append(frame)
        finally:
            capture.release()
    if not decoded or any(frame is None for frame in decoded):
        raise ValueError(f"{sample}: unreadable frames")
    return decoded

def _decode_even_frames(sample: Path, count: int) -> list[np.ndarray]:
    """Decode only evenly selected frames when the source is a compact video.

    Full video decode is correct but wasteful for clip regressors that always
    resample a fixed-length sequence.  Seeking to selected frame numbers keeps
    the same even temporal coverage while avoiding decode of ~30 unused frames
    per 32-frame aligned clip (and many more for legacy high-fps clips).
    """
    if count < 1:
        raise ValueError("count must be positive")
    paths = _frame_paths(sample)
    if paths:
        indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
        decoded = [cv2.imread(str(paths[int(index)]), cv2.IMREAD_COLOR) for index in indices]
    else:
        capture = cv2.VideoCapture(str(sample / "frames.mp4"))
        try:
            total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            indices = np.linspace(0, total - 1, count).round().astype(int)
            decoded = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                decoded.append(frame if ok else None)
        finally:
            capture.release()
        # Some H.264 builds report a frame count but cannot reliably random-seek
        # every index.  The clip remains valid, so fall back to sequential decode
        # rather than rejecting a strict sample for decoder behavior alone.
        if any(frame is None for frame in decoded):
            decoded = _decode_frames(sample)
            indices = np.linspace(0, len(decoded) - 1, count).round().astype(int)
            decoded = [decoded[int(index)] for index in indices]
    if not decoded or any(frame is None for frame in decoded):
        raise ValueError(f"{sample}: unreadable selected frames")
    return decoded

def _split_by_key(keys: tuple[str, ...], *, val_fraction: float, test_fraction: float,
                  seed: int) -> tuple[list[int], list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0 or not 0.0 < test_fraction < 1.0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("validation/test fractions must be positive and sum to less than one")
    groups = sorted(set(keys))
    if len(groups) < 3:
        raise ValueError("need at least three independent groups for train/validation/test")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(groups))
    n_test = max(1, round(len(groups) * test_fraction))
    n_val = max(1, round(len(groups) * val_fraction))
    if n_test + n_val >= len(groups):
        n_val = 1
        n_test = 1
    test_groups = set(shuffled[:n_test])
    val_groups = set(shuffled[n_test:n_test + n_val])
    train = [i for i, key in enumerate(keys) if key not in test_groups | val_groups]
    val = [i for i, key in enumerate(keys) if key in val_groups]
    test = [i for i, key in enumerate(keys) if key in test_groups]
    return train, val, test
