import difflib
import logging
import re
from functools import lru_cache
from typing import Literal, NamedTuple

import cv2
import numpy as np

from .known_tricks import KNOWN_TRICKS, MODIFIERS
from trueskate_ai.vision.vision_ocr import image_to_lines as vision_image_to_lines
from trueskate_ai.vision.vision_ocr import is_vision_available, vision_unavailable_reason


class TrickResult(NamedTuple):
    trick: str
    status: Literal["landed", "failed", "unknown"]


class AnchorInfo(NamedTuple):
    """Cheap (OCR-free) description of a trick-notification anchor in a frame.

    The anchor is the green (landed) / red (failed) score-digit cluster. `mask`
    is the boolean colour mask over ``frame[anchor_y_offset:680, :]`` and is
    consumed by `ocr_at_anchor`. `clipped` is True when the notification is
    mid-slide (anchor bbox hugging a frame edge) — its trick name is truncated
    and OCR on it would yield garbage fragments.
    """

    status: Literal["landed", "failed"]
    mask: np.ndarray
    anchor_y_offset: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    pixel_count: int
    clipped: bool


# Row where the anchor colour-search band starts (frames are 750x1624).
_ANCHOR_Y_OFFSET = 180
# A notification slides in from the left and out to the right; when its anchor
# bbox is within these margins of a frame edge the trick name is clipped
# off-screen. Asymmetric: the name extends far left of the score but only
# slightly right, and the right-edge slide-out is the dominant failure mode.
_CLIP_MARGIN_LEFT = 40
_CLIP_MARGIN_RIGHT = 150

# Sponsor/park-name words that show up in-frame and must never match a trick.
_BANNER_WORDS = frozenset({
    "TRUE", "SKATE", "SUPER", "CROWN", "STREET", "LEAGUE", "SLS",
    "CALIFORNIA", "SKATEPARKS", "SANTA", "CRUZ", "GLASSHOUSE",
})

_LOGGED_OCR_BACKEND = False


@lru_cache(maxsize=1)
def _resolve_ocr_backend() -> Literal["vision"]:
    # Apple Vision is the only OCR backend (pytesseract removed), so there is no
    # backend to select — this stays a function only to fail fast when Vision's
    # pyobjc deps are missing.
    if is_vision_available():
        return "vision"

    raise RuntimeError(
        "Apple Vision OCR is unavailable. "
        "Install pyobjc-core, pyobjc-framework-Cocoa, pyobjc-framework-Quartz, "
        f"and pyobjc-framework-Vision. Cause: {vision_unavailable_reason()}"
    )


def _extract_lines_with_ocr(crop: np.ndarray) -> list[str]:
    global _LOGGED_OCR_BACKEND
    backend = _resolve_ocr_backend()
    if not _LOGGED_OCR_BACKEND:
        logging.info("trick_info_reader: OCR backend=%s", backend)
        _LOGGED_OCR_BACKEND = True
    return vision_image_to_lines(crop)


def ensure_ocr_backend_ready() -> None:
    """Fail fast if OCR backend is misconfigured or unavailable."""
    _resolve_ocr_backend()


def _match_component(ocr_line: str) -> str | None:
    """Fuzzy match a single OCR line against KNOWN_TRICKS with modifier handling."""
    words = ocr_line.split()
    if not words:
        return None

    # Exact match before modifier stripping (handles "BACKSIDE 360" etc.)
    if ocr_line in KNOWN_TRICKS:
        return ocr_line

    modifier = None
    mod_match = difflib.get_close_matches(words[0], list(MODIFIERS), n=1, cutoff=0.5)
    if mod_match:
        modifier = mod_match[0]
        ocr_line = " ".join(words[1:])

    # Guard: bare rotation numbers ("360", "540", etc.) must not be fuzzy-
    # matched — they'd inflate to "360 FLIP" etc.  Reconstruct with modifier
    # and return directly if the combo is in KNOWN_TRICKS.
    if re.fullmatch(r"\d+", ocr_line):
        candidate = f"{modifier} {ocr_line}" if modifier else ocr_line
        if candidate in KNOWN_TRICKS:
            return candidate
        logging.warning("trick_info_reader: bare rotation %r not in KNOWN_TRICKS", candidate)
        return None

    matches = difflib.get_close_matches(ocr_line, KNOWN_TRICKS, n=1, cutoff=0.4)
    if matches:
        if modifier and not matches[0].startswith(modifier):
            return f"{modifier} {matches[0]}"
        return matches[0]

    logging.warning("trick_info_reader: no match for OCR output %r", ocr_line)
    return None


def _find_anchor(search: np.ndarray) -> tuple[np.ndarray, Literal["landed", "failed"]] | None:
    """Find the notification anchor band (green = landed, red = failed).

    Args:
        search: Cropped frame slice (frame[180:680, :]) in BGR.

    Returns:
        (mask, status) where mask is a bool array over search and status is
        "landed" or "failed", or None if no colour is found.
    """
    r = search[:, :, 2].astype(np.int32)
    g = search[:, :, 1].astype(np.int32)
    b = search[:, :, 0].astype(np.int32)

    green_mask = (g > 150) & ((g - r) > 40) & ((g - b) > 40)
    if green_mask.sum() >= 15:
        logging.debug("anchor search: green=%d, red=— (skipped)", green_mask.sum())
        return green_mask, "landed"

    # Filter red pixels to the left+center two-thirds of the frame to exclude
    # right-edge sponsor bars and stadium walls; notification banners appear on
    # the left or center of the screen.
    w = search.shape[1]
    left_center = np.zeros_like(r, dtype=bool)
    left_center[:, : 2 * w // 3] = True

    red_mask = (r > 150) & ((r - g) > 60) & ((r - b) > 60)
    red_filtered = red_mask & left_center

    logging.debug("anchor search: green=%d, red=%d (filtered)", green_mask.sum(), red_filtered.sum())

    if red_filtered.sum() >= 35:
        return red_filtered, "failed"

    return None


def _ocr_above_anchor(
    frame: np.ndarray,
    mask: np.ndarray,
    anchor_y_offset: int,
    status: Literal["landed", "failed"],
) -> tuple[TrickResult | None, dict]:
    """Crop above the anchor band, run OCR, and return a TrickResult.

    Args:
        frame: Full BGR frame.
        mask: Boolean mask over frame[anchor_y_offset:anchor_y_offset+mask.shape[0], :].
        anchor_y_offset: Row in frame where the search band starts (180).
        status: "landed" or "failed" — determines TrickResult.status.

    Returns:
        (TrickResult or None, diagnostics dict)
    """
    ys, xs = np.where(mask)
    anchor_row = int(ys.min())
    anchor_y_min = anchor_row + anchor_y_offset
    anchor_x_min = int(xs.min())
    anchor_x_max = int(xs.max())

    h, w = frame.shape[:2]
    y0 = max(0, anchor_y_min - 160)
    y1 = anchor_y_min
    x0 = max(0, anchor_x_min - 220)
    x1 = min(w, anchor_x_max + 220)

    band = frame[y0:y1, x0:x1]
    upscaled = cv2.resize(band, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    candidates = []
    lines = _extract_lines_with_ocr(upscaled) + _extract_lines_with_ocr(binary_inv)
    seen_lines: set[str] = set()
    for line in lines:
        if line in seen_lines:
            continue
        seen_lines.add(line)
        cleaned = re.sub(r"[^A-Z0-9 :-]", "", line.upper()).strip()
        # OCR can merge letter-digit and digit-letter boundaries — split them.
        cleaned = re.sub(r'([A-Z])(\d)', r'\1 \2', cleaned)
        cleaned = re.sub(r'(\d)([A-Z])', r'\1 \2', cleaned)
        # Normalize OCR rotation number misreads:
        #   "560" is a common misread of "360" (3 ↔ 5 confusion at small size).
        cleaned = re.sub(r'\b560\b', '360', cleaned)
        #   Any x40 number (140, 240, etc.) is a misread of "540" — the only
        #   valid x40 rotation in True Skate. "540" itself maps to itself.
        cleaned = re.sub(r'\b\d40\b', '540', cleaned)
        if not cleaned:
            continue
        if "SCORE" in cleaned:
            continue
        # Discard pure score values: any bare number > 1080 is a game score counter,
        # not a rotation (max valid rotation in True Skate is 1080).
        if re.fullmatch(r'\d+', cleaned) and int(cleaned) > 1080:
            continue
        if _BANNER_WORDS & set(cleaned.split()):
            continue
        # For failed detections the word "FAILED" appears in the crop — discard it.
        if difflib.get_close_matches(cleaned, ["FAILED"], n=1, cutoff=0.7):
            continue
        candidates.append(cleaned)

    diagnostics = {
        "crop_bounds": [x0, y0, x1, y1],
        "ocr_lines": lines,
        "ocr_candidates": candidates,
    }

    if not candidates:
        diagnostics["matched_components"] = []
        diagnostics["final_trick"] = None
        return None, diagnostics

    # Try merging adjacent candidates before individual matching.
    # e.g. ["360", "POP SHOVE-IT"] → try "360 POP SHOVE-IT" against KNOWN_TRICKS first.
    # If the merge matches, consume both and skip individual matching for that pair.
    matched_components = []
    i = 0
    while i < len(candidates):
        if i + 1 < len(candidates):
            merged = candidates[i] + " " + candidates[i + 1]
            if merged in KNOWN_TRICKS:
                matched_components.append(merged)
                i += 2
                continue
        match = _match_component(candidates[i])
        if match is not None:
            matched_components.append(match)
        i += 1

    if not matched_components:
        diagnostics["matched_components"] = []
        diagnostics["final_trick"] = None
        return None, diagnostics

    trick = " + ".join(matched_components)
    diagnostics["matched_components"] = matched_components
    diagnostics["final_trick"] = trick
    logging.info("trick_info_reader: %s — %s", status, trick)
    return TrickResult(trick=trick, status=status), diagnostics


def find_notification_anchor(frame: np.ndarray) -> AnchorInfo | None:
    """Cheap, OCR-free detection of the trick-notification anchor.

    Runs only colour thresholding (fast numpy) — no OCR. Used to score and rank
    many captured frames before paying for OCR on just the best few.

    Returns AnchorInfo, or None if no green/red notification colour is present.
    """
    search = frame[_ANCHOR_Y_OFFSET:680, :]
    result = _find_anchor(search)
    if result is None:
        return None

    mask, status = result
    ys, xs = np.where(mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min = int(ys.min()) + _ANCHOR_Y_OFFSET
    y_max = int(ys.max()) + _ANCHOR_Y_OFFSET
    w = frame.shape[1]
    clipped = x_min <= _CLIP_MARGIN_LEFT or x_max >= w - _CLIP_MARGIN_RIGHT
    return AnchorInfo(
        status=status,
        mask=mask,
        anchor_y_offset=_ANCHOR_Y_OFFSET,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        pixel_count=int(mask.sum()),
        clipped=clipped,
    )


def ocr_at_anchor(frame: np.ndarray, anchor: AnchorInfo) -> tuple[TrickResult | None, dict]:
    """Run the (expensive) OCR pipeline for a precomputed anchor."""
    return _ocr_above_anchor(frame, anchor.mask, anchor.anchor_y_offset, anchor.status)


def detect_trick_with_diagnostics(frame: np.ndarray) -> tuple[TrickResult | None, dict]:
    """Return trick detection result plus per-frame diagnostics."""
    diagnostics: dict = {
        "anchor_found": False,
        "anchor_status": None,
        "anchor_pixels": 0,
        "anchor_clipped": False,
        "crop_bounds": None,
        "ocr_lines": [],
        "ocr_candidates": [],
        "matched_components": [],
        "final_trick": None,
    }

    anchor = find_notification_anchor(frame)
    if anchor is None:
        return None, diagnostics

    diagnostics["anchor_found"] = True
    diagnostics["anchor_status"] = anchor.status
    diagnostics["anchor_pixels"] = anchor.pixel_count
    diagnostics["anchor_clipped"] = anchor.clipped
    trick_result, ocr_diag = ocr_at_anchor(frame, anchor)
    diagnostics.update(ocr_diag)
    return trick_result, diagnostics


def detect_trick(frame: np.ndarray) -> TrickResult | None:
    """Detect trick name (or combo) from a 750x1624 game frame.

    Finds green pixels (landed) or red pixels (failed) in a wide search
    band to anchor the score notification, then crops tightly above the
    anchor to isolate the trick name. Multiple lines are treated as a
    combo and joined with " + ".

    Returns e.g. TrickResult(trick="KICKFLIP + CROOKED GRIND", status="landed")
    or None if no notification is visible.
    """
    result, _ = detect_trick_with_diagnostics(frame)
    return result
