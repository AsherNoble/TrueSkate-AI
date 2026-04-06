import difflib
import logging
import re
from typing import Literal, NamedTuple

import cv2
import numpy as np
import pytesseract

from .known_tricks import KNOWN_TRICKS


class TrickResult(NamedTuple):
    trick: str
    status: Literal["landed", "failed"]


def _match_component(ocr_line: str) -> str | None:
    """Fuzzy match a single OCR line against KNOWN_TRICKS with modifier handling."""
    words = ocr_line.split()
    if not words:
        return None

    modifier = None
    mod_match = difflib.get_close_matches(words[0], ["FAKIE", "SWITCH", "BACKSIDE", "FRONTSIDE"], n=1, cutoff=0.5)
    if mod_match:
        modifier = mod_match[0]
        ocr_line = " ".join(words[1:])

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
        search: Cropped frame slice (frame[250:600, :]) in BGR.

    Returns:
        (mask, status) where mask is a bool array over search and status is
        "landed" or "failed", or None if neither colour is found.
    """
    r = search[:, :, 2].astype(np.int32)
    g = search[:, :, 1].astype(np.int32)
    b = search[:, :, 0].astype(np.int32)

    green_mask = (g > 180) & (r < 120) & (b < 120)
    if green_mask.sum() >= 20:
        logging.debug("anchor search: green=%d, red=— (skipped)", green_mask.sum())
        return green_mask, "landed"

    # Filter red pixels to the center third of the frame horizontally.
    # The FAILED notification is centered; stadium walls and sponsor bars
    # appear at the edges, so this eliminates most false positives.
    w = search.shape[1]
    center_col = np.zeros_like(r, dtype=bool)
    center_col[:, w // 3 : 2 * w // 3] = True

    red_mask = (r > 180) & (g < 80) & (b < 80)
    red_filtered = red_mask & center_col

    logging.debug("anchor search: green=%d, red=%d (filtered)", green_mask.sum(), red_filtered.sum())

    if red_filtered.sum() >= 50:
        return red_filtered, "failed"

    return None


def _ocr_above_anchor(
    frame: np.ndarray,
    mask: np.ndarray,
    anchor_y_offset: int,
    status: Literal["landed", "failed"],
) -> TrickResult | None:
    """Crop above the anchor band, run OCR, and return a TrickResult.

    Args:
        frame: Full BGR frame.
        mask: Boolean mask over frame[anchor_y_offset:anchor_y_offset+mask.shape[0], :].
        anchor_y_offset: Row in frame where the search band starts (250).
        status: "landed" or "failed" — determines TrickResult.status.

    Returns:
        TrickResult or None if no trick text is found.
    """
    ys, xs = np.where(mask)
    anchor_y_min = int(ys.min()) + anchor_y_offset
    anchor_x_min = int(xs.min())
    anchor_x_max = int(xs.max())

    h, w = frame.shape[:2]
    y0 = max(0, anchor_y_min - 100)
    y1 = anchor_y_min
    x0 = max(0, anchor_x_min - 150)
    x1 = min(w, anchor_x_max + 150)

    band = frame[y0:y1, x0:x1]
    upscaled = cv2.resize(band, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    _, crop = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    cv2.imwrite("/tmp/debug_crop.png", crop)

    config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :-"
    raw = pytesseract.image_to_string(crop, config=config)

    _BANNER_WORDS = {"TRUE", "SKATE", "SUPER", "CROWN", "STREET", "LEAGUE", "SLS", "CALIFORNIA", "SKATEPARKS", "SANTA", "CRUZ", "GLASSHOUSE"}

    candidates = []
    for line in raw.splitlines():
        cleaned = re.sub(r"[^A-Z0-9 :-]", "", line.upper()).strip()
        # Tesseract merges letter-digit and digit-letter boundaries — split them.
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
        if re.fullmatch(r"[\d ]+", cleaned):
            continue
        if _BANNER_WORDS & set(cleaned.split()):
            continue
        # For failed detections the word "FAILED" appears in the crop — discard it.
        if difflib.get_close_matches(cleaned, ["FAILED"], n=1, cutoff=0.7):
            continue
        candidates.append(cleaned)

    if not candidates:
        return None

    matched_components = [_match_component(c) for c in candidates]
    matched_components = [m for m in matched_components if m is not None]

    if not matched_components:
        return None

    trick = " + ".join(matched_components)
    logging.info("trick_info_reader: %s — %s", status, trick)
    return TrickResult(trick=trick, status=status)


def detect_trick(frame: np.ndarray) -> TrickResult | None:
    """Detect trick name (or combo) from a 750x1624 game frame.

    Finds green pixels (landed) or red pixels (failed) in a wide search
    band to anchor the score notification, then crops tightly above the
    anchor to isolate the trick name. Multiple lines are treated as a
    combo and joined with " + ".

    Returns e.g. TrickResult(trick="KICKFLIP + CROOKED GRIND", status="landed")
    or None if no notification is visible.
    """
    _ANCHOR_Y_OFFSET = 250
    search = frame[_ANCHOR_Y_OFFSET:600, :]

    result = _find_anchor(search)
    if result is None:
        return None

    mask, status = result
    return _ocr_above_anchor(frame, mask, _ANCHOR_Y_OFFSET, status)
