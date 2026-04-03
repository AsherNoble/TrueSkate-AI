import difflib
import logging
import re

import cv2
import numpy as np
import pytesseract

from .known_tricks import KNOWN_TRICKS


def _match_component(ocr_line: str) -> str | None:
    """Fuzzy match a single OCR line against KNOWN_TRICKS with modifier handling."""
    words = ocr_line.split()
    if not words:
        return None

    modifier = None
    mod_match = difflib.get_close_matches(words[0], ["FAKIE", "SWITCH"], n=1, cutoff=0.5)
    if mod_match:
        modifier = mod_match[0]
        ocr_line = " ".join(words[1:])

    matches = difflib.get_close_matches(ocr_line, KNOWN_TRICKS, n=1, cutoff=0.4)
    if matches:
        return f"{modifier} {matches[0]}" if modifier else matches[0]

    logging.warning("trick_info_reader: no match for OCR output %r", ocr_line)
    return None


def detect_trick(frame: np.ndarray) -> str | None:
    """Detect trick name (or combo) from a 750x1624 game frame.

    Finds green pixels in a wide search band to anchor the notification,
    then crops tightly above the green band to isolate the trick name.
    Multiple lines are treated as a combo and joined with " + ".

    Returns e.g. "KICKFLIP + CROOKED GRIND" or None.
    """
    search = frame[250:600, :]
    g = search[:, :, 1].astype(np.int32)
    r = search[:, :, 0].astype(np.int32)
    b = search[:, :, 2].astype(np.int32)
    green_mask = (g > 180) & (r < 120) & (b < 120)
    if green_mask.sum() < 20:
        return None

    ys, xs = np.where(green_mask)
    green_y_min = int(ys.min()) + 250
    green_x_min = int(xs.min())
    green_x_max = int(xs.max())

    h, w = frame.shape[:2]
    y0 = max(0, green_y_min - 100)
    y1 = green_y_min
    x0 = max(0, green_x_min - 150)
    x1 = min(w, green_x_max + 150)

    band = frame[y0:y1, x0:x1]
    upscaled = cv2.resize(band, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    _, crop = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    cv2.imwrite("/tmp/debug_crop.png", crop)

    config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :-"
    raw = pytesseract.image_to_string(crop, config=config)

    candidates = []
    for line in raw.splitlines():
        cleaned = re.sub(r"[^A-Z0-9 :-]", "", line.upper()).strip()
        if not cleaned:
            continue
        if "SCORE" in cleaned:
            continue
        if re.fullmatch(r"[\d ]+", cleaned):
            continue
        candidates.append(cleaned)

    if not candidates:
        return None

    matched_components = [_match_component(c) for c in candidates]
    matched_components = [m for m in matched_components if m is not None]

    if not matched_components:
        return None

    return " + ".join(matched_components)
