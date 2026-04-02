import difflib
import logging
import re

import cv2
import numpy as np
import pytesseract

from .known_tricks import KNOWN_TRICKS


def detect_trick(frame: np.ndarray) -> str | None:
    """Detect trick name from a 750x1624 game frame.

    Crops the notification band, checks for green pixels as a presence signal,
    then OCRs the white text line above the green band.

    Returns the matched trick name (e.g. "540 FLIP") or None.
    """
    notification = frame[285:370, :]

    g = notification[:, :, 1].astype(np.int32)
    r = notification[:, :, 0].astype(np.int32)
    b = notification[:, :, 2].astype(np.int32)
    green_mask = (g > 180) & (r < 120) & (b < 120)
    if green_mask.sum() < 20:
        return None

    text_crop = frame[298:330, :]
    gray = cv2.cvtColor(text_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    thresh = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    raw = pytesseract.image_to_string(thresh, config=config)
    ocr_result = re.sub(r"[^A-Z0-9 ]", "", raw.upper()).strip()

    matches = difflib.get_close_matches(ocr_result, KNOWN_TRICKS, n=1, cutoff=0.4)
    if matches:
        return matches[0]

    logging.warning("trick_info_reader: no match for OCR output %r", ocr_result)
    return None
