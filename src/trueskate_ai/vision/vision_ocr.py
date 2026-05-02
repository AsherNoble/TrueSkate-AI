import cv2
import numpy as np

try:
    from Foundation import NSData
    import Quartz
    import Vision
except Exception as exc:  # pragma: no cover - import behavior is platform-specific
    NSData = None
    Quartz = None
    Vision = None
    _VISION_IMPORT_ERROR = exc
else:
    _VISION_IMPORT_ERROR = None


def is_vision_available() -> bool:
    return _VISION_IMPORT_ERROR is None


def vision_unavailable_reason() -> str:
    if _VISION_IMPORT_ERROR is None:
        return ""
    return str(_VISION_IMPORT_ERROR)


def _as_cgimage(image: np.ndarray):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("vision_ocr: failed to encode OCR crop as PNG")

    png_bytes = encoded.tobytes()
    data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
    image_source = Quartz.CGImageSourceCreateWithData(data, None)
    if image_source is None:
        raise RuntimeError("vision_ocr: failed to create image source from PNG bytes")

    cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
    if cg_image is None:
        raise RuntimeError("vision_ocr: failed to create CGImage from image source")
    return cg_image


def image_to_lines(image: np.ndarray, *, min_confidence: float = 0.05) -> list[str]:
    if not is_vision_available():
        raise RuntimeError(
            "vision_ocr: Apple Vision is unavailable. "
            "Install pyobjc-core, pyobjc-framework-Cocoa, pyobjc-framework-Quartz, "
            f"and pyobjc-framework-Vision. Cause: {vision_unavailable_reason()}"
        )

    cg_image = _as_cgimage(image)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)
    request.setRecognitionLanguages_(["en-US"])
    request.setCustomWords_([
        "OLLIE", "NOLLIE", "KICKFLIP", "HEELFLIP", "SHOVE", "POP", "BACKSIDE",
        "FRONTSIDE", "VARIAL", "NIGHTMARE", "LASER", "HARD", "FLIP",
        "DOUBLE", "TRIPLE", "QUAD", "GRIND", "SLIDE", "MANUAL",
    ])

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"vision_ocr: VNImageRequestHandler failed: {error}")

    observations = request.results() or []
    lines_with_position: list[tuple[float, float, str]] = []
    for observation in observations:
        top_candidates = observation.topCandidates_(1)
        if not top_candidates:
            continue
        candidate = top_candidates[0]
        confidence = float(candidate.confidence())
        if confidence < min_confidence:
            continue
        text = str(candidate.string()).strip()
        if not text:
            continue
        box = observation.boundingBox()
        lines_with_position.append((float(box.origin.y), float(box.origin.x), text))

    lines_with_position.sort(key=lambda item: (-item[0], item[1]))
    return [line for _, _, line in lines_with_position]
