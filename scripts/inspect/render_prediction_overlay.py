"""Composite Model 1's predicted gesture onto a clip's own frames, as video.

Aggregate recovery says how often a prediction lands inside tolerance; it says
nothing about how the rest go wrong.  A static predicted line is no better: it
can trace the right shape while concealing that the model thinks the finger
travelled in the wrong direction or at the wrong speed.  So the overlay is
animated -- a moving dot, a decaying trail, explicit touch-down/lift-off.

Three panes per frame: the raw clip, the commanded gesture, the prediction.
Both animations use stored frame times and assume touch-down at t=0; the
model does not predict onset. These are commanded/predicted positions, not
measured finger locations. FFmpeg decoding does not establish synchronization.
Selection uses fixed hit/miss quotas and is not an accuracy estimate.

Run:
  python scripts/inspect/render_prediction_overlay.py \
      --corpus /abs/model1_linear_20260902 --checkpoint /abs/baseline.pth \
      --manifest /abs/manifest.json --out /tmp
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from trueskate_ai.data.clip_frames import _decode_even_frames, _decode_frames  # noqa: E402
from trueskate_ai.data.touch_labels import label_frames  # noqa: E402
from trueskate_ai.model1.linear.dataset import BasicLinearClipDataset  # noqa: E402
from trueskate_ai.model1.linear.regressor import BasicLinearRegressor  # noqa: E402
from trueskate_ai.model1.linear.training import (  # noqa: E402
    RECOVERY_DURATION_TOLERANCE_S, RECOVERY_ENDPOINT_TOLERANCE,
)

# BGR.  Ground truth green, prediction magenta -- distinct from the game's
# orange trace, which neither may be confused with.
_TRUTH = (0, 235, 0)
_PRED = (235, 0, 235)
_INK = (235, 235, 235)
_DIM = (120, 120, 120)
_BG = (18, 18, 18)
_TRAIL_LENGTH = 6
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _model_from_payload(payload: dict) -> BasicLinearRegressor:
    """Rebuild a checkpoint's recorded inference architecture.

    Mirrors ``scripts/model1/train_basic_linear_modal.py::_model_from_payload``.
    It is duplicated rather than imported because that module builds Modal app
    objects at import time; an offline inspection tool must not need Modal.
    """
    return BasicLinearRegressor(
        base_channels=int(payload["base_channels"]),
        start_onset=float(payload.get("start_onset", .24)),
        start_sigma=float(payload.get("start_sigma", .05)),
        end_onset=float(payload.get("end_onset", .24)),
        temporal_mixer=bool(payload.get("temporal_mixer", False)),
        trajectory_track=bool(payload.get("trajectory_track", False)),
        line_fit=bool(payload.get("line_fit", False)),
        irls_iterations=int(payload.get("irls_iterations") or 3),
        huber_delta=float(payload.get("huber_delta") or .02),
        knots=int(payload.get("knots") or 2),
    )


def _payload_dataset_kwargs(payload: dict) -> dict:
    """Decode shape the checkpoint was trained against; see the Modal twin."""
    return {"image_width": int(payload.get("image_width") or 128),
            "image_height": int(payload.get("image_height") or 288),
            "knots": int(payload.get("knots") or 2)}


def _decode(sample: Path, count: int, mode: str) -> list[np.ndarray]:
    """Evenly spaced clip frames, by one of three decoders.

    This matters more than it should.  On this build OpenCV yields only 29 of
    the 31 frames ffmpeg and ffprobe both count in these clips, on every clip
    sampled -- and ``cv2.CAP_PROP_FRAME_COUNT`` reports 31 regardless, so the
    seek path's indices are computed against a length it never delivers.  A
    short decode compresses the clip's time axis, which is precisely the axis
    the touch onset and the predicted duration are read off.

    ``opencv-seek`` and ``opencv-sequential`` are the library's own paths, kept
    so the tool can reproduce what the repo's loaders see.  ``ffmpeg`` decodes
    the full clip and is the default.
    """
    if mode == "opencv-seek":
        return _decode_even_frames(sample, count)
    if mode == "opencv-sequential":
        full = _decode_frames(sample)
    else:
        full = _ffmpeg_frames(sample)
    return [full[int(index)] for index in np.linspace(0, len(full) - 1, count).round().astype(int)]


def _ffmpeg_frames(sample: Path) -> list[np.ndarray]:
    """Every frame of the clip, as BGR arrays, decoded by ffmpeg."""
    video = sample / "frames.mp4"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:nk=1", str(video)],
        capture_output=True, text=True, check=True)
    width, height = (int(value) for value in probe.stdout.strip().split(","))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    if not len(frames):
        raise ValueError(f"{sample}: ffmpeg decoded no frames")
    return [frames[index] for index in range(len(frames))]


def _clip_frames(sample: Path, count: int, mode: str) -> tuple[list[np.ndarray], list[float]]:
    """Displayed frames and their touch-relative times.

    ``meta["n_frames"]`` is 32 while the mp4 holds 31 (the known EQ-018
    off-by-one), so images and times are resampled independently -- exactly as
    ``BasicLinearClipDataset.__getitem__`` does.
    """
    images = _decode(sample, count, mode)
    raw = np.asarray(json.loads((sample / "meta.json").read_text())["frame_times"], dtype=np.float32)
    selected = np.linspace(0, len(raw) - 1, count).round().astype(int)
    return images, [float(value) for value in raw[selected]]


def _model_input(images: list[np.ndarray], height: int, width: int) -> torch.Tensor:
    """Normalise decoded frames exactly as BasicLinearClipDataset.__getitem__ does.

    Built here rather than taken from the dataset so the prediction and the
    pixels on screen come from one decode; otherwise the overlay could describe
    frames the model never saw.
    """
    frames = [cv2.cvtColor(
        image if image.shape[:2] == (height, width)
        else cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB) for image in images]
    tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    return tensor.float().div_(255.0)


@torch.inference_mode()
def _predict(model: BasicLinearRegressor, sample: Path, payload: dict,
             mode: str) -> tuple[list[float], list[float], list[np.ndarray], list[float]]:
    """Predicted gesture, commanded target, and the frames both were read from.

    The dataset is rooted at the sample itself: it still enforces the strict
    linear admission rules and supplies the target, without discovering all
    13,100 clips to reach one.  Only the frames are decoded separately, so that
    what is scored and what is drawn cannot diverge.
    """
    kwargs = _payload_dataset_kwargs(payload)
    dataset = BasicLinearClipDataset(sample, **kwargs)
    if len(dataset) != 1:
        raise SystemExit(f"{sample}: strict loader admitted {len(dataset)} samples, expected 1")
    target = [float(v) for v in dataset[0]["target"]]
    images, times = _clip_frames(sample, int(payload.get("sequence_length") or 32), mode)
    frames = _model_input(images, kwargs["image_height"], kwargs["image_width"])
    out = model.predict_linear(frames.unsqueeze(0))
    predicted = [float(out[key][0]) for key in ("x0", "y0", "x1", "y1", "dur")]
    return predicted, target, images, times


def _errors(predicted: list[float], target: list[float]) -> dict:
    start = float(np.hypot(predicted[0] - target[0], predicted[1] - target[1]))
    end = float(np.hypot(predicted[2] - target[2], predicted[3] - target[3]))
    duration = abs(predicted[4] - target[4])
    return {"start_error": start, "end_error": end, "duration_error": duration,
            "recovered": bool(start <= RECOVERY_ENDPOINT_TOLERANCE
                              and end <= RECOVERY_ENDPOINT_TOLERANCE
                              and duration <= RECOVERY_DURATION_TOLERANCE_S)}


def _touch_down(canvas: np.ndarray, point: tuple[int, int], colour) -> None:
    """Circle with an inscribed cross: the finger arrived here."""
    cv2.circle(canvas, point, 9, colour, 2, cv2.LINE_AA)
    cv2.line(canvas, (point[0] - 6, point[1]), (point[0] + 6, point[1]), colour, 1, cv2.LINE_AA)
    cv2.line(canvas, (point[0], point[1] - 6), (point[0], point[1] + 6), colour, 1, cv2.LINE_AA)


def _lift_off(canvas: np.ndarray, point: tuple[int, int], colour) -> None:
    """Circle with an inscribed X: the finger left here."""
    cv2.circle(canvas, point, 9, colour, 2, cv2.LINE_AA)
    for dx in (-1, 1):
        cv2.line(canvas, (point[0] - 5 * dx, point[1] - 5), (point[0] + 5 * dx, point[1] + 5),
                 colour, 1, cv2.LINE_AA)


def _pane(image: np.ndarray, scale: int, gesture: list[float] | None,
          labels: list | None, index: int, colour) -> np.ndarray:
    """One upscaled frame, optionally carrying an animated gesture overlay."""
    height, width = image.shape[:2]
    canvas = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
    if gesture is None or labels is None:
        return canvas

    def point(x: float, y: float) -> tuple[int, int]:
        return int(round(x * width * scale)), int(round(y * height * scale))

    x0, y0, x1, y1, _duration = gesture
    # The complete path, dimmed: the shape, always visible for comparison.
    overlay = canvas.copy()
    cv2.line(overlay, point(x0, y0), point(x1, y1), colour, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, .45, canvas, .55, 0, canvas)

    # A decaying trail over recent active frames.  Direction and speed live
    # here: a correct shape traversed backwards looks identical without it.
    trail = [labels[i] for i in range(max(0, index - _TRAIL_LENGTH), index + 1) if labels[i].active]
    for age, (older, newer) in enumerate(zip(trail, trail[1:])):
        weight = (age + 1) / max(1, len(trail))
        shade = tuple(int(channel * (.35 + .65 * weight)) for channel in colour)
        cv2.line(canvas, point(older.x, older.y), point(newer.x, newer.y),
                 shade, max(1, int(round(weight * 3))), cv2.LINE_AA)

    # Both markers latch once passed, so a paused frame still says what has
    # already happened rather than only what is happening now.
    now = labels[index]
    touched = any(labels[i].active for i in range(index + 1))
    if touched:
        _touch_down(canvas, point(x0, y0), colour)
    if touched and not now.active:
        _lift_off(canvas, point(x1, y1), colour)
    if now.active:
        cv2.circle(canvas, point(now.x, now.y), 6, colour, -1, cv2.LINE_AA)
        cv2.circle(canvas, point(now.x, now.y), 6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _state_text(labels: list, index: int, time_s: float) -> str:
    if labels[index].active:
        return f"DOWN  t={time_s:+.3f}s"
    return "PRE-TOUCH" if not any(labels[i].active for i in range(index + 1)) else "LIFTED"


def _compose(image: np.ndarray, index: int, time_s: float, truth: list[float], predicted: list[float],
             truth_labels: list, pred_labels: list, meta: dict, errors: dict,
             sample_label: str, scale: int) -> np.ndarray:
    panes = [("RAW", _pane(image, scale, None, None, index, _INK), None, _INK),
             ("COMMAND", _pane(image, scale, truth, truth_labels, index, _TRUTH),
              _state_text(truth_labels, index, time_s), _TRUTH),
             ("PREDICTION", _pane(image, scale, predicted, pred_labels, index, _PRED),
              _state_text(pred_labels, index, time_s), _PRED)]
    pane_h, pane_w = panes[0][1].shape[:2]
    gap, header, caption = 8, 34, 115
    width = pane_w * 3 + gap * 4
    canvas = np.full((header + pane_h + caption + gap, width, 3), _BG, np.uint8)
    for column, (title, pane, state, colour) in enumerate(panes):
        left = gap + column * (pane_w + gap)
        canvas[header:header + pane_h, left:left + pane_w] = pane
        cv2.rectangle(canvas, (left - 1, header - 1), (left + pane_w, header + pane_h), _DIM, 1)
        cv2.putText(canvas, title, (left + 2, 15), _FONT, .42, colour, 1, cv2.LINE_AA)
        if state:
            cv2.putText(canvas, state, (left + 2, 29), _FONT, .38, _INK, 1, cv2.LINE_AA)

    base = header + pane_h + gap + 16
    verdict = "RECOVERED" if errors["recovered"] else "MISSED"
    lines = [
        (f"{sample_label}   {meta.get('park', '?')}   {meta.get('device', '?')}"
         f"   frame {index + 1}/32   t={time_s:+.3f}s", _INK),
        ("GT   x0,y0 -> x1,y1  "
         f"{truth[0]:.4f},{truth[1]:.4f} -> {truth[2]:.4f},{truth[3]:.4f}   dur {truth[4]:.4f}s", _TRUTH),
        ("PRED x0,y0 -> x1,y1  "
         f"{predicted[0]:.4f},{predicted[1]:.4f} -> {predicted[2]:.4f},{predicted[3]:.4f}"
         f"   dur {predicted[4]:.4f}s", _PRED),
        (f"start_err {errors['start_error']:.4f}   end_err {errors['end_error']:.4f}"
         f"   dur_err {errors['duration_error']:.4f}s"
         f"   [tol {RECOVERY_ENDPOINT_TOLERANCE} / {RECOVERY_DURATION_TOLERANCE_S}s]   {verdict}",
         _TRUTH if errors["recovered"] else (60, 60, 255)),
    ]
    lines.append(("TIMING ASSUMED: stored t=0; onset is not predicted or verified", _INK))
    for row, (text, colour) in enumerate(lines):
        cv2.putText(canvas, text, (gap, base + row * 19), _FONT, .40, colour, 1, cv2.LINE_AA)
    # yuv420p needs even dimensions; pad rather than resize so nothing shifts.
    pad_h, pad_w = canvas.shape[0] % 2, canvas.shape[1] % 2
    if pad_h or pad_w:
        canvas = np.pad(canvas, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=18)
    return canvas


def _encode(frames: list[np.ndarray], target: Path, fps: float) -> None:
    with tempfile.TemporaryDirectory() as work:
        for index, frame in enumerate(frames):
            assert cv2.imwrite(str(Path(work) / f"frame_{index:03d}.png"), frame)
        # -vsync was removed in FFmpeg 9; -fps_mode is its replacement.
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.6f}",
             "-i", str(Path(work) / "frame_%03d.png"), "-fps_mode", "passthrough",
             "-c:v", "libx264", "-preset", "veryslow", "-crf", "16",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
            check=True)


def _render(sample: Path, model: BasicLinearRegressor, payload: dict, out: Path,
            scale: int, slowdown: float, label: str, mode: str) -> dict:
    meta = json.loads((sample / "meta.json").read_text())
    predicted, truth, images, times = _predict(model, sample, payload, mode)
    errors = _errors(predicted, truth)

    truth_labels = label_frames([tuple(point) for point in meta["waypoints"]],
                                float(meta["duration"]), float(meta.get("easing_power", 1.0)), times)
    # The model emits a two-point constant-velocity drag and no onset, so its
    # touch-down is the clip's own t=0 -- the same time base as the command.
    pred_labels = label_frames([(predicted[0], predicted[1]), (predicted[2], predicted[3])],
                               predicted[4], 1.0, times)

    frames = [_compose(image, index, times[index], truth, predicted, truth_labels,
                       pred_labels, meta, errors, label, scale)
              for index, image in enumerate(images)]
    span = max(times[-1] - times[0], 1e-3)
    fps = len(frames) / (span * slowdown)
    target = out / f"{label}.mp4"
    _encode(frames, target, fps)
    return {"sample": str(sample), "video": str(target), "predicted": predicted,
            "target": truth, "fps": fps, "frames": len(frames), "decode": mode, **errors}


def _candidates(manifest: Path, corpus: Path, partition: str, seed: int) -> list[Path]:
    entries = json.loads(manifest.read_text())["entries"]
    paths = []
    for entry in entries:
        if entry.get("partition") != partition:
            continue
        candidate = Path(entry["path"])
        if not candidate.exists():  # rebase a relocated manifest onto --corpus
            parts = candidate.parts
            candidate = corpus.joinpath(*parts[parts.index(corpus.name) + 1:]) \
                if corpus.name in parts else corpus / candidate.name
        paths.append(candidate)
    random.Random(seed).shuffle(paths)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="corpus root holding sample dirs")
    parser.add_argument("--checkpoint", type=Path, required=True, help="basic linear .pth payload")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON with per-entry partition")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partition", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--require-miss", type=int, default=1,
                        help="exact miss quota; remaining clips must pass recovery")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--slowdown", type=float, default=4.0,
                        help="playback stretch relative to the stored frame-time span")
    parser.add_argument("--decode", default="ffmpeg",
                        choices=("ffmpeg", "opencv-sequential", "opencv-seek"),
                        help="FFmpeg default; OpenCV modes reproduce the shared loader for comparison")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    for path in (args.corpus, args.checkpoint, args.manifest):
        if not path.exists():
            raise SystemExit(f"missing: {path}")
    if args.count < 1 or not 0 <= args.require_miss <= args.count:
        raise SystemExit("require count >= 1 and 0 <= require-miss <= count")
    if args.scale < 1 or not np.isfinite(args.slowdown) or args.slowdown <= 0:
        raise SystemExit("scale must be >= 1 and slowdown finite and > 0")
    args.out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _model_from_payload(payload)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    wanted_miss = args.require_miss
    wanted_hit = args.count - wanted_miss
    # Score until both quotas fill. Every screened clip is an exposure, even
    # if not rendered. Never interpret this quota-selected set as accuracy.
    chosen: list[tuple[Path, bool]] = []
    screened = []
    hits = misses = 0
    for sample in _candidates(args.manifest, args.corpus, args.partition, args.seed):
        if hits >= wanted_hit and misses >= wanted_miss:
            break
        if not (sample / "meta.json").exists():
            continue
        predicted, truth, _, _ = _predict(model, sample, payload, args.decode)
        recovered = _errors(predicted, truth)["recovered"]
        screened.append({"sample": str(sample), "recovered": recovered})
        if recovered and hits >= wanted_hit:
            continue
        if not recovered and misses >= wanted_miss:
            continue
        chosen.append((sample, recovered))
        hits += int(recovered)
        misses += int(not recovered)

    if hits < wanted_hit or misses < wanted_miss:
        raise SystemExit(f"quota unmet: {hits}/{wanted_hit} recovered, {misses}/{wanted_miss} missed")

    records: list[dict] = []
    for index, (sample, recovered) in enumerate(chosen):
        label = (f"m1_overlay_{index:02d}_{'hit' if recovered else 'MISS'}"
                 f"_{sample.parent.name}_{sample.name}")
        record = _render(sample, model, payload, args.out, args.scale, args.slowdown,
                         label, args.decode)
        record["partition"] = args.partition
        records.append(record)
        print(f"{label}: {'RECOVERED' if recovered else 'MISSED'} "
              f"start={record['start_error']:.4f} end={record['end_error']:.4f} "
              f"dur={record['duration_error']:.4f}s -> {record['video']}")

    sidecar = args.out / "overlays.json"
    sidecar.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "corpus": str(args.corpus), "partition": args.partition, "seed": args.seed,
        "slowdown": args.slowdown, "scale": args.scale, "decode": args.decode,
        "endpoint_tolerance": RECOVERY_ENDPOINT_TOLERANCE,
        "duration_tolerance_s": RECOVERY_DURATION_TOLERANCE_S,
        "timing": "stored frame times; both onsets assumed t=0, unverified",
        "selection": {"method": "fixed hit/miss quotas", "misses": wanted_miss,
                      "hits": wanted_hit, "screened": screened},
        "clips": records,
    }, indent=2) + "\n")
    print(f"wrote {len(records)} overlays ({misses} missed) to {args.out}; sidecar {sidecar}")

    if not args.no_open:
        subprocess.run(["open", "-a", "QuickTime Player", *[r["video"] for r in records]], check=False)


if __name__ == "__main__":
    main()
