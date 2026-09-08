"""Build Model 2 training clips: run Model 1 over expert recordings -> clip.json.

This is the missing bridge (step D+E) between the two BC models. Model 1 (the
trained trace extractor) predicts zero or more touch peaks from each frame of an
expert recording.  Peaks are associated causally across frames before
`bc.assemble.assemble_strokes` turns each touch track into the discrete strokes
Asher actually made; we write them as the `clip.json` that
`bc.sequence_dataset.SequenceDataset` consumes to train Model 2.

    expert clip dir (frame_*.png @ fps)
        --[Model 1 per frame]-->  zero or more (x, y) peaks
        --[causal tracking]----->  concurrent touch tracks
        --[assemble_strokes]---->  strokes [x0..y2,dur,easing,delay]
        --[write_clip_json]----->  <out>/<clip>/{frame_000000.png(symlink), clip.json}

The output clip dir is a valid SequenceDataset clip (frames named frame_%06d.png
+ clip.json = {"fps", "strokes":[{"params":[9], "t_start", "t_end"}]}).

Model 1 inference is pure torch; this script has no device/Appium deps.

Usage:
    # real: label every clip subdir under an expert corpus
    python scripts/data/build_bc_clips.py \
        --model notebooks/models/trace_extractor_v1.pth \
        --clips-root data/expert/<session> --out data/bc_clips/<session> --fps 30

    # smoke: synthesize a known clip, write clip.json, round-trip via SequenceDataset
    python scripts/data/build_bc_clips.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import numbers
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.bc.assemble import Stroke, assemble_strokes  # noqa: E402
from trueskate_ai.bc.frame_prep import prep_frame_rgb  # noqa: E402
from trueskate_ai.bc.gesture_tokens import STROKE_BOUNDS  # noqa: E402
from trueskate_ai.vision.touch_peaks import TouchPeak, extract_touch_peaks  # noqa: E402


# --- Model 1 inference -----------------------------------------------------

_LEGACY_MODEL_TYPE = "gaussian_bump_predictor_v1"
_TEMPORAL_MODEL_TYPE = "temporal_trace_predictor_v1"
_LEGACY_LATENCY_S = 0.45
_HISTORICAL_INFERENCE_DEFAULTS: dict[str, float | int] = {
    "peak_threshold": 0.30,
    "activity_threshold": 0.50,
    "peak_nms_radius_px": 6,
    "max_touches": 2,
}


@dataclass(frozen=True)
class LoadedTraceExtractor:
    """Loaded Model 1 plus the inference metadata needed by this bridge."""

    model: object
    h: int
    w: int
    model_type: str
    latency_s: float | None
    peak_threshold: float | None = None
    activity_threshold: float | None = None
    peak_nms_radius_px: int | None = None
    max_touches: int | None = None

    @property
    def is_temporal(self) -> bool:
        return self.model_type == _TEMPORAL_MODEL_TYPE


def _checkpoint_latency_s(ckpt: dict) -> float | None:
    """Read label latency from current or defensively supported metadata."""

    for container in (ckpt, ckpt.get("training_config"), ckpt.get("dataset_config"),
                      ckpt.get("metadata")):
        if isinstance(container, dict) and container.get("latency_s") is not None:
            return float(container["latency_s"])
    return None


def _validate_inference_value(name: str, value, *, context: str) -> float | int:
    """Validate one decoder setting without silently coercing malformed metadata."""

    qualified_name = f"{context}.{name}"
    if name in {"peak_threshold", "activity_threshold"}:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(
                f"{qualified_name} must be a finite number in [0, 1], got {value!r}"
            )
        result = float(value)
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError(
                f"{qualified_name} must be a finite number in [0, 1], got {value!r}"
            )
        return result

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        minimum = 0 if name == "peak_nms_radius_px" else 1
        raise ValueError(
            f"{qualified_name} must be an integer >= {minimum}, got {value!r}"
        )
    result = int(value)
    minimum = 0 if name == "peak_nms_radius_px" else 1
    if result < minimum:
        raise ValueError(
            f"{qualified_name} must be an integer >= {minimum}, got {value!r}"
        )
    return result


def _checkpoint_inference_config(ckpt: dict) -> dict[str, float | int | None]:
    """Read and validate optional top-level Model-1 decoder metadata."""

    raw = ckpt.get("inference_config")
    if raw is None:
        return {name: None for name in _HISTORICAL_INFERENCE_DEFAULTS}
    if not isinstance(raw, Mapping):
        raise ValueError(
            "checkpoint inference_config must be a mapping, "
            f"got {type(raw).__name__}"
        )

    values: dict[str, float | int | None] = {}
    for name in _HISTORICAL_INFERENCE_DEFAULTS:
        value = raw.get(name)
        values[name] = (
            None
            if value is None
            else _validate_inference_value(name, value, context="inference_config")
        )
    return values


def _checkpoint_model_type(ckpt: dict) -> str:
    """Identify legacy versus recurrent checkpoints without filename guesses."""

    explicit = ckpt.get("model_type")
    if explicit is None:
        # Legacy GaussianBumpPredictor checkpoints predate model_type.  A
        # temporal state dict is distinctive enough to produce a clear error
        # instead of accidentally loading it into the legacy architecture.
        state = ckpt.get("model_state", ckpt.get("state_dict", {}))
        if any(str(key).startswith(("recurrent.", "activity_head.")) for key in state):
            return _TEMPORAL_MODEL_TYPE
        return _LEGACY_MODEL_TYPE
    aliases = {
        "legacy": _LEGACY_MODEL_TYPE,
        "single_frame": _LEGACY_MODEL_TYPE,
        "gaussian_bump_predictor": _LEGACY_MODEL_TYPE,
        _LEGACY_MODEL_TYPE: _LEGACY_MODEL_TYPE,
        "temporal": _TEMPORAL_MODEL_TYPE,
        "temporal_trace": _TEMPORAL_MODEL_TYPE,
        "temporal_trace_predictor": _TEMPORAL_MODEL_TYPE,
        _TEMPORAL_MODEL_TYPE: _TEMPORAL_MODEL_TYPE,
    }
    try:
        return aliases[str(explicit).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported Model 1 checkpoint model_type={explicit!r}") from exc


def load_trace_extractor(model_path: Path, device) -> LoadedTraceExtractor:
    """Load either Model 1 architecture and its causal-inference metadata."""
    import torch  # local so --smoke path importing this module stays light

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Model 1 checkpoint must be a mapping, got {type(ckpt).__name__}")
    inference_config = _checkpoint_inference_config(ckpt)
    model_type = _checkpoint_model_type(ckpt)
    state = ckpt.get("model_state", ckpt.get("state_dict"))
    if state is None:
        raise ValueError(f"Model 1 checkpoint {model_path} has no model_state")

    if model_type == _TEMPORAL_MODEL_TYPE:
        from trueskate_ai.vision.temporal_trace_predictor import (
            TemporalTraceConfig,
            TemporalTracePredictor,
        )

        raw_config = ckpt.get("model_config", ckpt.get("config"))
        if raw_config is None:
            raise ValueError(
                f"temporal Model 1 checkpoint {model_path} has no model_config; "
                "retrain/save it with checkpoint_version=3"
            )
        if isinstance(raw_config, TemporalTraceConfig):
            config = raw_config
        elif isinstance(raw_config, dict):
            allowed = set(TemporalTraceConfig.__dataclass_fields__)
            config = TemporalTraceConfig(**{
                key: value for key, value in raw_config.items() if key in allowed
            })
        else:
            raise ValueError(
                f"temporal model_config must be a mapping, got {type(raw_config).__name__}"
            )
        model = TemporalTracePredictor.from_config(config)
    else:
        from trueskate_ai.vision.gaussian_bump_predictor import GaussianBumpPredictor

        model = GaussianBumpPredictor(
            in_channels=3, base_channels=int(ckpt.get("base_channels", 32))
        )

    model.load_state_dict(state)
    model.to(device).eval()
    try:
        h, w = int(ckpt["h"]), int(ckpt["w"])
    except KeyError as exc:
        raise ValueError(f"Model 1 checkpoint {model_path} has no h/w input geometry") from exc
    return LoadedTraceExtractor(
        model=model,
        h=h,
        w=w,
        model_type=model_type,
        latency_s=_checkpoint_latency_s(ckpt),
        peak_threshold=inference_config["peak_threshold"],
        activity_threshold=inference_config["activity_threshold"],
        peak_nms_radius_px=inference_config["peak_nms_radius_px"],
        max_touches=inference_config["max_touches"],
    )


def resolve_latency_s(extractor: LoadedTraceExtractor,
                      explicit_latency_s: float | None) -> tuple[float, str]:
    """Resolve touch/trace timing with explicit CLI precedence.

    Temporal checkpoints must carry their label latency (or receive an explicit
    override): silently borrowing the historical 0.45-second MJPEG value would
    corrupt XCTest-labelled stroke timing.  The fallback remains only for old
    single-frame checkpoints that could not have stored this metadata.
    """

    if explicit_latency_s is not None:
        return float(explicit_latency_s), "CLI override"
    if extractor.latency_s is not None:
        return float(extractor.latency_s), "checkpoint metadata"
    if extractor.is_temporal:
        raise ValueError(
            "temporal Model 1 checkpoint has no latency_s metadata; pass --latency-s "
            "explicitly or use a checkpoint_version=3 checkpoint"
        )
    return _LEGACY_LATENCY_S, "legacy 0.45s compatibility fallback"


def resolve_inference_config(
    extractor: LoadedTraceExtractor,
    *,
    peak_threshold: float | None = None,
    activity_threshold: float | None = None,
    peak_nms_radius_px: int | None = None,
    max_touches: int | None = None,
) -> tuple[dict[str, float | int], dict[str, str]]:
    """Resolve decoder settings as CLI → checkpoint → historical defaults."""

    explicit = {
        "peak_threshold": peak_threshold,
        "activity_threshold": activity_threshold,
        "peak_nms_radius_px": peak_nms_radius_px,
        "max_touches": max_touches,
    }
    resolved: dict[str, float | int] = {}
    sources: dict[str, str] = {}
    for name, fallback in _HISTORICAL_INFERENCE_DEFAULTS.items():
        if explicit[name] is not None:
            resolved[name] = _validate_inference_value(
                name, explicit[name], context="CLI"
            )
            sources[name] = "CLI override"
            continue
        checkpoint_value = getattr(extractor, name)
        if checkpoint_value is not None:
            resolved[name] = _validate_inference_value(
                name, checkpoint_value, context="checkpoint inference_config"
            )
            sources[name] = "checkpoint inference_config"
            continue
        resolved[name] = fallback
        sources[name] = "historical default"
    return resolved, sources


@dataclass
class TouchTrack:
    """Causally associated observations belonging to one physical touch.

    Missing detections are not fabricated.  A track can remain available for
    association across a short gap, while assembly later sees only genuinely
    observed coordinates and decides whether that gap is bridgeable.
    """

    frame_indices: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    def append(self, frame_index: int, time_s: float, peak: TouchPeak,
               *, velocity_smoothing: float) -> None:
        point = np.array([peak.x, peak.y], dtype=np.float64)
        if self.times:
            dt = float(time_s - self.times[-1])
            if dt > 1e-9:
                instant = (point - np.array([self.xs[-1], self.ys[-1]])) / dt
                if len(self.times) == 1:
                    self.velocity = instant
                else:
                    alpha = float(np.clip(velocity_smoothing, 0.0, 1.0))
                    self.velocity = alpha * instant + (1.0 - alpha) * self.velocity
        self.frame_indices.append(int(frame_index))
        self.times.append(float(time_s))
        self.xs.append(float(peak.x))
        self.ys.append(float(peak.y))
        self.scores.append(float(peak.score))

    def predict(self, time_s: float) -> np.ndarray:
        """Constant-velocity prediction using observations strictly from the past."""
        last = np.array([self.xs[-1], self.ys[-1]], dtype=np.float64)
        return last + self.velocity * max(0.0, float(time_s - self.times[-1]))


def track_touch_peaks(detections: list[list[TouchPeak]], times: np.ndarray, *,
                      max_gap_s: float = 0.10, max_match_distance: float = 0.18,
                      velocity_smoothing: float = 0.65) -> list[TouchTrack]:
    """Associate per-frame peaks into touch tracks using causal motion matching.

    Each open track predicts its next location with a smoothed constant velocity.
    Hungarian assignment gives a one-to-one global match, which prevents two
    nearby touches from both consuming the same peak.  A track may bridge a
    short missing-detection gap (for example, two bumps momentarily merge at a
    crossing), but never receives an invented observation.
    """
    from scipy.optimize import linear_sum_assignment

    frame_times = np.asarray(times, dtype=np.float64)
    if frame_times.ndim != 1 or len(frame_times) != len(detections):
        raise ValueError("times must be one-dimensional and match detections")
    if len(frame_times) > 1 and np.any(np.diff(frame_times) < 0):
        raise ValueError("times must be monotonically non-decreasing")
    if max_gap_s < 0 or max_match_distance <= 0:
        raise ValueError("max_gap_s must be >= 0 and max_match_distance must be > 0")

    tracks: list[TouchTrack] = []
    for frame_index, (time_s, peaks) in enumerate(zip(frame_times, detections)):
        if not peaks:
            continue
        eligible = [i for i, track in enumerate(tracks)
                    if float(time_s - track.times[-1]) <= max_gap_s + 1e-9]
        matched_peaks: set[int] = set()
        if eligible:
            costs = np.full((len(eligible), len(peaks)), np.inf, dtype=np.float64)
            for row, track_index in enumerate(eligible):
                track = tracks[track_index]
                predicted = track.predict(float(time_s))
                last = np.array([track.xs[-1], track.ys[-1]], dtype=np.float64)
                for col, peak in enumerate(peaks):
                    point = np.array([peak.x, peak.y], dtype=np.float64)
                    predicted_distance = float(np.linalg.norm(point - predicted))
                    last_distance = float(np.linalg.norm(point - last))
                    # Prediction is the primary identity cue at crossings; the
                    # small continuity term stabilises noisy velocity estimates.
                    if predicted_distance <= max_match_distance:
                        costs[row, col] = predicted_distance + 0.10 * last_distance
            if np.isfinite(costs).any():
                safe_costs = np.where(np.isfinite(costs), costs, 1e6)
                rows, cols = linear_sum_assignment(safe_costs)
                for row, col in zip(rows, cols):
                    if not np.isfinite(costs[row, col]):
                        continue
                    tracks[eligible[row]].append(frame_index, float(time_s), peaks[col],
                                                 velocity_smoothing=velocity_smoothing)
                    matched_peaks.add(int(col))

        for peak_index, peak in enumerate(peaks):
            if peak_index in matched_peaks:
                continue
            track = TouchTrack()
            track.append(frame_index, float(time_s), peak,
                         velocity_smoothing=velocity_smoothing)
            tracks.append(track)
    return tracks


def heatmaps_to_touch_tracks(heatmaps: np.ndarray, times: np.ndarray, *,
                             active_thresh: float = 0.30, max_touches: int = 2,
                             peak_nms_radius_px: int = 6,
                             track_max_gap_s: float = 0.10,
                             track_match_distance: float = 0.18) -> list[TouchTrack]:
    """Pure-numpy/scipy bridge from Model-1 heatmaps to temporal touch tracks."""
    maps = np.asarray(heatmaps)
    if maps.ndim != 3:
        raise ValueError(f"heatmaps must have shape (frames,h,w), got {maps.shape}")
    detections = [extract_touch_peaks(hm, threshold=active_thresh,
                                      max_peaks=max_touches,
                                      nms_radius_px=peak_nms_radius_px)
                  for hm in maps]
    return track_touch_peaks(detections, times, max_gap_s=track_max_gap_s,
                             max_match_distance=track_match_distance)


def frames_to_touch_detections(model, frame_paths: list[Path], times: np.ndarray,
                               h: int, w: int, device, *,
                               model_type: str = _LEGACY_MODEL_TYPE,
                               active_thresh: float = 0.30,
                               activity_thresh: float = 0.50,
                               max_touches: int = 2,
                               peak_nms_radius_px: int = 6,
                               batch: int = 32) -> list[list[TouchPeak]]:
    """Run Model 1 and return every spatial peak for each chronological frame.

    Legacy checkpoints are independent and can be batched.  Temporal Model 1 is
    deliberately stepped one frame at a time: ``state=None`` is the explicit
    clip-boundary reset, every later step consumes the prior prediction/state,
    and no ground-truth heatmap is available for teacher forcing.
    """
    import cv2
    import torch

    n = len(frame_paths)
    frame_times = np.asarray(times, dtype=np.float64)
    if frame_times.ndim != 1 or len(frame_times) != n:
        raise ValueError("times must be one-dimensional and match frame_paths")
    if n > 1 and np.any(np.diff(frame_times) < 0):
        raise ValueError("times must be monotonically non-decreasing")
    if not 0.0 <= activity_thresh <= 1.0:
        raise ValueError("activity_thresh must be in [0,1]")
    detections: list[list[TouchPeak]] = [[] for _ in range(n)]

    if model_type == _TEMPORAL_MODEL_TYPE:
        state = None  # explicit reset: state must never leak between expert clips
        use_time_deltas = bool(getattr(getattr(model, "config", None),
                                       "use_time_deltas", True))
        with torch.no_grad():
            for i, path in enumerate(frame_paths):
                img = cv2.imread(str(path))
                if img is None:
                    raise FileNotFoundError(path)
                x = torch.from_numpy(
                    prep_frame_rgb(img, h, w).transpose(2, 0, 1)[None]
                ).to(device)
                delta_t = 0.0 if i == 0 else float(frame_times[i] - frame_times[i - 1])
                # Supplying only state makes previous_heatmap exactly the last
                # model prediction.  feedback_heatmap is intentionally absent:
                # expert-label generation is a fully autoregressive rollout.
                output = model.step(
                    x,
                    state,
                    delta_t=delta_t if use_time_deltas else None,
                )
                state = output.state
                activity_probability = float(
                    torch.sigmoid(output.active_logits).reshape(-1)[0].item()
                )
                if activity_probability < activity_thresh:
                    continue
                heatmap = output.heatmap.detach().squeeze(0).squeeze(0).cpu().numpy()
                detections[i] = extract_touch_peaks(
                    heatmap,
                    threshold=active_thresh,
                    max_peaks=max_touches,
                    nms_radius_px=peak_nms_radius_px,
                )
        return detections

    if model_type != _LEGACY_MODEL_TYPE:
        raise ValueError(f"unsupported Model 1 model_type={model_type!r}")
    with torch.no_grad():
        for lo in range(0, n, batch):
            paths = frame_paths[lo:lo + batch]
            imgs = []
            for p in paths:
                img = cv2.imread(str(p))
                if img is None:
                    raise FileNotFoundError(p)
                imgs.append(prep_frame_rgb(img, h, w))
            x = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)).to(device)
            hm = model(x)                                    # (B,1,h,w) sigmoid
            hm = hm.squeeze(1).cpu().numpy()                 # (B,h,w)
            for j, hmap in enumerate(hm):
                detections[lo + j] = extract_touch_peaks(
                    hmap, threshold=active_thresh, max_peaks=max_touches,
                    nms_radius_px=peak_nms_radius_px,
                )
    return detections


def frames_to_touch_tracks(model, frame_paths: list[Path], times: np.ndarray,
                           h: int, w: int, device, *,
                           model_type: str = _LEGACY_MODEL_TYPE,
                           active_thresh: float = 0.30,
                           activity_thresh: float = 0.50,
                           max_touches: int = 2, peak_nms_radius_px: int = 6,
                           track_max_gap_s: float = 0.10,
                           track_match_distance: float = 0.18,
                           batch: int = 32) -> list[TouchTrack]:
    """Run Model 1 and preserve all distinct simultaneous touch tracks."""
    detections = frames_to_touch_detections(
        model,
        frame_paths,
        times,
        h,
        w,
        device,
        model_type=model_type,
        active_thresh=active_thresh,
        activity_thresh=activity_thresh,
        max_touches=max_touches,
        peak_nms_radius_px=peak_nms_radius_px,
        batch=batch,
    )
    return track_touch_peaks(detections, times, max_gap_s=track_max_gap_s,
                             max_match_distance=track_match_distance)


def touch_tracks_to_strokes(tracks: list[TouchTrack], *, max_gap_s: float = 0.10,
                            min_frames: int = 2) -> list[Stroke]:
    """Fit every physical touch independently, then merge in execution order.

    Fitting a track independently is what preserves overlapping strokes.  Once
    merged by start time, `delay_before` is recomputed against the preceding
    stroke, retaining a negative delay for concurrent W3C execution.
    """
    strokes: list[Stroke] = []
    for track in tracks:
        if not track.times:
            continue
        observed = np.ones(len(track.times), dtype=bool)
        strokes.extend(assemble_strokes(
            observed,
            np.asarray(track.xs, dtype=np.float64),
            np.asarray(track.ys, dtype=np.float64),
            np.asarray(track.times, dtype=np.float64),
            max_gap_s=max_gap_s,
            min_frames=min_frames,
        ))

    strokes.sort(key=lambda stroke: (stroke.t_start, stroke.t_end,
                                     float(stroke.params[0]), float(stroke.params[1])))
    delay_lo, delay_hi = STROKE_BOUNDS[-1]
    merged: list[Stroke] = []
    prev_end: float | None = None
    for stroke in strokes:
        params = np.array(stroke.params, dtype=np.float64, copy=True)
        params[-1] = (0.0 if prev_end is None else
                      float(np.clip(stroke.t_start - prev_end, delay_lo, delay_hi)))
        merged.append(Stroke(params=params, t_start=stroke.t_start, t_end=stroke.t_end))
        prev_end = stroke.t_end
    return merged


# --- clip.json writer ------------------------------------------------------

def _sorted_frames(clip_dir: Path) -> list[Path]:
    """Frames in a clip dir, temporal order. Prefer frame_* (png>jpg); else any image
    (covers legacy img_*.jpg from extract_frames.py)."""
    for pat in ("frame_*.png", "frame_*.jpg", "*.png", "*.jpg"):
        fp = sorted(clip_dir.glob(pat))
        if fp:
            return fp
    return []


def _extracted_fps(clip_dir: Path) -> float | None:
    """True achieved fps from extract_expert_frames.py's `_extract_meta.json`
    sidecar, if present. Extraction never upsamples, so a source video whose
    native fps is below --fps is written out slower than requested; trusting
    --fps blindly here would desync stroke t_start/t_end from the real frames."""
    meta_path = clip_dir / "_extract_meta.json"
    if not meta_path.exists():
        return None
    try:
        return float(json.loads(meta_path.read_text())["fps"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def write_clip_json(out_dir: Path, fps: float, strokes: list[Stroke],
                    src_frames: list[Path]) -> Path:
    """Write a SequenceDataset clip: frame_%06d.png symlinks + clip.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(src_frames):
        link = out_dir / f"frame_{i:06d}.png"
        if link.exists() or link.is_symlink():
            link.unlink()
        # symlink so we don't duplicate a 31GB corpus; resolve to absolute.
        link.symlink_to(src.resolve())
    meta = {
        "fps": float(fps),
        "strokes": [
            {"params": [float(v) for v in s.params], "t_start": s.t_start, "t_end": s.t_end}
            for s in strokes
        ],
    }
    (out_dir / "clip.json").write_text(json.dumps(meta, indent=2))
    return out_dir / "clip.json"


def build_clip(clip_dir: Path, out_dir: Path, model, h: int, w: int, device,
               *, fps: float, active_thresh: float, latency_s: float,
               model_type: str = _LEGACY_MODEL_TYPE,
               activity_thresh: float = 0.50,
               max_touches: int = 2, peak_nms_radius_px: int = 6,
               track_max_gap_s: float = 0.10,
               track_match_distance: float = 0.18) -> int:
    """Label one expert clip dir -> its clip.json. Returns #strokes assembled.

    Model 1 is trained (self_label `latency_s`) to predict the LAGGING orange
    trace — its touch at frame time `t` reflects the real finger at `t - latency`
    (see self_label.label_frames: `t = ft - latency_s`). So we place each frame's
    touch at `t - latency_s`, otherwise every assembled stroke's t_start/t_end is
    ~latency late and Model 2 would condition on post-touch frames. The shift is
    constant, so run-segmentation gaps/durations are unchanged.
    """
    frames = _sorted_frames(clip_dir)
    if not frames:
        raise RuntimeError(f"no frames in {clip_dir}")
    clip_fps = _extracted_fps(clip_dir) or fps
    times = np.arange(len(frames), dtype=np.float64) / clip_fps - latency_s
    tracks = frames_to_touch_tracks(
        model, frames, times, h, w, device,
        model_type=model_type,
        active_thresh=active_thresh,
        activity_thresh=activity_thresh,
        max_touches=max_touches,
        peak_nms_radius_px=peak_nms_radius_px,
        track_max_gap_s=track_max_gap_s,
        track_match_distance=track_match_distance,
    )
    strokes = touch_tracks_to_strokes(tracks, max_gap_s=track_max_gap_s)
    write_clip_json(out_dir, clip_fps, strokes, frames)
    return len(strokes)


# --- entry points ----------------------------------------------------------

def _run(args) -> None:
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    extractor = load_trace_extractor(args.model, device)
    latency_s, latency_source = resolve_latency_s(extractor, args.latency_s)
    inference_config, inference_sources = resolve_inference_config(
        extractor,
        peak_threshold=args.active_thresh,
        activity_threshold=args.activity_thresh,
        peak_nms_radius_px=args.peak_nms_radius_px,
        max_touches=args.max_touches,
    )
    clip_dirs = [d for d in sorted(args.clips_root.glob("**/")) if _sorted_frames(d)]
    if not clip_dirs:
        raise SystemExit(f"no clips with frames under {args.clips_root}")
    print(f"model {args.model.name}  type={extractor.model_type}  "
          f"({extractor.h}x{extractor.w})  device={device}  clips={len(clip_dirs)}")
    print(f"latency_s={latency_s:g} ({latency_source})")
    for name, value in inference_config.items():
        print(f"{name}={value:g} ({inference_sources[name]})")
    total = 0
    for cd in clip_dirs:
        rel = cd.relative_to(args.clips_root)
        out = args.out / rel
        n = build_clip(cd, out, extractor.model, extractor.h, extractor.w, device,
                       fps=args.fps,
                       active_thresh=float(inference_config["peak_threshold"]),
                       activity_thresh=float(inference_config["activity_threshold"]),
                       latency_s=latency_s,
                       model_type=extractor.model_type,
                       max_touches=int(inference_config["max_touches"]),
                       peak_nms_radius_px=int(inference_config["peak_nms_radius_px"]),
                       track_max_gap_s=args.track_max_gap_s,
                       track_match_distance=args.track_match_distance)
        total += n
        print(f"  {rel}: {n} strokes -> {out/'clip.json'}")
    print(f"done: {len(clip_dirs)} clips, {total} strokes total")


def _smoke() -> None:
    """Synthesize a known clip, write clip.json, and load it via SequenceDataset.

    Proves the writer + schema without a trained model or real corpus: build a
    per-frame touch track from two known gestures (as in assemble's self-test),
    assemble it, write a clip, then confirm SequenceDataset yields decision
    points with the right tensor shapes.
    """
    import tempfile

    import cv2
    from trueskate_ai.vision.self_label import label_frames
    from trueskate_ai.bc.model2 import SequencePolicyConfig
    from trueskate_ai.bc.sequence_dataset import SequenceDataset

    fps = 30.0
    g1 = dict(waypoints=[(0.30, 0.72), (0.46, 0.50), (0.74, 0.44)], dur=0.40)
    g2 = dict(waypoints=[(0.60, 0.40), (0.50, 0.60), (0.40, 0.80)], dur=0.30)
    t1 = np.arange(0.10, 0.10 + g1["dur"] + 1e-9, 1 / fps)
    t2 = np.arange(0.80, 0.80 + g2["dur"] + 1e-9, 1 / fps)
    clip_t = np.concatenate([np.arange(0.0, 0.10, 1 / fps), t1, np.arange(0.55, 0.80, 1 / fps), t2,
                             np.arange(1.15, 1.30, 1 / fps)])
    l1 = label_frames(g1["waypoints"], g1["dur"], 1.0, list(clip_t - t1[0]))
    l2 = label_frames(g2["waypoints"], g2["dur"], 1.0, list(clip_t - t2[0]))
    active = np.array([a.active or b.active for a, b in zip(l1, l2)])
    xs = np.array([a.x if a.active else b.x for a, b in zip(l1, l2)])
    ys = np.array([a.y if a.active else b.y for a, b in zip(l1, l2)])
    times = clip_t

    strokes = assemble_strokes(active, xs, ys, times)
    assert len(strokes) == 2, f"expected 2 strokes, got {len(strokes)}"

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        frame_paths = []
        for i in range(len(times)):
            p = src / f"frame_{i:06d}.png"
            cv2.imwrite(str(p), np.zeros((208, 96, 3), dtype=np.uint8))  # blank portrait frames
            frame_paths.append(p)
        out = Path(td) / "clip0"
        cj = write_clip_json(out, fps, strokes, frame_paths)
        assert cj.exists()
        loaded = json.loads(cj.read_text())
        assert loaded["fps"] == fps and len(loaded["strokes"]) == 2
        assert len(loaded["strokes"][0]["params"]) == 9

        cfg = SequencePolicyConfig()
        ds = SequenceDataset(Path(td), cfg=cfg)
        assert len(ds) >= 1, "no decision points produced"
        item = ds[0]
        assert tuple(item["frames"].shape) == (cfg.n_frames, 3, cfg.img_h, cfg.img_w), item["frames"].shape
        assert tuple(item["target"].shape) == (cfg.m_out, 9), item["target"].shape
    print("SMOKE OK — synthesized clip.json round-trips through SequenceDataset "
          f"(2 strokes, frames {cfg.n_frames}×3×{cfg.img_h}×{cfg.img_w})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Model 2 clip.json from expert recordings via Model 1.")
    ap.add_argument("--smoke", action="store_true", help="Synthetic writer+schema round-trip; no model/corpus.")
    ap.add_argument("--model", type=Path, help="Model 1 checkpoint (trace_extractor_*.pth).")
    ap.add_argument("--clips-root", type=Path, help="Dir whose subdirs are expert clips (frames each).")
    ap.add_argument("--out", type=Path, help="Output root; mirrors --clips-root structure.")
    ap.add_argument("--fps", type=float, default=30.0, help="Frame rate of the expert recordings.")
    ap.add_argument("--active-thresh", type=float, default=None,
                    help="Min peak heatmap response to count a frame as a touch. Default: "
                         "checkpoint inference_config, then 0.30.")
    ap.add_argument("--activity-thresh", type=float, default=None,
                    help="Temporal Model 1 activity-head probability gate; a spatial peak "
                         "must also clear --active-thresh (ignored for legacy Model 1). "
                         "Default: checkpoint inference_config, then 0.50.")
    ap.add_argument("--max-touches", type=int, default=None,
                    help="Maximum simultaneous peaks. Default: checkpoint inference_config, "
                         "then 2.")
    ap.add_argument("--peak-nms-radius-px", type=int, default=None,
                    help="Heatmap-pixel NMS radius. Default: checkpoint inference_config, "
                         "then 6.")
    ap.add_argument("--track-max-gap-s", type=float, default=0.10,
                    help="Maximum missing-detection gap over which a touch track stays open.")
    ap.add_argument("--track-match-distance", type=float, default=0.18,
                    help="Maximum normalised residual from a track's motion prediction.")
    ap.add_argument("--latency-s", type=float, default=None,
                    help="Explicit trace-latency override. Default: checkpoint latency_s; only "
                         "metadata-free legacy checkpoints fall back to 0.45s. Temporal "
                         "checkpoints without metadata fail instead of assuming the wrong path.")
    args = ap.parse_args()

    if args.smoke:
        _smoke()
        return
    if not (args.model and args.clips_root and args.out):
        ap.error("real mode needs --model, --clips-root, and --out (or use --smoke)")
    _run(args)


if __name__ == "__main__":
    main()
