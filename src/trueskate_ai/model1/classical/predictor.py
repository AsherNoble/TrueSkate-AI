"""Colour-component tracking from elapsed video time, never command metadata."""
from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    hue_low: int = 5
    hue_high: int = 28
    saturation: int = 80
    value: int = 90
    difference: int = 70
    min_area: int = 3
    position: str = 'centroid'
    motion_fraction: float = .12
    duration_correction: float = 0.0
    endpoint_extension: float = 0.0


def extract(frames, times, config=Config()):
    """Return up to five orange components per frame, in normalized coordinates."""
    frames = np.asarray(frames)
    times = np.asarray(times, dtype=float)
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(times) != len(frames):
        raise ValueError('expected BGR frames [time,height,width,3] and elapsed times')
    if len(times) < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('need at least three strictly increasing finite timestamps')
    h, w = frames.shape[1:3]
    reference = np.median(frames[:min(5, len(frames))], axis=0)
    result = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        diff = np.abs(frame.astype(float) - reference).sum(axis=2)
        mask = ((hsv[..., 0] >= config.hue_low) & (hsv[..., 0] <= config.hue_high)
                & (hsv[..., 1] >= config.saturation) & (hsv[..., 2] >= config.value)
                & (diff >= config.difference)).astype('uint8')
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for i in sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:5]:
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < config.min_area:
                continue
            ys, xs = np.where(labels == i)
            xy = np.column_stack((xs / (w - 1), ys / (h - 1)))
            center = xy.mean(axis=0)
            _, _, axes = np.linalg.svd(xy - center, full_matrices=False)
            direction = axes[0]
            projection = (xy - center) @ direction
            lo, hi = np.quantile(projection, [.05, .95])
            candidates.append({'area': area, 'center': center.tolist(),
                               'ends': [(center + lo * direction).tolist(),
                                        (center + hi * direction).tolist()]})
        result.append(candidates)
    return {'times': times.tolist(), 'components': result}


def predict_features(features, config=Config()):
    """Select a continuous moving component and estimate its active movement span."""
    times = np.asarray(features['times'])
    tracks = []
    for index, candidates in enumerate(features['components']):
        available = set(range(len(candidates)))
        for track in sorted(tracks, key=len, reverse=True):
            if index - track[-1][0] > 2 or not available:
                continue
            last = np.array(track[-1][1]['center'])
            chosen = min(available, key=lambda j: np.linalg.norm(np.array(candidates[j]['center']) - last))
            if np.linalg.norm(np.array(candidates[chosen]['center']) - last) <= .25:
                track.append((index, candidates[chosen])); available.remove(chosen)
        for chosen in sorted(available):
            tracks.append([(index, candidates[chosen])])
    tracks = [t for t in tracks if len(t) >= 3]
    if not tracks:
        return None
    def score(track):
        xy = np.array([c['center'] for _, c in track])
        return np.linalg.norm(xy[-1] - xy[0]) * np.sqrt(len(track))
    track = max(tracks, key=score)
    indices = np.array([i for i, _ in track])
    xy = np.array([c['center'] for _, c in track])
    displacement = xy[-1] - xy[0]
    length = np.linalg.norm(displacement)
    if length < .015:
        return None
    direction = displacement / length
    if config.position == 'tip':
        xy = np.array([max(c['ends'], key=lambda p: np.dot(p, direction)) for _, c in track])
    progress = xy @ direction
    speed = np.diff(progress) / np.diff(times[indices])
    moving = np.where(speed > max(.01, float(np.max(speed)) * config.motion_fraction))[0]
    if not len(moving):
        return None
    first, last = moving[0], moving[-1] + 1
    start, end = xy[first].copy(), xy[last].copy()
    start -= direction * config.endpoint_extension
    end += direction * config.endpoint_extension
    duration = float(times[indices[last]] - times[indices[first]] + config.duration_correction)
    if duration <= 0:
        return None
    return [*np.clip(start, 0, 1), *np.clip(end, 0, 1), duration]


def predict(frames, elapsed_times, config=Config()):
    return predict_features(extract(frames, elapsed_times, config), config)


def metrics(predictions, targets):
    """All-clip denominator; failures have no finite error but always fail recovery."""
    if not len(targets) or len(predictions) != len(targets):
        raise ValueError('nonempty equal-length predictions and targets required')
    errors = []; recovered = []; missing = 0
    for pred, target in zip(predictions, targets):
        if pred is None or np.asarray(pred).shape != (5,) or not np.isfinite(pred).all():
            missing += 1; recovered.append(False); continue
        p, t = np.asarray(pred), np.asarray(target)
        e = [np.linalg.norm(p[:2] - t[:2]), np.linalg.norm(p[2:4] - t[2:4]), abs(p[4] - t[4])]
        errors.append(e)
        recovered.append(e[0] <= .03 and e[1] <= .03 and e[2] <= .10)
    a = np.asarray(errors)
    return {'samples': len(targets), 'recovered': int(sum(recovered)),
            'recovery': float(np.mean(recovered)), 'missing': missing,
            'conditional_error_mean': a.mean(axis=0).tolist() if len(a) else None,
            'conditional_error_median': np.median(a, axis=0).tolist() if len(a) else None,
            'conditional_error_p90': np.quantile(a, .9, axis=0).tolist() if len(a) else None}
