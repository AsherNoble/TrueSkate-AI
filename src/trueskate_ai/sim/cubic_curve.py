"""Nine-number Bernstein cubic in time; exact-ms, quantized continuous-touch compiler.

One cubic cannot describe arbitrary trajectories. No easing, clipping or repair.
Legacy gesture schemas and their duration truncation remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from trueskate_ai.data.control_hitboxes import CONTROL_ENDPOINT_EXCLUSIONS, segment_is_safe
from trueskate_ai.sim.gestures import X_BOUND_MIN, X_BOUND_MAX, Y_BOUND_MIN, Y_BOUND_MAX

SCHEMA = 'cubic_in_time_v1'
CENTRAL_RECT = (.27, .22, .92, .82)


def _hull(points):
    points = sorted(set(map(tuple, points)))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    halves = []
    for seq in (points, points[::-1]):
        half = []
        for p in seq:
            while len(half) >= 2 and cross(*half[-2:], p) <= 0:
                half.pop()
            half.append(p)
        halves.append(half)
    return np.asarray(halves[0][:-1] + halves[1][:-1] or points, dtype=float)


def _hull_touches_rect(hull, rect):
    x0, y0, x1, y1 = rect
    box = np.array([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])
    axes = [np.array([1.,0.]), np.array([0.,1.])]
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        edge = b-a
        if np.any(edge):
            axes.append(np.array([-edge[1],edge[0]]))
    return all(not ((hull@axis).max() < (box@axis).min() or
                    (box@axis).max() < (hull@axis).min()) for axis in axes)


@dataclass(frozen=True)
class CubicInTime:
    coefficients: tuple[tuple[float, float], ...]
    duration_s: float

    def __post_init__(self):
        p = np.asarray(self.coefficients, dtype=float)
        if p.shape != (4, 2) or not np.isfinite(p).all():
            raise ValueError('four finite 2D Bernstein coefficients required')
        if isinstance(self.duration_s, bool) or not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError('duration_s must be finite and positive')
        object.__setattr__(self, 'coefficients', tuple(map(tuple, p.tolist())))
        object.__setattr__(self, 'duration_s', float(self.duration_s))

    def evaluate(self, u):
        u = np.asarray(u, dtype=float)
        if not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
            raise ValueError('normalized time must be finite in [0,1]')
        basis = np.stack([(1-u)**3, 3*u*(1-u)**2, 3*u*u*(1-u), u**3], axis=-1)
        return basis @ np.asarray(self.coefficients)

    def derivative(self, u, order=1, *, seconds=False):
        u = np.asarray(u, dtype=float)
        if not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
            raise ValueError('normalized time must be finite in [0,1]')
        p = np.asarray(self.coefficients)
        if order == 1:
            result = np.stack([(1-u)**2, 2*u*(1-u), u*u], axis=-1) @ (3*np.diff(p, axis=0))
        elif order == 2:
            result = np.stack([1-u, u], axis=-1) @ (6*np.diff(p, n=2, axis=0))
        else:
            raise ValueError('derivative order must be 1 or 2')
        return result / self.duration_s**order if seconds else result

    def validate_safe(self, *, central=False):
        p = np.asarray(self.coefficients)
        x0,y0,x1,y1 = CENTRAL_RECT if central else (X_BOUND_MIN,Y_BOUND_MIN,X_BOUND_MAX,Y_BOUND_MAX)
        if np.any(p < [x0,y0]) or np.any(p > [x1,y1]):
            raise ValueError('coefficient hull outside gesture bounds')
        hull = _hull(p)
        if any(_hull_touches_rect(hull, h.rect) for h in CONTROL_ENDPOINT_EXCLUSIONS):
            raise ValueError('coefficient hull touches expanded control region')

    def to_dict(self):
        return dict(schema=SCHEMA, coefficients=[list(p) for p in self.coefficients], duration_s=self.duration_s)

    @classmethod
    def from_dict(cls, value):
        if set(value) != {'schema','coefficients','duration_s'} or value['schema'] != SCHEMA:
            raise ValueError('invalid cubic schema')
        return cls(value['coefficients'], value['duration_s'])


def fit_eight_positions(positions, duration_s):
    """Endpoint-constrained least squares at eight evenly spaced normalized times.

    Fitting does not confer safety. The compiler rejects unsafe fitted hulls.
    """
    p = np.asarray(positions, dtype=float)
    if p.shape != (8,2) or not np.isfinite(p).all():
        raise ValueError('eight finite 2D positions required')
    u = np.linspace(0,1,8)
    a = np.stack([3*u*(1-u)**2, 3*u*u*(1-u)], axis=1)
    rhs = p - (1-u[:,None])**3*p[0] - u[:,None]**3*p[-1]
    inner = np.linalg.lstsq(a, rhs, rcond=None)[0]
    return CubicInTime((p[0],inner[0],inner[1],p[-1]), duration_s)


@dataclass(frozen=True)
class CompiledCurve:
    boundaries_ms: tuple[int, ...]
    points_device: tuple[tuple[int,int], ...]
    device_size: tuple[int,int]
    approximation_bound: float

    @property
    def segment_durations_ms(self):
        return tuple(b-a for a,b in zip(self.boundaries_ms,self.boundaries_ms[1:]))

    @property
    def points_normalized(self):
        return tuple((x/self.device_size[0], y/self.device_size[1]) for x,y in self.points_device)

    def to_dict(self):
        return dict(boundaries_ms=list(self.boundaries_ms), points_device=[list(p) for p in self.points_device],
                    device_size=list(self.device_size), approximation_bound=self.approximation_bound,
                    segment_durations_ms=list(self.segment_durations_ms), n=len(self.points_device)-1,
                    min_segment_duration_ms=min(self.segment_durations_ms),
                    max_segment_duration_ms=max(self.segment_durations_ms))


def compile_curve(curve, *, device_size=(414,896), n=None, max_segment_duration_ms=None,
                  min_segment_duration_ms=1):
    curve.validate_safe()
    if (n is None) == (max_segment_duration_ms is None):
        raise ValueError('provide either n or maximum segment duration')
    if len(device_size) != 2 or any(isinstance(v,bool) or not isinstance(v,int) or v <= 0 for v in device_size):
        raise ValueError('device size must contain two positive integers')
    if not math.isfinite(min_segment_duration_ms) or min_segment_duration_ms < 1:
        raise ValueError('minimum segment duration must be at least 1 ms')
    total = round(curve.duration_s*1000)
    if max_segment_duration_ms is not None:
        if not math.isfinite(max_segment_duration_ms) or max_segment_duration_ms < 1:
            raise ValueError('maximum segment duration must be at least 1 ms')
        # Integer durations cannot exceed the requested maximum.
        n = max(1, math.ceil(total / math.floor(max_segment_duration_ms)))
    if isinstance(n,bool) or not isinstance(n,int) or n < 1 or n > total:
        raise ValueError('n must produce positive millisecond segments')
    bounds = tuple(round(i*total/n) for i in range(n+1))
    durations = np.diff(bounds)
    if np.any(durations < min_segment_duration_ms):
        raise ValueError('schedule violates minimum segment duration')
    if max_segment_duration_ms is not None and np.any(durations > max_segment_duration_ms):
        raise ValueError('schedule violates maximum segment duration')
    times = np.asarray(bounds)/total
    p = curve.evaluate(times)
    q = np.rint(p*np.asarray(device_size)).astype(int)
    normalized = q/np.asarray(device_size)
    for a,b in zip(normalized,normalized[1:]):
        if not segment_is_safe(tuple(a),tuple(b)):
            raise ValueError('quantized segment touches expanded control region')
    if np.any(normalized < [X_BOUND_MIN,Y_BOUND_MIN]) or np.any(normalized > [X_BOUND_MAX,Y_BOUND_MAX]):
        raise ValueError('quantized position outside gesture bounds')
    # C'' is linear in u; its norm is convex, hence maximal at interval ends.
    interval_bounds = []
    for a,b in zip(times,times[1:]):
        m = float(np.linalg.norm(curve.derivative([a,b],order=2),axis=1).max())
        interval_bounds.append(m*(b-a)**2/8)
    quantization = float(np.linalg.norm(normalized-p,axis=1).max())
    bound = float(max(interval_bounds)+quantization)
    return CompiledCurve(bounds,tuple(map(tuple,q.tolist())),tuple(device_size),bound)


def curve_pointer(compiled):
    from trueskate_ai.sim.touch_actions import make_touch_pointer, build_curved_drag
    finger = make_touch_pointer('cubic')
    build_curved_drag(finger,compiled.points_device,segment_durations_ms=compiled.segment_durations_ms)
    return finger
