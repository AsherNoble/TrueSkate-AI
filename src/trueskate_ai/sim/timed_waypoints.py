"""Explicit normalized positions and integer millisecond timestamps; linear intervals."""
from dataclasses import dataclass
import math
from trueskate_ai.data.control_hitboxes import segment_is_safe

@dataclass(frozen=True)
class TimedWaypoints:
    points: tuple[tuple[float, float], ...]
    times_ms: tuple[int, ...]

    def __post_init__(self):
        if not 2 <= len(self.points) <= 15 or len(self.points) != len(self.times_ms):
            raise ValueError('2–15 matching timed waypoints required')
        if self.times_ms[0] != 0 or any(type(t) is not int for t in self.times_ms):
            raise ValueError('integer timestamps starting at zero required')
        if any(b <= a for a,b in zip(self.times_ms,self.times_ms[1:])):
            raise ValueError('timestamps must increase strictly')
        if any(len(p)!=2 or any(not math.isfinite(v) or not 0<=v<=1 for v in p) for p in self.points):
            raise ValueError('finite normalized positions required')

    def quantized(self, width=414, height=896):
        return tuple((int(x*width),int(y*height)) for x,y in self.points)

    def payload(self, width=414, height=896):
        points=self.quantized(width,height)
        normalized=[(x/width,y/height) for x,y in points]
        if any(not segment_is_safe(a,b) for a,b in zip(normalized,normalized[1:])):
            raise ValueError('quantized path intersects protected controls')
        moves=[dict(type='pointerMove',duration=b-a,x=p[0],y=p[1],origin='viewport')
               for p,a,b in zip(points[1:],self.times_ms,self.times_ms[1:])]
        return {'actions':[dict(type='pointer',id='timed_path',parameters={'pointerType':'touch'},actions=[
            dict(type='pointerMove',duration=0,x=points[0][0],y=points[0][1],origin='viewport'),
            dict(type='pointerDown',button=0),*moves,dict(type='pointerUp',button=0)])]}

    def position(self, time_ms, *, quantized=True):
        if not 0<=time_ms<self.times_ms[-1]:return None
        points=self.quantized() if quantized else self.points
        for i,end in enumerate(self.times_ms[1:]):
            if time_ms<=end:
                u=(time_ms-self.times_ms[i])/(end-self.times_ms[i])
                return tuple(a+(b-a)*u for a,b in zip(points[i],points[i+1]))
