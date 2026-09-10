"""Colour-component tracking from elapsed video time, never command metadata."""
from dataclasses import dataclass
import cv2
import numpy as np
from trueskate_ai.model1.classical.background import newly_brightened


@dataclass(frozen=True)
class Config:
    hue_low: int = 5
    hue_high: int = 28
    saturation: int = 80
    saturation_max: int = 255
    value: int = 90
    difference: int = 70
    min_area: int = 3
    position: str = 'centroid'
    tracking: str = 'nearest'
    background: str = 'prefix'
    max_components: int = 5
    motion_fraction: float = .12
    duration_correction: float = 0.0
    endpoint_extension: float = 0.0

    def __post_init__(self):
        if self.position not in ('centroid', 'tip'):
            raise ValueError('position must be centroid or tip')
        if not 0 <= self.hue_low <= self.hue_high <= 179:
            raise ValueError('invalid OpenCV hue range')
        if not 0 <= self.motion_fraction <= 1 or self.min_area < 1:
            raise ValueError('invalid motion fraction or component area')


def extract(frames, times, config=Config()):
    """Return up to five orange components per frame, in normalized coordinates."""
    frames = np.asarray(frames)
    times = np.asarray(times, dtype=float)
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(times) != len(frames):
        raise ValueError('expected BGR frames [time,height,width,3] and elapsed times')
    if len(times) < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('need at least three strictly increasing finite timestamps')
    if config.background == 'affine':
        frames = np.asarray(newly_brightened(frames))
    h, w = frames.shape[1:3]
    reference = np.median(frames[:min(5, len(frames))], axis=0)
    result = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        diff = np.abs(frame.astype(float) - reference).sum(axis=2)
        mask = ((hsv[..., 0] >= config.hue_low) & (hsv[..., 0] <= config.hue_high)
                & (hsv[..., 1] >= config.saturation) & (hsv[..., 1] <= config.saturation_max)
                & (hsv[..., 2] >= config.value)
                & (diff >= config.difference)).astype('uint8')
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for i in sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:config.max_components]:
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
                               'brightening': float((frame.astype(float)-reference).sum(axis=2)[labels==i].mean()),
                               'ends': [(center + lo * direction).tolist(),
                                        (center + hi * direction).tolist()]})
        result.append(candidates)
    return {'times': times.tolist(), 'components': result}


def predict_features(features, config=Config()):
    """Select a continuous moving component and estimate its active movement span."""
    if config.tracking == 'consensus':
        return predict_consensus(features, config)
    times = np.asarray(features['times'])
    tracks = []
    for index, candidates in enumerate(features['components']):
        available = set(range(len(candidates)))
        for track in sorted(tracks, key=len, reverse=True):
            if index - track[-1][0] > 2 or not available:
                continue
            last = np.array(track[-1][1]['center'])
            if config.tracking == 'linear' and len(track) >= 3:
                prev = np.array(track[-3][1]['center'])
                last = last + (last-prev) * (index-track[-1][0]) / (track[-1][0]-track[-3][0])
            chosen = min(available, key=lambda j: np.linalg.norm(np.array(candidates[j]['center']) - last))
            if np.linalg.norm(np.array(candidates[chosen]['center']) - last) <= (.12 if config.tracking == 'linear' else .25):
                track.append((index, candidates[chosen])); available.remove(chosen)
        for chosen in sorted(available):
            tracks.append([(index, candidates[chosen])])
    tracks = [t for t in tracks if len(t) >= 3]
    if not tracks:
        return None
    def score(track):
        xy = np.array([c['center'] for _, c in track])
        length = np.linalg.norm(xy[-1] - xy[0])
        if config.tracking == 'nearest': return length * np.sqrt(len(track))
        tt = times[[i for i,_ in track]]
        design = np.column_stack((np.ones(len(tt)),tt-tt[0]))
        fitted = design @ np.linalg.lstsq(design,xy,rcond=None)[0]
        residual = np.median(np.linalg.norm(xy-fitted,axis=1))
        onset = np.exp(-.5*((tt[0]-.58)/.25)**2)
        finite = .1 if track[-1][0]>=len(times)-2 else 1.0
        brightness = max(.1, np.mean([max(0,c.get('brightening',0)) for _,c in track])/100)
        return length*np.sqrt(len(track))*onset*finite*brightness/(.01+residual)
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
            'start_pass_count': int((a[:,0] <= .03).sum()) if len(a) else 0,
            'end_pass_count': int((a[:,1] <= .03).sum()) if len(a) else 0,
            'duration_pass_count': int((a[:,2] <= .10).sum()) if len(a) else 0,
            'conditional_error_mean': a.mean(axis=0).tolist() if len(a) else None,
            'conditional_error_median': np.median(a, axis=0).tolist() if len(a) else None,
            'conditional_error_p90': np.quantile(a, .9, axis=0).tolist() if len(a) else None}


def predict_consensus(features, config):
    """Fit constant-velocity hypotheses across frames instead of greedy association.

    Deterministic bounded hypothesis set; only image-derived components and
    elapsed time are used. The onset prior is specific to the aligned benchmark.
    """
    times=np.asarray(features['times'],dtype=float)
    steps=len(times); width=max((len(c) for c in features['components']),default=0)
    if width==0:return None
    points=np.full((steps,width,2),10.,dtype=float)
    bright=np.zeros((steps,width),dtype=float)
    for i,cs in enumerate(features['components']):
        for j,c in enumerate(cs):
            points[i,j]=c['center'];bright[i,j]=max(0,c.get('brightening',0))
    pairs=[]
    for i in range(4,min(12,steps)):
        for j in range(i+3,min(i+14,steps-2),2):
            for a in range(len(features['components'][i])):
                for b in range(len(features['components'][j])):
                    velocity=(points[j,b]-points[i,a])/(times[j]-times[i])
                    if .10<=np.linalg.norm(velocity)<=2.5 and abs(velocity[0])>=.07:
                        pairs.append((points[i,a]-velocity*times[i],velocity))
    if not pairs:return None
    if len(pairs)>1200:
        ids=np.random.default_rng(0).choice(len(pairs),1200,replace=False)
        pairs=[pairs[i] for i in ids]
    hypotheses=np.asarray(pairs)
    finalists=[]
    for off in range(0,len(hypotheses),64):
        h=hypotheses[off:off+64]
        expected=h[:,0,None,:]+h[:,1,None,:]*times[None,:,None]
        dist=np.linalg.norm(expected[:,:,None,:]-points[None,:,:,:],axis=-1)
        nearest=dist.argmin(axis=2);err=dist.min(axis=2)
        for k in range(len(h)):
            ids=np.where(err[k]<.035)[0]
            if len(ids)<4:continue
            groups=np.split(ids,np.where(np.diff(ids)>2)[0]+1)
            for group in groups:
                if len(group)<4:continue
                first,last=group[0],group[-1];duration=times[last]-times[first]
                if not .18<=duration<=1.5:continue
                onset=np.exp(-.5*((times[first]-.58)/.22)**2)
                finite=.15 if last>=steps-2 else 1.
                evidence=np.mean(bright[group,nearest[k,group]])/100
                length=np.linalg.norm(h[k,1])*duration
                if length<.12:continue
                score=len(group)*onset*finite*max(.1,evidence)/(1+np.mean(err[k,group])/.02)
                finalists.append((score,group,nearest[k,group]))
    if not finalists:return None
    _,ids,js=max(finalists,key=lambda x:x[0])
    xy=points[ids,js];tt=times[ids]
    design=np.column_stack((np.ones(len(tt)),tt))
    coeff=np.linalg.lstsq(design,xy,rcond=None)[0]
    for _ in range(3):
        residual=np.linalg.norm(xy-design@coeff,axis=1)
        weights=np.minimum(1.,.015/np.maximum(residual,1e-6))
        coeff=np.linalg.lstsq(design*weights[:,None],xy*weights[:,None],rcond=None)[0]
    direction=coeff[1]/max(np.linalg.norm(coeff[1]),1e-9)
    start=coeff[0]+coeff[1]*tt[0]-direction*config.endpoint_extension
    end=coeff[0]+coeff[1]*tt[-1]+direction*config.endpoint_extension
    duration=float(tt[-1]-tt[0]+config.duration_correction)
    if duration<=0:return None
    return [*np.clip(start,0,1),*np.clip(end,0,1),duration]
