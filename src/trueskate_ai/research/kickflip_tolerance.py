"""Kickflip position tolerance (explore mode): how far can the pop and flick land from the
approved recipe and the kickflip still land?

Every point of `candidate_19`'s pop and flick (KICKFLIP-REPEATABILITY-20261009, 19/19 landed)
moves a fixed distance in a random direction. Distance is Euclidean in normalized screen
coordinates (x/414, y/896), the metric of Model 1's 0.03 recovery tolerance
(`model1/certification.py`, `model1/linear/training.py`). Push and catch are unchanged.
Attempts run through the repeatability runner's execution path and live guards; only its
approval/review gates and setup cap are skipped. Never training data.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np

from trueskate_ai.research import kickflip_repeatability as recipe
from trueskate_ai.research.curve_protocol import save_new

EXPERIMENT = 'KICKFLIP-TOLERANCE-20261015'
BASE_CANDIDATE = 'candidate_19'
# The approved recipe's sealed hash; a changed recipe is a different experiment.
BASE_SHA256 = '430175177b3eda537ee1e714c75be5e1db635c02ed67434202ec72a8eeec1dc4'
LEVELS = (0., .02, .05, .1)
PER_LEVEL = 10
SEED = 20261015
MAX_REDRAWS = 1000
METRIC = 'Euclidean distance in normalized screen coordinates (x/414, y/896), as Model 1 recovery'
# Operator review: kickflip_review.py serves each attempt's native frames (the review UI).
VERDICTS = {'L': 'landed kickflip', 'F': 'not a landed kickflip', 'U': 'unsure'}
FAILED_STATES = ('technical failure', 'incomplete')


def base_candidate(repo):
    candidate = next(c for c in recipe.prepare_manifest(repo)['candidates'] if c['id'] == BASE_CANDIDATE)
    if candidate['sha256'] != BASE_SHA256:
        raise ValueError(f'{BASE_CANDIDATE} no longer matches the approved kickflip recipe')
    return candidate


def jitter_gesture(name, points, durations_s, distance, rng):
    """Move every point `distance` in a uniformly random direction.

    The whole gesture is redrawn while any point leaves the screen or any segment crosses a
    protected control (the recipe's own safety check), so the kept directions are uniform
    among valid gestures. The redraw count is logged.
    """
    if distance == 0:
        return recipe.direct_gesture(name, points, durations_s), dict(
            distance=0., angles_rad=None, redraws=0, base_points=points, points=points)
    for redraws in range(MAX_REDRAWS):
        angles = rng.uniform(0., 2 * math.pi, len(points))
        moved = [[x + distance * math.cos(a), y + distance * math.sin(a)] for (x, y), a in zip(points, angles)]
        try:
            gesture = recipe.direct_gesture(name, moved, durations_s)
        except ValueError:
            continue
        return gesture, dict(distance=distance, angles_rad=angles.tolist(), redraws=redraws,
                            base_points=points, points=moved)
    raise ValueError(f'{name}: no valid directions within {MAX_REDRAWS} draws at distance {distance}')


def build_plan(repo, *, seed=SEED, levels=LEVELS, per_level=PER_LEVEL, source_commit=None):
    base = base_candidate(repo)
    params = base['parameters']
    gestures = dict(pop=([params['pop_start'], params['pop_end']], [params['pop_s']]),
                   flick=(params['flick_points'], params['flick_durations_s']))
    # Blocks of one attempt per level, shuffled within each block, so drift and an early stop
    # affect every level alike.
    order_rng = np.random.default_rng([seed, 0])
    order = [float(level) for _ in range(per_level) for level in order_rng.permutation(levels)]
    push, _, _, catch = base['contacts']
    attempts = []
    for number, distance in enumerate(order, 1):
        rng = np.random.default_rng([seed, number])
        pop, pop_jitter = jitter_gesture('pop', *gestures['pop'], distance, rng)
        flick, flick_jitter = jitter_gesture('flick', *gestures['flick'], distance, rng)
        candidate = recipe.seal(dict(id=f'attempt_{number:02d}', experiment=EXPERIMENT, level=distance,
            contacts=[push, pop, flick, catch], pop_wait_s=base['pop_wait_s'], catch_wait_s=base['catch_wait_s'],
            base_candidate=BASE_CANDIDATE, base_sha256=base['sha256'], jitter=dict(pop=pop_jitter, flick=flick_jitter),
            training_admission=False))
        attempts.append(dict(number=number, level=distance, candidate=candidate))
    return dict(experiment=EXPERIMENT, base_candidate=BASE_CANDIDATE, base_sha256=base['sha256'], metric=METRIC,
                levels=[float(m) for m in levels], per_level=per_level, seed=seed,
                order='blocks of one attempt per level, shuffled within each block',
                jittered=['pop', 'flick'], unchanged=['push', 'catch'], source_commit=source_commit,
                wda_revision=recipe.WDA_REVISION, size=list(recipe.SIZE), fps=recipe.FPS,
                attempts=attempts, training_admission=False)


def prepare(root, repo, source_commit=None):
    root = Path(root)
    plan = build_plan(repo, source_commit=source_commit)
    root.mkdir(parents=True, exist_ok=False)
    save_new(root / 'plan.json', plan)
    return plan


def load_plan(root):
    plan = recipe.read_json(Path(root) / 'plan.json')
    if plan.get('experiment') != EXPERIMENT:
        raise ValueError('not a kickflip tolerance experiment')
    for attempt in plan['attempts']:
        recipe.verify_seal(attempt['candidate'])
    return plan


def attempt_dir(root, number):
    return Path(root) / 'attempts' / f'attempt_{number:02d}'


def attempt_state(trial):
    trial = Path(trial)
    if not trial.exists():
        return 'pending'
    if (trial / 'live-failure.json').exists():
        return 'technical failure'
    if not (trial / 'execution.json').exists():
        return 'incomplete'  # Reserved, never finished: a crash, or the attempt running now.
    if recipe.read_json(trial / 'execution.json').get('error'):
        return 'technical failure'
    if not (trial / 'admission.json').exists():
        return 'recorded'
    return 'admitted' if recipe.read_json(trial / 'admission.json')['accepted'] else 'admission failed'


def states(root, plan):
    return {a['number']: attempt_state(attempt_dir(root, a['number'])) for a in plan['attempts']}


def pending_attempts(root, plan, after_failure_reason=None):
    """Attempts still to run, in plan order.

    A technical failure stops the run. Continuing needs an explicit reason that names the
    failure; a failed attempt is never rerun or replaced.
    """
    root = Path(root)
    current = states(root, plan)
    pending = [a for a in plan['attempts'] if current[a['number']] == 'pending']
    if not pending:
        raise ValueError('every planned attempt has run; nothing is rerun or replaced')
    if any(current[a['number']] != 'pending' for a in plan['attempts'] if a['number'] > pending[0]['number']):
        raise ValueError('attempts must run in plan order')
    log = root / 'continuations.jsonl'
    acknowledged = set()
    if log.exists():
        for line in log.read_text().splitlines():
            acknowledged.update(json.loads(line)['acknowledged_failures'])
    failed = sorted(n for n, s in current.items() if s in FAILED_STATES)
    unacknowledged = [n for n in failed if n not in acknowledged]
    if unacknowledged:
        if not (after_failure_reason or '').strip():
            raise ValueError(f'technical failure in attempt(s) {unacknowledged}; continuing needs an explicit '
                             'reason, and failed attempts are never rerun')
        with log.open('a') as fh:
            fh.write(json.dumps(dict(acknowledged_failures=failed, reason=after_failure_reason,
                                     next_attempt=pending[0]['number'],
                                     at=dt.datetime.now(dt.timezone.utc).isoformat())) + '\n')
    return pending


def reserve_attempt(root, plan, number):
    attempt = next(a for a in plan['attempts'] if a['number'] == number)
    candidate = recipe.verify_seal(attempt['candidate'])
    trial = attempt_dir(root, number)
    trial.mkdir(parents=True, exist_ok=False)  # A reservation counts even if the start fails.
    save_new(trial / 'reservation.json', dict(attempt=number, candidate_sha256=candidate['sha256']))
    return trial, candidate


def _admit(args):
    trial, revision = args
    try:
        result = recipe.admit_trial(trial, revision)
        return Path(trial).name, dict(admitted=True, heldout_middle_error_s=result['heldout_middle_error_s'])
    except Exception as exc:
        return Path(trial).name, dict(admitted=False, error=f'{type(exc).__name__}: {exc}')


def admit_all(root, jobs=1, retry_failed=False):
    """Admit every recorded attempt with the repeatability admission (same gates), off the rig.

    `retry_failed` reruns the same gates on the same video after a laptop-side fault (say a
    full disk); the earlier result is kept beside it. A genuine rejection cannot change.
    """
    plan = load_plan(root)
    current = states(root, plan)
    if retry_failed:
        for number in (n for n, s in current.items() if s == 'admission failed'):
            trial = attempt_dir(root, number)
            kept = len(list(trial.glob('admission-failed-*.json'))) + 1
            (trial / 'admission.json').rename(trial / f'admission-failed-{kept}.json')
            current[number] = 'recorded'
    todo = [(attempt_dir(root, n), plan['wda_revision']) for n, s in current.items() if s == 'recorded']
    if jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            return dict(pool.map(_admit, todo))
    return dict(map(_admit, todo))


TOKEN = re.compile(r'(\d{1,2})\s*([LFU])', re.IGNORECASE)


def write_verdicts(root, checked, *, replace=False, source='codes'):
    """Write checked (attempt, code, note) rows, each tied to its admitted video's hash.

    Every row is checked before any is written, and an existing verdict is never silently
    replaced.
    """
    root = Path(root)
    for number, _, _ in checked:
        trial = attempt_dir(root, number)
        if attempt_state(trial) != 'admitted':
            raise ValueError(f'attempt {number} has no admitted recording to judge')
        if (trial / 'verdict.json').exists() and not replace:
            raise ValueError(f'attempt {number} already has a verdict; pass --replace to change it')
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for number, code, note in checked:
        trial = attempt_dir(root, number)
        admission = recipe.read_json(trial / 'admission.json')
        (trial / 'verdict.json').write_text(json.dumps(dict(attempt=number, code=code, verdict=VERDICTS[code],
            note=note, source=source, video_sha256=admission['video_sha256'], recorded_at=now), indent=2) + '\n')
    return len(checked)


def record_verdicts(root, codes, *, replace=False):
    """Operator verdicts as `07L 08F 09U`: L landed kickflip, F not, U unsure (when the review page is unavailable)."""
    root = Path(root)
    plan = load_plan(root)
    numbers = {a['number'] for a in plan['attempts']}
    tokens = TOKEN.findall(codes)
    if not tokens or TOKEN.sub('', codes).replace(',', ' ').split():
        raise ValueError('codes look like "07L 08F 09U" (L landed kickflip, F not, U unsure)')
    checked, seen = [], set()
    for number, code in tokens:
        number, code = int(number), code.upper()
        if number not in numbers:
            raise ValueError(f'attempt {number} is not in the plan')
        if number in seen:
            raise ValueError(f'attempt {number} is listed twice')
        seen.add(number)
        checked.append((number, code, ''))
    return write_verdicts(root, checked, replace=replace)


def summarize(root):
    """Per-level outcome counts. Landing rate is landed / (landed + not landed); unsure stays out."""
    root = Path(root)
    plan = load_plan(root)
    rows, levels = [], {m: dict(planned=0, run=0, technical_failures=0, admitted=0, L=0, F=0, U=0, unjudged=0)
                        for m in plan['levels']}
    for attempt in plan['attempts']:
        trial = attempt_dir(root, attempt['number'])
        state, level = attempt_state(trial), levels[attempt['level']]
        level['planned'] += 1
        code = None
        if state != 'pending':
            level['run'] += 1
        if state in (*FAILED_STATES, 'admission failed'):
            level['technical_failures'] += 1
        if state == 'admitted':
            level['admitted'] += 1
            if (trial / 'verdict.json').exists():
                verdict = recipe.read_json(trial / 'verdict.json')
                if verdict['video_sha256'] != recipe.read_json(trial / 'admission.json')['video_sha256']:
                    raise ValueError(f"verdict for attempt {attempt['number']} belongs to another video")
                code = verdict['code']
                level[code] += 1
            else:
                level['unjudged'] += 1
        rows.append(dict(attempt=attempt['number'], level=attempt['level'], state=state, verdict=code))
    for level in levels.values():
        judged = level['L'] + level['F']
        level['landing_rate'] = level['L'] / judged if judged else None
    return dict(experiment=EXPERIMENT, metric=METRIC, levels={f'{m:g}': v for m, v in levels.items()}, attempts=rows)


def summary_markdown(summary):
    lines = [f"# {summary['experiment']}", '', f"Distance: {summary['metric']}.", '',
             '| Distance | Planned | Run | Technical failures | Landed kickflip | Not | Unsure | Unjudged | Landing rate |',
             '|---|---|---|---|---|---|---|---|---|']
    for m, v in summary['levels'].items():
        rate = '–' if v['landing_rate'] is None else f"{v['L']}/{v['L'] + v['F']} ({v['landing_rate']:.0%})"
        lines.append(f"| {m} | {v['planned']} | {v['run']} | {v['technical_failures']} | {v['L']} | {v['F']} | "
                     f"{v['U']} | {v['unjudged']} | {rate} |")
    return '\n'.join(lines) + '\n'


def render_paths(plan, out_dir):
    """Base and jittered pop/flick paths on a blank 414x896 canvas, one PNG per level (offline check)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = recipe.SIZE
    paths = []
    for level in plan['levels']:
        canvas = np.full((height, width, 3), 255, np.uint8)
        for attempt in (a for a in plan['attempts'] if a['level'] == level):
            for gesture, colour in zip(attempt['candidate']['contacts'][1:3], ((40, 40, 220), (220, 120, 30))):
                points = [(int(x * width), int(y * height)) for x, y in gesture['points']]
                cv2.polylines(canvas, [np.array(points, np.int32)], False, colour, 1, cv2.LINE_AA)
                cv2.circle(canvas, points[0], 3, colour, -1, cv2.LINE_AA)
        base = plan['attempts'][0]['candidate']['jitter']
        for key in ('pop', 'flick'):
            points = [(int(x * width), int(y * height)) for x, y in base[key]['base_points']]
            cv2.polylines(canvas, [np.array(points, np.int32)], False, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f'distance {level:g} (pop red, flick blue, base black)', (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
        path = out_dir / f'paths_{level:g}.png'
        cv2.imwrite(str(path), canvas)
        paths.append(path)
    return paths
