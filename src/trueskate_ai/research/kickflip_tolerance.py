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
import html
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np

from trueskate_ai.research import kickflip_repeatability as recipe
from trueskate_ai.research.curve_measurement import _decode_source_frames
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
# Review frames: just before the pop, then after the flick's encoded end, where the banner shows.
SHEET_OFFSETS_S = (.5, 1., 2., 3.)
SHEET_FRAME = (276, 597)  # A third of the 828x1792 native frame, so banner text stays legible.
SHEET_LABEL_PX, SHEET_GAP_PX = 26, 6
SHEET_SIZE = ((len(SHEET_OFFSETS_S) + 1) * (SHEET_FRAME[0] + SHEET_GAP_PX) - SHEET_GAP_PX,
              SHEET_FRAME[1] + SHEET_LABEL_PX)
VERDICTS = {'L': 'landed kickflip', 'F': 'not a landed kickflip', 'U': 'unsure'}
FAILED_STATES = ('technical failure', 'incomplete')


def base_candidate(repo):
    candidate = next(c for c in recipe.prepare_manifest(repo)['candidates'] if c['id'] == BASE_CANDIDATE)
    if candidate['sha256'] != BASE_SHA256:
        raise ValueError(f'{BASE_CANDIDATE} no longer matches the approved kickflip recipe')
    return candidate


def jitter_stroke(name, points, durations_s, distance, rng):
    """Move every point `distance` in a uniformly random direction.

    The whole stroke is redrawn while any point leaves the screen or any segment crosses a
    protected control (the recipe's own safety check), so the kept directions are uniform
    among valid strokes. The redraw count is logged.
    """
    if distance == 0:
        return recipe.direct_stroke(name, points, durations_s), dict(
            distance=0., angles_rad=None, redraws=0, base_points=points, points=points)
    for redraws in range(MAX_REDRAWS):
        angles = rng.uniform(0., 2 * math.pi, len(points))
        moved = [[x + distance * math.cos(a), y + distance * math.sin(a)] for (x, y), a in zip(points, angles)]
        try:
            stroke = recipe.direct_stroke(name, moved, durations_s)
        except ValueError:
            continue
        return stroke, dict(distance=distance, angles_rad=angles.tolist(), redraws=redraws,
                            base_points=points, points=moved)
    raise ValueError(f'{name}: no valid directions within {MAX_REDRAWS} draws at distance {distance}')


def build_plan(repo, *, seed=SEED, levels=LEVELS, per_level=PER_LEVEL, source_commit=None):
    base = base_candidate(repo)
    params = base['parameters']
    strokes = dict(pop=([params['pop_start'], params['pop_end']], [params['pop_s']]),
                   flick=(params['flick_points'], params['flick_durations_s']))
    # Blocks of one attempt per level, shuffled within each block, so drift and an early stop
    # affect every level alike.
    order_rng = np.random.default_rng([seed, 0])
    order = [float(level) for _ in range(per_level) for level in order_rng.permutation(levels)]
    push, _, _, catch = base['contacts']
    attempts = []
    for number, distance in enumerate(order, 1):
        rng = np.random.default_rng([seed, number])
        pop, pop_jitter = jitter_stroke('pop', *strokes['pop'], distance, rng)
        flick, flick_jitter = jitter_stroke('flick', *strokes['flick'], distance, rng)
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


def admit_all(root, jobs=1):
    """Admit every recorded attempt with the repeatability admission (same gates), off the rig."""
    plan = load_plan(root)
    current = states(root, plan)
    todo = [(attempt_dir(root, n), plan['wda_revision']) for n, s in current.items() if s == 'recorded']
    if jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            return dict(pool.map(_admit, todo))
    return dict(map(_admit, todo))


def contact_sheet(root, number, out_dir):
    """One JPG per attempt: the frame before the pop, then 0.5/1/2/3 s after the flick ends.

    Labelled only with the attempt number, so the review does not show the level.
    """
    trial = attempt_dir(root, number)
    admission = recipe.read_json(trial / 'admission.json')
    if not admission['accepted']:
        raise ValueError(f'attempt {number} has no admitted recording')
    pts = np.array(admission['original_pts_s'])
    gestures = {g['role']: g for g in admission['gestures']}
    flick_end = gestures['flick']['video_start_s'] + gestures['flick']['encoded_duration_s']
    targets = [('before pop', gestures['pop']['video_start_s'] - admission['native_frame_s'])]
    targets += [(f'+{offset:g} s after flick', flick_end + offset) for offset in SHEET_OFFSETS_S]
    indices = [int(np.argmin(np.abs(pts - t))) for _, t in targets]
    selected = [i for i, t in enumerate(pts) if pts[min(indices)] <= t <= pts[max(indices)]]
    frames, position = {}, iter(selected)
    def keep(image):
        index = next(position)
        if index in indices:
            frames[index] = cv2.resize(image, SHEET_FRAME, interpolation=cv2.INTER_AREA)
    _decode_source_frames(trial / 'original.mov', pts, start=pts[min(indices)], end=pts[max(indices)],
                          keep=False, inspect=keep)
    tiles = []
    for (label, _), index in zip(targets, indices):
        tile = np.full((SHEET_FRAME[1] + SHEET_LABEL_PX, SHEET_FRAME[0], 3), 255, np.uint8)
        tile[SHEET_LABEL_PX:] = frames[index]
        cv2.putText(tile, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    gap = np.full((tiles[0].shape[0], SHEET_GAP_PX, 3), 255, np.uint8)
    sheet = np.hstack([part for tile in tiles for part in (tile, gap)][:-1])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f'attempt_{number:02d}.jpg'
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise ValueError('contact sheet write failed')
    return path, [dict(label=label, target_s=float(t), source_frame=i, source_pts_s=float(pts[i]))
                  for (label, t), i in zip(targets, indices)]


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kickflip review</title>
<style>
:root {{ --bg: #fff; --fg: #111; --muted: #666; --line: #ddd; --accent: #0b62d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #111; --fg: #eee; --muted: #aaa; --line: #333; --accent: #6aa8ff; }} }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.4 -apple-system, system-ui, sans-serif;
       margin: 0 auto; max-width: 1120px; padding: 16px 16px 140px; }}
section {{ border-top: 1px solid var(--line); padding: 12px 0; }}
img {{ max-width: 100%; height: auto; display: block; margin: 6px 0; }}
label {{ margin-right: 18px; cursor: pointer; }}
.muted {{ color: var(--muted); }}
footer {{ position: fixed; left: 0; right: 0; bottom: 0; background: var(--bg); border-top: 1px solid var(--line);
         padding: 10px 16px; }}
textarea {{ width: 100%; box-sizing: border-box; font: 13px ui-monospace, monospace; background: var(--bg);
           color: var(--fg); border: 1px solid var(--line); }}
button {{ font: inherit; padding: 4px 12px; }}
</style></head><body>
<h1>Kickflip review</h1>
<p>For each attempt: did the skater land a kickflip? Use the banner and the board in the frames after the flick.
Choose <b>Unsure</b> if the frames don't settle it, and say so with the attempt number. When done, press
<b>Copy</b> and paste the codes into the chat.</p>
{sections}
<footer><div><span id="count">0</span> of {total} judged <button id="copy" type="button">Copy</button></div>
<textarea id="codes" rows="2" readonly></textarea></footer>
<script>
const codes = document.getElementById('codes');
function update() {{
  const picked = [...document.querySelectorAll('input[type=radio]:checked')]
    .map(r => r.name.slice(1) + r.value).sort();
  codes.value = picked.join(' ');
  document.getElementById('count').textContent = picked.length;
}}
document.addEventListener('change', update);
document.getElementById('copy').addEventListener('click', () => {{
  codes.select(); navigator.clipboard ? navigator.clipboard.writeText(codes.value) : document.execCommand('copy');
}});
</script></body></html>
"""


def review_page(root):
    """Contact sheets for every admitted attempt plus one static review page (no server)."""
    root = Path(root)
    plan = load_plan(root)
    out = root / 'review'
    sections, provenance, judged = [], {}, 0
    for attempt in plan['attempts']:
        number = attempt['number']
        state = attempt_state(attempt_dir(root, number))
        title = f'Attempt {number:02d}'
        if state != 'admitted':
            sections.append(f'<section><h2>{title}</h2><p class="muted">No admitted recording ({html.escape(state)}).</p></section>')
            continue
        sheet, provenance[number] = contact_sheet(root, number, out)
        choices = ''.join(f'<label><input type="radio" name="a{number:02d}" value="{code}"> {html.escape(text.capitalize())}</label>'
                          for code, text in VERDICTS.items())
        pop = next(g for g in recipe.read_json(attempt_dir(root, number) / 'admission.json')['gestures'] if g['role'] == 'pop')
        sections.append(f'<section><h2>{title}</h2><a href="{sheet.name}"><img src="{sheet.name}" alt="{title} frames" '
                        f'width="{SHEET_SIZE[0]}" height="{SHEET_SIZE[1]}"></a><div>{choices} '
                        f'<a class="muted" href="../attempts/attempt_{number:02d}/original.mov">source movie</a> '
                        f'<span class="muted">(pop at {pop["video_start_s"]:.1f} s)</span></div></section>')
        judged += 1
    page = out / 'index.html'
    page.write_text(PAGE.format(sections='\n'.join(sections), total=judged))
    (out / 'sheets.json').write_text(json.dumps(dict(experiment=EXPERIMENT, sheets=provenance), indent=2))
    return page


TOKEN = re.compile(r'(\d{1,2})\s*([LFU])', re.IGNORECASE)


def record_verdicts(root, codes, *, replace=False):
    """Operator verdicts as `07L 08F 09U`: L landed kickflip, F not, U unsure.

    Every token is checked before any verdict is written. Each verdict is tied to its
    admitted video's hash.
    """
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
        trial = attempt_dir(root, number)
        if attempt_state(trial) != 'admitted':
            raise ValueError(f'attempt {number} has no admitted recording to judge')
        if (trial / 'verdict.json').exists() and not replace:
            raise ValueError(f'attempt {number} already has a verdict; pass --replace to change it')
        checked.append((trial, number, code))
    for trial, number, code in checked:
        admission = recipe.read_json(trial / 'admission.json')
        (trial / 'verdict.json').write_text(json.dumps(dict(attempt=number, code=code, verdict=VERDICTS[code],
            video_sha256=admission['video_sha256'], recorded_at=dt.datetime.now(dt.timezone.utc).isoformat()),
            indent=2) + '\n')
    return len(checked)


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
            for stroke, colour in zip(attempt['candidate']['contacts'][1:3], ((40, 40, 220), (220, 120, 30))):
                points = [(int(x * width), int(y * height)) for x, y in stroke['points']]
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
