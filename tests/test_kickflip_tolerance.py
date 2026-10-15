"""Synthetic checks for the kickflip position-tolerance plan, run mode and review; no phone."""
import base64
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from trueskate_ai.collection.scene_settle import SettleResult
from trueskate_ai.research import kickflip_repeatability as recipe
from trueskate_ai.research import kickflip_tolerance as module
from trueskate_ai.research.curve_protocol import save_new

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def plan():
    return module.build_plan(REPO)


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / 'experiment'
    module.prepare(root, REPO, source_commit='synthetic')
    return root


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts/collection' / f'{name}.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_plan_is_deterministic_blocked_and_changes_only_pop_and_flick(plan):
    assert module.build_plan(REPO) == plan
    base = next(c for c in recipe.prepare_manifest(REPO)['candidates'] if c['id'] == module.BASE_CANDIDATE)
    assert base['sha256'] == plan['base_sha256'] == module.BASE_SHA256
    levels = [a['level'] for a in plan['attempts']]
    assert len(levels) == 40 and [a['number'] for a in plan['attempts']] == list(range(1, 41))
    # Every block of four holds each distance once, so drift or an early stop hits all alike.
    assert all(sorted(levels[i:i + 4]) == plan['levels'] for i in range(0, 40, 4))
    assert levels[:4] != sorted(levels[:4]) or levels[4:8] != sorted(levels[4:8])
    for attempt in plan['attempts']:
        candidate = recipe.verify_seal(attempt['candidate'])
        push, pop, flick, catch = candidate['contacts']
        assert (push, catch) == (base['contacts'][0], base['contacts'][3])
        assert (candidate['pop_wait_s'], candidate['catch_wait_s']) == (base['pop_wait_s'], base['catch_wait_s'])
        assert [s['name'] for s in candidate['contacts']] == ['push', 'pop', 'flick', 'catch']
        if attempt['level'] == 0:
            assert candidate['contacts'] == base['contacts']
    assert module.build_plan(REPO, seed=1)['attempts'] != plan['attempts']


def test_every_pop_and_flick_point_moves_exactly_the_distance(plan):
    for attempt in plan['attempts']:
        for key, stroke, count in zip(('pop', 'flick'), attempt['candidate']['contacts'][1:3], (2, 3)):
            jitter = attempt['candidate']['jitter'][key]
            assert len(stroke['points']) == len(jitter['base_points']) == count
            for moved, original in zip(stroke['points'], jitter['base_points']):
                assert math.dist(moved, original) == pytest.approx(attempt['level'], abs=1e-12)
            recipe.safe_scaled_path(stroke['points'])  # On screen and clear of protected controls.


class Directions:
    """Fake generator that returns the given direction sets in turn."""
    def __init__(self, *draws):
        self.draws = list(draws)
    def uniform(self, low, high, size):
        return np.array(self.draws.pop(0), float)


@pytest.mark.parametrize('points,bad', [([[.5, .82], [.5, .85]], math.pi / 2),  # Down into the bottom row.
                                        ([[.03, .6], [.2, .6]], math.pi)])          # Left off the screen.
def test_jitter_redraws_directions_that_leave_the_screen_or_hit_controls(points, bad):
    up = 3 * math.pi / 2
    stroke, jitter = module.jitter_stroke('pop', points, [.075], .05, Directions([bad, bad], [up, up]))
    assert jitter['redraws'] == 1 and jitter['angles_rad'] == [up, up]
    recipe.safe_scaled_path(stroke['points'])
    for moved, original in zip(stroke['points'], points):
        assert moved == pytest.approx([original[0], original[1] - .05])


def test_jitter_gives_up_rather_than_clipping(monkeypatch):
    monkeypatch.setattr(module, 'MAX_REDRAWS', 20)
    # Inside the reset button every direction stays inside a control.
    with pytest.raises(ValueError, match='no valid directions'):
        module.jitter_stroke('pop', [[.5, .03], [.5, .04]], [.075], .01, np.random.default_rng(0))


def test_changed_base_recipe_is_refused(monkeypatch):
    monkeypatch.setattr(module, 'BASE_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='approved kickflip recipe'):
        module.build_plan(REPO)


def record(root, number, *, error=None, live_failure=False):
    trial = module.attempt_dir(root, number)
    trial.mkdir(parents=True)
    save_new(trial / 'execution.json', dict(error=error, events=[]))
    if live_failure:
        save_new(trial / 'live-failure.json', dict(error='synthetic', no_replacements=True))
    return trial


def test_failures_stop_the_run_until_an_explicit_reason_and_are_never_rerun(experiment):
    plan = module.load_plan(experiment)
    assert [a['number'] for a in module.pending_attempts(experiment, plan)] == list(range(1, 41))
    record(experiment, 1)
    record(experiment, 2, error='recorder stop failed', live_failure=True)
    with pytest.raises(ValueError, match=r'technical failure in attempt\(s\) \[2\]'):
        module.pending_attempts(experiment, plan)
    with pytest.raises(ValueError, match='explicit reason'):
        module.pending_attempts(experiment, plan, '  ')
    pending = module.pending_attempts(experiment, plan, 'operator: retrieval failed; continue the plan')
    assert pending[0]['number'] == 3
    # Acknowledged once; a new failure needs a new reason.
    assert module.pending_attempts(experiment, plan)[0]['number'] == 3
    module.attempt_dir(experiment, 3).mkdir(parents=True)  # Reserved, never finished.
    with pytest.raises(ValueError, match=r'\[3\]'):
        module.pending_attempts(experiment, plan)
    log = [json.loads(line) for line in (experiment / 'continuations.jsonl').read_text().splitlines()]
    assert log[0]['acknowledged_failures'] == [2] and log[0]['next_attempt'] == 3


def test_attempts_run_in_plan_order_and_none_twice(experiment):
    plan = module.load_plan(experiment)
    record(experiment, 2)
    with pytest.raises(ValueError, match='plan order'):
        module.pending_attempts(experiment, plan)
    for number in [1, *range(3, 41)]:
        record(experiment, number)
    with pytest.raises(ValueError, match='every planned attempt has run'):
        module.pending_attempts(experiment, plan)
    with pytest.raises(FileExistsError):
        module.reserve_attempt(experiment, plan, 1)


@pytest.mark.parametrize('failure', [None, 'attempt', 'reconnect'])
def test_live_tolerance_run_keeps_every_guard_and_stops_without_replacement(experiment, monkeypatch, failure):
    import trueskate_ai.collection.gameplay_filter as gameplay
    import trueskate_ai.collection.scene_settle as scene
    import trueskate_ai.collection.wda_action_timing as timing
    import trueskate_ai.collection.xctest_capture as capture
    import trueskate_ai.sim.device as device
    cli = load_script('probe_kickflip_repeatability')
    plan = module.load_plan(experiment)
    env_file = experiment / 'synthetic.env'
    env_file.write_text('IPHONE_XR_UDID=synthetic\n')
    monkeypatch.setattr(cli.socket, 'gethostname', lambda: 'training-server')
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda *a, **k: 'state = running')
    monkeypatch.setattr(cli.signal, 'signal', lambda *a: None)
    runs, resets, connections, cleanup_checks = [], [], [], []
    def cleanup(udid):
        cleanup_checks.append(len(connections))
        if failure == 'reconnect' and len(cleanup_checks) == 3:
            raise RuntimeError('synthetic attachment left on the phone')
    monkeypatch.setattr(cli, 'recording_cleanup_ready', cleanup)
    class Worker:
        def __init__(self, *a, **k):
            self.driver = None
        def connect(self):
            connections.append('connect')
            self.driver = SimpleNamespace(capabilities={'udid': 'synthetic'}, session_id='synthetic-appium',
                query_app_state=lambda b: 4, execute=lambda action, payload: resets.append(payload))
        def disconnect(self):
            self.driver = None
        def _active_bundle_id(self):
            return device.BUNDLE_ID
    def http(url):
        if url.endswith('/wda/actionTiming'):
            return dict(value=dict(schema_version=1, build_revision=recipe.WDA_REVISION, enabled=False))
        if url.endswith('/appium/sessions'):
            return dict(value=[dict(id='synthetic-appium')] if len(connections) > len(runs) else [])
        if url.endswith('/status'):
            return dict(sessionId='synthetic-wda' if len(connections) > len(runs) else None)
        if url.endswith('activeAppInfo'):
            return dict(value=dict(bundleId=device.BUNDLE_ID))
        if url.endswith('screenshot'):
            return dict(value=base64.b64encode(b'synthetic screenshot').decode())
        return dict(value=None)
    monkeypatch.setattr(device, 'DeviceSession', Worker)
    monkeypatch.setattr(timing, '_http_json', http)
    monkeypatch.setattr(timing, 'WDAActionTimingCapture', lambda **kw: None)
    monkeypatch.setattr(capture, 'XCTestScreenRecorder', lambda *a, **kw: None)
    monkeypatch.setattr(scene, 'wait_for_centre_settle', lambda *a, **k: SettleResult(True, .5))
    monkeypatch.setattr(gameplay, 'is_menu_frame', lambda *a, **k: False)
    monkeypatch.setattr(gameplay, 'is_editor_frame', lambda *a, **k: False)
    monkeypatch.setattr(gameplay, '_to_rgb01', lambda *a: np.zeros((2, 2, 3), np.float32))
    def run_trial(**kw):
        runs.append(kw['candidate']['sha256'])
        assert kw['revision'] == recipe.WDA_REVISION and kw['foreground_guard'] is not None
        error = 'synthetic recorder failure' if failure == 'attempt' and len(runs) == 4 else None
        save_new(kw['out'] / 'execution.json', dict(error=error, events=[]))
        if error:
            raise RuntimeError(error)
    monkeypatch.setattr(recipe, 'run_trial', run_trial)
    monkeypatch.setattr(recipe, 'admit_trial', lambda *a: pytest.fail('admission belongs on the laptop'))
    kwargs = dict(root=experiment, candidate_id=None, repeatability=False, tolerance=True, env_file=env_file,
                  ready_note='operator confirms Workshop waypoint', settings_note='synthetic waypoint/settings',
                  appium_port=4725)
    expected = [a['candidate']['sha256'] for a in plan['attempts']]
    if failure is None:
        cli.live_batch(**kwargs)
        assert runs == expected and len(connections) == len(resets) == 40
        # Checked before the first connection and before every reconnect.
        assert cleanup_checks == list(range(40))
        assert set(module.states(experiment, plan).values()) == {'recorded'}
        assert recipe.read_json(experiment / 'context.json')['appium_port'] == 4725
        with pytest.raises(ValueError, match='every planned attempt has run'):
            cli.live_batch(**kwargs)
        return
    with pytest.raises(RuntimeError, match='synthetic'):
        cli.live_batch(**kwargs)
    if failure == 'reconnect':
        # The cleanup check failed before attempt 3 was reserved: attempts 1-2 stay good, nothing is lost.
        assert runs == expected[:2]
        assert [module.attempt_state(module.attempt_dir(experiment, n)) for n in (1, 2, 3)] == \
            ['recorded', 'recorded', 'pending']
        return
    assert runs == expected[:4]
    assert module.attempt_state(module.attempt_dir(experiment, 4)) == 'technical failure'
    assert (module.attempt_dir(experiment, 4) / 'live-failure.json').exists()
    with pytest.raises(ValueError, match='explicit reason'):
        cli.live_batch(**kwargs)
    assert len(runs) == 4
    cli.live_batch(**kwargs, after_failure_reason='operator: continue after the recorder failure')
    assert runs == expected[:4] + expected[4:]  # Attempt 4 is never rerun.


def test_changed_settings_or_port_need_a_new_experiment(experiment, monkeypatch):
    cli = load_script('probe_kickflip_repeatability')
    monkeypatch.setattr(cli.socket, 'gethostname', lambda: 'training-server')
    save_new(experiment / 'context.json', dict(settings_note='first waypoint', appium_port=4725))
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda *a, **k: pytest.fail('phone contacted'))
    with pytest.raises(ValueError, match='waypoint/settings changed'):
        cli.live_batch(experiment, candidate_id=None, repeatability=False, tolerance=True,
                       env_file=experiment / 'missing.env', ready_note='ready', settings_note='another waypoint',
                       appium_port=4725)


def test_run_holds_the_busy_lock_and_refuses_when_someone_else_does(experiment, monkeypatch):
    script = load_script('probe_kickflip_tolerance')
    lock = experiment / 'RIG-BUSY'
    seen = []
    monkeypatch.setattr(script.socket, 'gethostname', lambda: 'training-server')
    monkeypatch.setattr(script, 'load_runner', lambda: SimpleNamespace(
        live_batch=lambda *a, **k: seen.append((json.loads(lock.read_text())['out'], k['tolerance']))))
    kwargs = dict(env_file=Path('/rig/.env'), ready_note='ready', settings_note='waypoint', appium_port=4725,
                  after_failure_reason=None, lock=lock)
    script.run(experiment, **kwargs)
    assert seen == [(str(experiment), True)] and not lock.exists()
    lock.write_text(json.dumps(dict(owner='learn-replay pilot', pid=1)))
    with pytest.raises(RuntimeError, match='another recording'):
        script.run(experiment, **kwargs)
    assert len(seen) == 1 and json.loads(lock.read_text())['pid'] == 1  # Someone else's lock is left alone.
    monkeypatch.setattr(script.socket, 'gethostname', lambda: 'laptop')
    lock.unlink()
    with pytest.raises(ValueError, match='training-server'):
        script.run(experiment, **kwargs)
    assert not lock.exists()


def admitted(root, number):
    trial = module.attempt_dir(root, number)
    if not trial.exists():
        record(root, number)
    (trial / 'original.mov').write_bytes(f'synthetic {number}'.encode())
    pts = (np.arange(1200) / 60).tolist()
    gestures = [dict(role=r, video_start_s=t, encoded_duration_s=d) for r, t, d in
                (('push', 14., .13), ('pop', 14.6, .075), ('flick', 15.0, .1), ('catch', 15.6, .52))]
    save_new(trial / 'admission.json', dict(accepted=True, video_sha256=recipe.file_sha(trial / 'original.mov'),
             original_pts_s=pts, native_frame_s=1 / 60, gestures=gestures))
    return trial


def test_verdicts_are_checked_together_and_bound_to_the_video(experiment):
    for number in (1, 2, 3):
        admitted(experiment, number)
    record(experiment, 4, error='synthetic', live_failure=True)
    for bad, message in (('01X', 'codes look like'), ('1L 1F', 'twice'), ('41L', 'not in the plan'),
                         ('01L 04F', 'no admitted recording'), ('01L and 02F', 'codes look like'), ('', 'codes look like')):
        with pytest.raises(ValueError, match=message):
            module.record_verdicts(experiment, bad)
        assert not (module.attempt_dir(experiment, 1) / 'verdict.json').exists()
    assert module.record_verdicts(experiment, '01L, 2f 03u') == 3
    verdict = recipe.read_json(module.attempt_dir(experiment, 2) / 'verdict.json')
    assert verdict['code'] == 'F' and verdict['verdict'] == 'not a landed kickflip'
    with pytest.raises(ValueError, match='already has a verdict'):
        module.record_verdicts(experiment, '02L')
    module.record_verdicts(experiment, '02L', replace=True)
    summary = module.summarize(experiment)
    plan = module.load_plan(experiment)
    level_of = {a['number']: f"{a['level']:g}" for a in plan['attempts']}
    total = {k: sum(v[k] for v in summary['levels'].values()) for k in ('planned', 'run', 'technical_failures', 'L', 'F', 'U')}
    assert total == dict(planned=40, run=4, technical_failures=1, L=2, F=0, U=1)
    assert summary['levels'][level_of[1]]['L'] >= 1
    (module.attempt_dir(experiment, 1) / 'original.mov').write_bytes(b'another video')
    admission = recipe.read_json(module.attempt_dir(experiment, 1) / 'admission.json')
    admission['video_sha256'] = recipe.file_sha(module.attempt_dir(experiment, 1) / 'original.mov')
    (module.attempt_dir(experiment, 1) / 'admission.json').write_text(json.dumps(admission))
    with pytest.raises(ValueError, match='another video'):
        module.summarize(experiment)
    assert '| Distance |' in module.summary_markdown(summary)


def test_review_page_shows_attempt_numbers_but_not_distances(experiment, monkeypatch):
    for number in (1, 2):
        admitted(experiment, number)
    record(experiment, 3, error='synthetic', live_failure=True)
    decoded = []
    def decode(video, pts, *, start, end, keep, inspect):
        selected = [i for i, t in enumerate(pts) if start <= t <= end]
        decoded.append((selected[0], selected[-1]))
        for i in selected:
            inspect(np.full((1792, 828, 3), i % 255, np.uint8))
        return [], [], selected
    monkeypatch.setattr(module, '_decode_source_frames', decode)
    page = module.review_page(experiment)
    text = page.read_text()
    assert 'Attempt 01' in text and 'Attempt 02' in text and 'technical failure' in text
    assert 'name="a01"' in text and 'value="L"' in text
    for word in ('level', 'distance', '0.02', '0.05', '0.1'):
        assert word not in text
    sheet = cv2.imread(str(page.parent / 'attempt_01.jpg'))
    assert sheet.shape[:2] == (474, 1059)
    labels = [f['label'] for f in recipe.read_json(page.parent / 'sheets.json')['sheets']['1']]
    assert labels[0] == 'before pop' and len(labels) == 5
    # Before the pop (14.6 s) through 3 s after the flick's end (15.1 s): one decode per attempt.
    assert decoded[0] == (round(14.6 * 60) - 1, round(18.1 * 60))


def test_admit_all_uses_the_plan_revision_for_recorded_attempts_only(experiment, monkeypatch):
    record(experiment, 1)
    record(experiment, 2, error='synthetic', live_failure=True)
    calls = []
    def admit(trial, revision):
        calls.append((Path(trial).name, revision))
        if Path(trial).name == 'attempt_03':
            raise ValueError('held-out middle calibration failed')
        return dict(heldout_middle_error_s=.004)
    monkeypatch.setattr(recipe, 'admit_trial', admit)
    record(experiment, 3)
    results = module.admit_all(experiment, jobs=1)
    assert calls == [('attempt_01', recipe.WDA_REVISION), ('attempt_03', recipe.WDA_REVISION)]
    assert results['attempt_01']['admitted'] and not results['attempt_03']['admitted']
