"""Native 60 Hz review and conservative scoring for the preloaded USB pilot."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import importlib.util

import cv2
import numpy as np

from trueskate_ai.research.audit_video import _decode_source_frames
from trueskate_ai.research.review_provenance import bundle, file_sha256, verify_bundle
from trueskate_ai.research import pico_curve_pilot as pilot

GAIN_WINDOW = (.125, .275)  # middle of each 400 ms plateau; tolerates coarse clock drift


def native_pts(movie):
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_streams', '-show_entries',
        'frame=best_effort_timestamp_time:stream=width,height', '-of', 'json', str(movie)]))
    if (probe['streams'][0]['width'], probe['streams'][0]['height']) != (828, 1792):
        raise ValueError('XR native 828x1792 movie required')
    pts = [float(f['best_effort_timestamp_time']) for f in probe['frames']]
    if len(pts) < 2 or not all(math.isfinite(t) for t in pts) or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError('invalid native PTS')
    if abs(float(np.median(np.diff(pts)))-1/60) > .003:
        raise ValueError('native 60 Hz capture required')
    return pts


def capture_proof(report, program, movie, pts):
    if report.get('error') or report.get('cleanup_errors') or report.get('pico_program_sha256') != program['sha256']:
        raise ValueError('successful matching capture and cleanup required')
    if report.get('park') != 'The Workshop' or report.get('park_provenance') != 'operator readiness prompt':
        raise ValueError('operator-attested Workshop provenance required')
    cases = report['cases']
    if len(cases) != 1 or cases[0]['lifecycle'] != 'passed' or cases[0].get('analysis_error'):
        raise ValueError('one successful isolated capture required')
    video = cases[0]['video']
    if (cases[0]['fps_requested'] != 60 or video['duration_s'] < 59 or
        video['effective_fps'] < 58.8 or video['max_gap_s'] > .100001 or
        video['frames'] != len(pts) or video['sha256'] != file_sha256(movie)):
        raise ValueError('capture admission or movie identity failed')
    return report


def load_export(private, exported, media_root):
    identity = verify_bundle(private, exported, media_root)
    proof = identity['provenance']
    pilot.validate_program(proof['program'])
    pilot.verify_receipts(proof['receipts'], proof['program'])
    # Private absolute source path is outside the web root. Re-probe the original
    # before importing marks, so stale/resealed PTS or capture metadata fail.
    movie = Path(proof['movie_path'])
    pts = native_pts(movie)
    if pts != proof['source_pts_s'] or file_sha256(movie) != proof['movie_sha256']:
        raise ValueError('original source pixels/PTS changed')
    capture_proof(proof['capture_report'], proof['program'], movie, pts)
    rows = exported.get('annotations', {})
    ids = {clip['id'] for clip in identity['clips']}
    if set(rows) != ids:
        raise ValueError('one annotation row per displayed clip required (unknowns allowed)')
    for clip in identity['clips']:
        frames = {f['source_frame'] for f in clip['frames']}
        row = rows[clip['id']]
        for mark in row.get('frames', []):
            if mark.get('source_frame') not in frames:
                raise ValueError('annotation frame was not displayed in this clip')
        for key in ('first_contact_frame', 'first_lifted_frame'):
            if row.get(key) is not None and row[key] not in frames:
                raise ValueError('boundary frame was not displayed')
            if row.get(key) is not None and row[key]-1 not in frames:
                raise ValueError('previous boundary frame was not displayed')
    return identity, rows


def control_fit(private, exported, media_root):
    identity, rows = load_export(private, exported, media_root)
    if identity['kind'] != 'pico-usb-controls-v1':
        raise ValueError('Pico control review required')
    proof = identity['provenance']
    controls = {clip['role']: dict(rows[clip['id']], confirmed=rows[clip['id']].get('boundary_confirmed') is True)
                for clip in identity['clips']}
    fit = pilot.fit_timebase(proof['program'], proof['receipts'], controls, proof['source_pts_s'])
    return fit, dict(bundle_sha256=private['bundle_sha256'], export_sha256=pilot.digest(exported),
                     annotations=rows, fit=fit)


def static_position(clip, annotation):
    if annotation.get('static_confirmed') is not True:
        raise ValueError('every stationary window must be reviewed')
    frames = annotation.get('frames', [])
    if len({f['source_frame'] for f in frames}) != len(frames):
        raise ValueError('duplicate static frame')
    marks = [f for f in frames if f.get('point_pt') is not None]
    if len(marks) < 3 or len(marks)/len(clip['frames']) < .6:
        raise ValueError('inadequate stationary tracking')
    positions = np.asarray([f['point_pt'] for f in marks], dtype=float)
    if positions.shape != (len(marks), 2) or not np.isfinite(positions).all():
        raise ValueError('invalid static cursor position')
    uncertainties = [f.get('uncertainty_pt', math.inf) for f in marks]
    centre = np.median(positions, axis=0)
    radius = float(np.linalg.norm(positions-centre, axis=1).max())
    if radius > 2 or max(uncertainties) > 2:
        raise ValueError('stationary position too uncertain')
    return centre.tolist(), max(radius, max(uncertainties))


def gain_fit(private, exported, media_root):
    identity, rows = load_export(private, exported, media_root)
    if identity['kind'] != 'pico-usb-gain-review-v1':
        raise ValueError('gain review required')
    proof = identity['provenance']
    program, receipt = proof['program'], proof['receipts']
    pilot.verify_receipts(receipt, program)
    clips = {c['id']: c for c in identity['clips']}
    passes = []
    for command in program['commands']:
        a, b = command['id']+'-before', command['id']+'-after'
        before, ua = static_position(clips[a], rows[a])
        after, ub = static_position(clips[b], rows[b])
        passes.append(dict(id=command['id'], before_pt=before, after_pt=after,
                           uncertainty_pt=max(ua, ub), confirmed=True, stable=True))
    parks = [static_position(clips[f'park-{i}'], rows[f'park-{i}'])[0] for i in range(2)]
    observations = dict(program_sha256=program['sha256'], passes=passes, park_points_pt=parks,
                        bundle_sha256=private['bundle_sha256'], export_sha256=pilot.digest(exported))
    return pilot.fit_gain(program, observations)


def measure_gain(program, receipt, movie, capture_report, out, origin_s, *, finder=None):
    """Automatic stationary gain analysis, using the run-11 disc finder verbatim.

    This estimates spatial gain only. Its coarse origin never becomes a curve
    timebase. No observed curve contributes to this fit. All static samples and
    compact annotated crops are preserved, including on measurement failure.
    """
    pilot.validate_program(program)
    pilot.verify_receipts(receipt, program)
    if program['kind'] != 'gain' or not math.isfinite(origin_s) or origin_s < 0:
        raise ValueError('gain program and finite locating origin required')
    pts = native_pts(movie)
    capture_proof(capture_report, program, movie, pts)
    if finder is None:
        path = Path(__file__).resolve().parents[3]/'hardware/pico_hover_rate/measure.py'
        spec = importlib.util.spec_from_file_location('pico_run11_cursor_finder', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        finder = module.find_cursor
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    windows = {}
    for c in program['commands']:
        start = origin_s+receipt['accepted_us'][c['event_start']]/1e6
        end = origin_s+receipt['accepted_us'][c['event_end']-1]/1e6+c['step_us']/1e6
        windows[c['id']+'-before'] = (start-GAIN_WINDOW[1], start-GAIN_WINDOW[0])
        windows[c['id']+'-after'] = (end+GAIN_WINDOW[0], end+GAIN_WINDOW[1])
    for i, section in enumerate(program['sections']):
        end = origin_s+section['end_us']/1e6
        windows[f'park-{i}'] = (end-GAIN_WINDOW[1], end-GAIN_WINDOW[0])
    selected = {name:[i for i, t in enumerate(pts) if a <= t < b] for name, (a, b) in windows.items()}
    if any(a < pts[0] or b > pts[-1] for a, b in windows.values()) or any(len(v) < 6 for v in selected.values()):
        raise ValueError('gain windows not fully captured')
    owners = {}
    for name, frames in selected.items():
        for i in frames:
            owners.setdefault(i, []).append(name)
    detections = {name:[] for name in windows}
    index = 0
    def inspect(frame):
        nonlocal index
        i = index
        index += 1
        if i not in owners:
            return
        d = finder(frame, (290, 460), board_top_pt=380, polarity=0)
        if not d or d[0] < 8:
            return
        for name in owners[i]:
            detections[name].append(dict(source_frame=i, pts_s=pts[i], score=float(d[0]),
                                         point_pt=[float(d[1]), float(d[2])], radius_pt=float(d[3]), polarity=int(d[4])))
            if len(detections[name]) == 1:
                x, y = round(d[1]*2), round(d[2]*2)
                crop = frame[max(0,y-55):min(frame.shape[0],y+56), max(0,x-55):min(frame.shape[1],x+56)].copy()
                if crop.size == 0 or not cv2.imwrite(str(out/(name+'.jpg')), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise ValueError('gain measurement crop unavailable')
    _decode_source_frames(movie, pts, keep=False, inspect=inspect)
    evidence = dict(schema='pico-usb-static-gain-evidence-v1', program_sha256=program['sha256'],
                    receipt_sha256=receipt['sha256'], movie_sha256=file_sha256(movie),
                    source_pts_s=pts, origin_s=origin_s, selected=selected, detections=detections,
                    method='run-11 circle finder; stationary medians; no spike repair or curve fitting',
                    crops={path.name:file_sha256(path) for path in out.glob('*.jpg')})
    def save(path, value):
        with path.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
    save(out/'evidence.json', evidence)
    try:
        positions, uncertainty = {}, {}
        for name, rows in detections.items():
            if len(rows)/len(selected[name]) < .9:
                raise ValueError(f'{name}: inadequate stationary tracking')
            points = np.array([row['point_pt'] for row in rows])
            centre = np.median(points, axis=0)
            radius = float(np.linalg.norm(points-centre, axis=1).max())
            if radius > 1 or any(not 7 <= row['radius_pt'] <= 10 for row in rows):
                raise ValueError(f'{name}: unstable or ambiguous cursor')
            positions[name] = centre.tolist()
            uncertainty[name] = max(.5, radius)
        observations = dict(program_sha256=program['sha256'], evidence_sha256=pilot.digest(evidence),
            confirmation='automatic stationary tracker quality checks; not human contact review',
            passes=[dict(id=c['id'], before_pt=positions[c['id']+'-before'], after_pt=positions[c['id']+'-after'],
                         uncertainty_pt=max(uncertainty[c['id']+'-before'], uncertainty[c['id']+'-after']),
                         stable=True, confirmed=True) for c in program['commands']],
            park_points_pt=[positions[f'park-{i}'] for i in range(2)])
        profile = pilot.fit_gain(program, observations)
        save(out/'observations.json', observations)
        save(out/'profile.json', profile)
    except Exception as exc:
        save(out/'failure.json', dict(error=str(exc), next_step='inspect preserved gain windows; no automatic live replacement'))
        raise
    return profile


def score_review(private, exported, media_root):
    identity, rows = load_export(private, exported, media_root)
    if identity['kind'] != 'pico-usb-curves-review-v1':
        raise ValueError('curve review required')
    proof = identity['provenance']
    program, receipt, fit, pts = (proof[k] for k in ('program', 'receipts', 'fit', 'source_pts_s'))
    pilot.validate_program(program)
    pilot.verify_receipts(receipt, program)
    controls = proof['control_review']['annotations']
    observed = {c['role']: dict(controls[c['id']], confirmed=controls[c['id']].get('boundary_confirmed') is True)
                for c in program['commands'] if c['kind'] == 'control'}
    if pilot.fit_timebase(program, receipt, observed, pts) != fit:
        raise ValueError('curve timebase differs from independently reviewed controls')
    commands = {c['id']: c for c in program['commands'] if c['kind'] == 'curve'}
    results = [pilot.score_curve(commands[clip['id']], receipt, fit, pts, rows[clip['id']]) for clip in identity['clips']]
    verdict = ('pass' if all(r['verdict'] == 'pass' for r in results) else
               'fail' if any(r['verdict'] == 'fail' for r in results) else 'inconclusive')
    return pilot.seal(dict(schema='pico-usb-curve-result-v1', kind=program['kind'], verdict=verdict,
                           results=results, fit=fit, program_sha256=program['sha256'],
                           profile_sha256=program['profile']['sha256'], receipt_sha256=receipt['sha256'],
                           movie_sha256=proof['movie_sha256'], bundle_sha256=private['bundle_sha256'],
                           export_sha256=pilot.digest(exported), training_admission=False))


def aggregate(first, second):
    for result in (first, second):
        pilot.checked(result)
        if result.get('schema') != 'pico-usb-curve-result-v1':
            raise ValueError('scored pilot batches required')
    if first['kind'] != 'curves-1' or second['kind'] != 'curves-2' or first['profile_sha256'] != second['profile_sha256']:
        raise ValueError('two distinct batches with the same measured gain required')
    expected = {pilot.case(f, 150, r)['id'] for f in pilot.FAMILIES for r in (0, 1)} | {
        pilot.case(f, 600)['id'] for f in pilot.FAMILIES}
    results = first['results']+second['results']
    if len(results) != 12 or {r['id'] for r in results} != expected:
        raise ValueError('exact frozen twelve-case coverage required')
    verdict = ('pass' if all(r['verdict'] == 'pass' for r in results) else
               'fail' if any(r['verdict'] == 'fail' for r in results) else 'inconclusive')
    return pilot.seal(dict(schema='pico-usb-curve-pilot-result-v1', verdict=verdict, results=results,
                          batches=[first['sha256'], second['sha256']], training_admission=False,
                          next_step='broader held-out audit' if verdict == 'pass' else 'diagnose gain, compilation, alignment, visibility and timing separately'))


def build_review(program, receipt, movie, capture_report, out, template_root, *, origin_s=None, control_review=None):
    pilot.validate_program(program)
    pilot.verify_receipts(receipt, program)
    pts = native_pts(movie)
    capture_proof(capture_report, program, movie, pts)
    if Path(movie).resolve().is_relative_to(Path(out).resolve()):
        raise ValueError('raw movie must be outside viewer root')
    windows, clips = [], []
    fit = calibration = None
    kind = 'pico-usb-gain-review-v1' if program['kind'] == 'gain' else 'pico-usb-controls-v1'
    if control_review:
        private, exported, root = control_review
        if private['identity']['provenance']['program']['sha256'] != program['sha256'] or private['identity']['provenance']['receipts'] != receipt or private['identity']['provenance']['movie_sha256'] != file_sha256(movie):
            raise ValueError('control review belongs to a different capture')
        fit, calibration = control_fit(private, exported, root)
        if not fit['accepted']:
            raise ValueError('held-out middle control failed: preserve as inconclusive')
        kind = 'pico-usb-curves-review-v1'
    elif origin_s is None or not math.isfinite(origin_s) or origin_s < 0:
        raise ValueError('approximate play-origin-s required to locate review windows')

    def add_window(id_, start, end, **extra):
        if start < pts[0] or end > pts[-1]:
            raise ValueError('review window outside captured movie; no replacement run')
        indices = [i for i, t in enumerate(pts) if start <= t < end]
        if len(indices) < 3:
            raise ValueError('not enough native frames in review window')
        clip = dict(id=id_, frames=[], **extra)
        clips.append(clip)
        windows.append((clip, set(indices)))

    if program['kind'] == 'gain':
        for c in program['commands']:
            # Human must confirm stability; coarse locating time is not a curve clock fit.
            start = origin_s+receipt['accepted_us'][c['event_start']]/1e6
            end = origin_s+receipt['accepted_us'][c['event_end']-1]/1e6+c['step_us']/1e6
            add_window(c['id']+'-before', start-GAIN_WINDOW[1], start-GAIN_WINDOW[0], purpose='static')
            add_window(c['id']+'-after', end+GAIN_WINDOW[0], end+GAIN_WINDOW[1], purpose='static')
        for i, section in enumerate(program['sections']):
            end = origin_s+section['end_us']/1e6
            add_window(f'park-{i}', end-GAIN_WINDOW[1], end-GAIN_WINDOW[0], purpose='static')
    else:
        for c in program['commands']:
            if kind == 'pico-usb-controls-v1' and c['kind'] == 'control':
                onset = origin_s+receipt['accepted_us'][c['event_start']]/1e6
                add_window(c['id'], onset-.75, onset+.65, purpose='control', role=c['role'])
            elif kind == 'pico-usb-curves-review-v1' and c['kind'] == 'curve':
                onset = fit['intercept_s']+fit['rate']*c['press_us']/1e6
                lift = fit['intercept_s']+fit['rate']*c['lift_us']/1e6
                add_window(c['id'], onset-.15, lift+.20, purpose='curve', command=c, onset_s=onset)
    Path(out).mkdir(parents=True, exist_ok=False)
    out = Path(out)
    for clip in clips:
        (out/clip['id']).mkdir()
    count = 0
    def inspect(frame):
        nonlocal count
        i = count
        count += 1
        for clip, selected in windows:
            if i not in selected:
                continue
            path = f'{clip["id"]}/{i:04}.jpg'
            if not cv2.imwrite(str(out/path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise ValueError('native JPEG export failed')
            target = None
            if clip['purpose'] == 'curve':
                ms = (pts[i]-clip['onset_s'])/fit['rate']*1000
                if 0 <= ms <= clip['command']['path']['duration_ms']:
                    target = pilot.position(clip['command']['path'], ms)
            clip['frames'].append(dict(path=path, source_frame=i, pts_s=pts[i], time_s=pts[i]-pts[min(selected)],
                                       target=target, sha256=file_sha256(out/path)))
    # One exact FFmpeg decode; no resampling or OpenCV PTS substitution.
    _decode_source_frames(movie, pts, keep=False, inspect=inspect)
    proof = dict(program=program, receipts=receipt, capture_report=capture_report, movie_path=str(Path(movie).resolve()),
                 movie_sha256=file_sha256(movie), source_pts_s=pts, fit=fit, control_review=calibration)
    private = bundle(kind, clips, proof, dict(training_admission=False, approximate_origin_s=origin_s))
    # Kept outside the web root; export import verifies every displayed JPEG against it.
    private_path = out.with_name(out.name+'-bundle.json')
    with private_path.open('x') as f:
        json.dump(private, f, indent=2, allow_nan=False)
        f.write('\n')
    data = dict(schema=kind, version=2, bundle_sha256=private['bundle_sha256'], clips=clips)
    (out/'data.js').write_text('const DATA='+json.dumps(data, allow_nan=False).replace('<', '\\u003c')+';\n')
    shutil.copyfile(template_root/'pico_curve_pilot.html', out/'index.html')
    shutil.copyfile(template_root/'review_integrity.js', out/'review_integrity.js')
    return private_path
