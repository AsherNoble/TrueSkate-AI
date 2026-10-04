"""Versioned review identities bind frozen commands, successful execution and media bytes."""
from pathlib import Path
import hashlib
import math
import warnings

from trueskate_ai.collection.wda_action_timing import validate_action_timing_report
from trueskate_ai.research.curve_protocol import digest

VERSION = 2


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify_execution(frozen, commands, planned, execution, timing, payloads):
    if digest({k: v for k, v in frozen.items() if k != 'sha256'}) != frozen.get('sha256'):
        raise ValueError('frozen manifest checksum mismatch')
    for value in (planned, execution):
        if value.get('manifest_sha256') != frozen['sha256']:
            raise ValueError('execution manifest provenance mismatch')
        for field in ('device', 'park', 'experiment'):
            if field in frozen and value.get(field) != frozen[field]:
                raise ValueError(f'execution {field} differs from frozen manifest')
    if digest(planned.get('commands')) != digest(commands):
        raise ValueError('planned ordered command specifications differ from manifest')
    if execution.get('execution_schema') != 'research-execution-v2' or execution.get('error') is not None:
        raise ValueError('successful content-bound execution receipts required')
    events = execution.get('events')
    if not isinstance(events, list) or len(events) != len(commands) or len(payloads) != len(commands):
        raise ValueError('incomplete ordered execution receipts')
    previous = -math.inf
    for spec, event, payload in zip(commands, events, payloads):
        if digest(event.get('spec')) != digest(spec) or digest(event.get('payload')) != digest(payload):
            raise ValueError('executed specification or payload differs from frozen command')
        if event.get('payload_sha256') != digest(payload) or event.get('success') is not True:
            raise ValueError('missing successful payload receipt')
        start, end = event.get('call_start_monotonic_s'), event.get('call_end_monotonic_s')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)) or not previous <= start <= end:
            raise ValueError('unordered or invalid execution timestamps')
        previous = end
    revision = planned.get('wda_revision')
    if not isinstance(revision, str) or not revision or execution.get('wda_revision') != revision:
        raise ValueError('missing WDA execution revision provenance')
    records = validate_action_timing_report(timing, expected_revision=revision, expected_count=len(commands))
    if any(b['request_entered']['monotonic_s'] < a['request_finished']['monotonic_s'] for a, b in zip(records, records[1:])):
        raise ValueError('overlapping or reordered WDA execution receipts')
    return dict(manifest_sha256=frozen['sha256'], planned=planned, execution=execution,
                wda_timing=timing, payloads=payloads)


def bundle(kind, clips, provenance, mapping):
    identity = dict(version=VERSION, kind=kind, clips=clips, provenance=provenance, mapping=mapping)
    return dict(schema=kind, version=VERSION, bundle_sha256=digest(identity), identity=identity)


def verify_bundle(private, export, media_root):
    identity = private.get('identity')
    if private.get('version') != VERSION or not isinstance(identity, dict) or identity.get('version') != VERSION:
        raise ValueError('review bundle v2 required')
    actual = digest(identity)
    if private.get('bundle_sha256') != actual or export.get('bundle_sha256') != actual or export.get('schema') != identity.get('kind'):
        raise ValueError('review content identity mismatch')
    if media_root is None:
        raise ValueError('v2 review import requires --media-root for byte verification')
    root = Path(media_root).resolve()
    identifiers = [clip.get('id', clip.get('token')) for clip in identity['clips']]
    if any(not isinstance(token, str) or not token for token in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError('invalid or duplicate review clip identity')
    for clip in identity['clips']:
        if not clip.get('frames'):
            raise ValueError('empty review clip')
        for frame in clip['frames']:
            path = (root / frame['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file() or file_sha256(path) != frame.get('sha256'):
                raise ValueError('review frame bytes changed or media missing')
    return identity


def legacy_provenance(allow_legacy):
    if not allow_legacy:
        raise ValueError('legacy v1 review requires explicit --allow-legacy; weaker provenance')
    message = 'legacy v1: weaker provenance; execution content and displayed frame bytes are not bound'
    warnings.warn(message, UserWarning, stacklevel=2)
    return message


def curve_execution(frozen, planned, execution, timing):
    from trueskate_ai.research.curve_protocol import verify_manifest
    from trueskate_ai.research.curve_probe import scheduled_commands, command_payload
    verify_manifest(frozen, check_source=False)
    device, stage = planned['device'], planned['stage']
    rows = frozen['devices'][device][stage]
    if stage == 'confirmation':
        rule = str(planned['commands'][1]['rule_ms'])
        rows = rows[rule]
    start = planned['segment_index'] * 8
    commands = scheduled_commands(rows[start:start + 8])
    size = planned['device_size']
    if size != [414, 896]:
        raise ValueError('unexpected XR device profile')
    payloads = [command_payload(spec, size) for spec in commands]
    return verify_execution(frozen, commands, planned, execution, timing, payloads)


def curved_execution(frozen, segment, planned, execution, timing):
    from trueskate_ai.research.curved_audit import verify_manifest
    verify_manifest(frozen)
    if type(segment) is not int or not 1 <= segment <= len(frozen['recordings']):
        raise ValueError('invalid curved audit segment')
    row = frozen['recordings'][segment - 1]
    for value in (planned, execution):
        if value.get('identity') != frozen['identity'] or value.get('segment') != segment:
            raise ValueError('curved audit identity/segment mismatch')
        if any(value.get(field) != row[field] for field in ('device', 'park')):
            raise ValueError('curved audit device/park differs from frozen segment')
    commands = row['commands']
    return verify_execution(frozen, commands, planned, execution, timing, [c['payload'] for c in commands])


def curve_row_provenance(row):
    proof = row.get('execution_provenance')
    if not isinstance(proof, dict) or 'frozen_manifest' not in proof:
        raise ValueError('content-bound curve execution provenance required')
    verified = curve_execution(proof['frozen_manifest'], proof['planned'], proof['execution'], proof['wda_timing'])
    matches = [(i, event['spec']) for i, event in enumerate(verified['execution']['events'])
               if event['spec'].get('command_id') == row['command_id']]
    if len(matches) != 1:
        raise ValueError('curve row missing unique execution receipt')
    index, spec = matches[0]
    if row.get('device') != proof['planned']['device']:
        raise ValueError('curve row device differs from executed command')
    for field in ('command_id', 'family', 'curve_id', 'repeat', 'rule_ms', 'stage', 'offline_eligible'):
        if row.get(field) != spec.get(field):
            raise ValueError('curve row condition differs from executed command')
    if row.get('duration_s') != spec['curve']['duration_s'] or row.get('n') != spec['compiled']['n']:
        raise ValueError('curve row duration or segmentation differs from executed command')
    for field in ('min_segment_duration_ms', 'max_segment_duration_ms'):
        if row.get(field) != spec['compiled'][field]:
            raise ValueError('curve row segmentation differs from executed command')
    calibration = proof.get('calibration', {})
    if calibration.get('accepted') is not True:
        raise ValueError('curve row has no accepted recording calibration')
    fit = calibration['fit']
    onset = fit['intercept_s'] + fit['rate'] * proof['wda_timing']['records'][index]['submitted_to_ios']['monotonic_s']
    if row.get('onset_s') != onset:
        raise ValueError('curve row onset differs from executed command calibration')
    if file_sha256(row['original_video']) != proof.get('original_sha256'):
        raise ValueError('curve original video bytes changed')
    return proof
