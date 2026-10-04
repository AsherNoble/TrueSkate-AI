import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from trueskate_ai.collection.wda_action_timing import BOUNDARIES
from trueskate_ai.research.curve_protocol import digest
from trueskate_ai.research.linear_speed_probe import manifest
from trueskate_ai.research.review_provenance import bundle, file_sha256, verify_bundle, verify_execution

ROOT = Path(__file__).resolve().parents[1]


def execution_fixture():
    frozen = json.loads(json.dumps(manifest()))
    commands = frozen['recordings'][0]
    meta = dict(manifest_sha256=frozen['sha256'], experiment=frozen['experiment'],
                device=frozen['device'], park=frozen['park'], wda_revision='fixture')
    planned = dict(**meta, commands=commands)
    events, records = [], []
    for i, spec in enumerate(commands):
        events.append(dict(spec=spec, payload=spec['payload'], payload_sha256=digest(spec['payload']),
                           success=True, call_start_monotonic_s=spec['slot_s'],
                           call_end_monotonic_s=spec['slot_s'] + .1))
        row = dict(sequence=i, outcome='success', session_id='fixture', missing_ios_callback=False, ios_callback_result=True)
        row.update({name: dict(monotonic_s=spec['slot_s'] + k * .001, epoch_s=1000 + spec['slot_s'] + k * .001)
                    for k, name in enumerate(BOUNDARIES)})
        records.append(row)
    execution = dict(**meta, events=events, error=None, execution_schema='research-execution-v2')
    timing = dict(schema_version=1, build_revision='fixture', dropped_records=0, records=records)
    return frozen, commands, planned, execution, timing, [c['payload'] for c in commands]


def test_complete_ordered_receipts_bind_payload_and_frozen_conditions():
    assert verify_execution(*execution_fixture())['wda_timing']['records'][-1]['outcome'] == 'success'


@pytest.mark.parametrize('mode', ['condition_same_id', 'order', 'payload', 'failed', 'missing_end', 'receipt_order',
                                  'receipt_failed', 'receipt_callback', 'receipt_count', 'manifest', 'park', 'planned', 'legacy'])
def test_provenance_mismatch_rejected(mode):
    frozen, commands, planned, execution, timing, payloads = copy.deepcopy(execution_fixture())
    if mode == 'condition_same_id':
        execution['events'][2]['spec'] = dict(execution['events'][2]['spec'], duration_ms=999)
    elif mode == 'order':
        execution['events'][0], execution['events'][1] = execution['events'][1], execution['events'][0]
    elif mode == 'payload':
        execution['events'][2]['payload'] = {'actions': []}
        execution['events'][2]['payload_sha256'] = digest({'actions': []})
    elif mode == 'failed': execution['events'][2]['success'] = False
    elif mode == 'missing_end': del execution['events'][2]['call_end_monotonic_s']
    elif mode == 'receipt_order': timing['records'][2]['sequence'] = 0
    elif mode == 'receipt_failed': timing['records'][2]['outcome'] = 'failure'
    elif mode == 'receipt_callback': timing['records'][2]['ios_callback_result'] = False
    elif mode == 'receipt_count': timing['records'].pop()
    elif mode == 'manifest': execution['manifest_sha256'] = 'other'
    elif mode == 'park': execution['park'] = 'other'
    elif mode == 'planned': planned['commands'] = commands[::-1]
    elif mode == 'legacy': execution.pop('execution_schema')
    with pytest.raises(ValueError):
        verify_execution(frozen, commands, planned, execution, timing, payloads)


def media_fixture(tmp_path):
    media = tmp_path / 'public'
    (media / 'opaque').mkdir(parents=True)
    frame = media / 'opaque/000.jpg'
    frame.write_bytes(b'synthetic JPEG bytes')
    clips = [dict(token='opaque', frames=[dict(path='opaque/000.jpg', source_frame=1, pts_s=.1, sha256=file_sha256(frame))])]
    private = bundle('blind-curve-v2', clips, dict(rows_sha256='rows'),
                     {'opaque': dict(row_index=0, kind='annotations', source_frames=[1], pts_s=[.1])})
    export = dict(schema='blind-curve-v2', bundle_sha256=private['bundle_sha256'], annotations={'opaque': {'centreline': []}})
    return media, frame, private, export


def test_mutated_jpeg_and_rebuilt_bundle_invalidate_prior_assessment(tmp_path):
    media, frame, private, export = media_fixture(tmp_path)
    verify_bundle(private, export, media)
    frame.write_bytes(b'changed JPEG bytes')
    with pytest.raises(ValueError, match='bytes changed'):
        verify_bundle(private, export, media)
    identity = copy.deepcopy(private['identity'])
    identity['clips'][0]['frames'][0]['sha256'] = file_sha256(frame)
    rebuilt = bundle(identity['kind'], identity['clips'], identity['provenance'], identity['mapping'])
    assert rebuilt['bundle_sha256'] != private['bundle_sha256']
    with pytest.raises(ValueError, match='identity mismatch'):
        verify_bundle(rebuilt, export, media)


def test_bundle_binds_execution_and_rejects_path_traversal(tmp_path):
    media, frame, private, export = media_fixture(tmp_path)
    private['identity']['provenance']['rows_sha256'] = 'different execution'
    with pytest.raises(ValueError, match='identity mismatch'):
        verify_bundle(private, export, media)
    private['identity']['clips'][0]['frames'][0]['path'] = '../outside.jpg'
    private['bundle_sha256'] = export['bundle_sha256'] = digest(private['identity'])
    with pytest.raises(ValueError, match='bytes changed'):
        verify_bundle(private, export, media)


def load_curve_importer():
    spec = importlib.util.spec_from_file_location('curve_importer', ROOT / 'scripts/inspect/build_curve_audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_curve_import_checks_bytes_and_legacy_is_explicit(tmp_path):
    media, frame, private, export = media_fixture(tmp_path)
    p, e = tmp_path / 'private.json', tmp_path / 'export.json'
    p.write_text(json.dumps(private)); e.write_text(json.dumps(export))
    load_curve_importer().import_marks(p, e, tmp_path / 'valid.json', media_root=media)
    assert json.loads((tmp_path / 'valid.json').read_text())['provenance'].startswith('v2')
    frame.write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        load_curve_importer().import_marks(p, e, tmp_path / 'bad.json', media_root=media)
    assert not (tmp_path / 'bad.json').exists()
    legacy = dict(bundle_sha256='legacy', rows_sha256='rows', mapping=private['identity']['mapping'])
    p.write_text(json.dumps(legacy)); e.write_text(json.dumps(dict(bundle_sha256='legacy', annotations={})))
    with pytest.raises(ValueError, match='--allow-legacy'):
        load_curve_importer().import_marks(p, e, tmp_path / 'legacy-rejected.json')
    with pytest.warns(UserWarning, match='weaker provenance'):
        load_curve_importer().import_marks(p, e, tmp_path / 'legacy.json', allow_legacy=True)
    assert 'weaker provenance' in json.loads((tmp_path / 'legacy.json').read_text())['provenance']


def test_http_browser_verifier_hashes_actual_displayed_bytes():
    node = shutil.which('node')
    if not node: pytest.skip('Node required for offline browser integrity check')
    helper = (ROOT / 'scripts/inspect/templates/review_integrity.js').read_text()
    checks = r'''
const assert=require('node:assert/strict'),native=require('node:crypto');
for(const length of [0,1,3,55,56,63,64,65,999,10000]){
 const bytes=Uint8Array.from({length},(_,i)=>(i*19)%256);
 assert.equal(reviewSHA256(bytes),native.createHash('sha256').update(bytes).digest('hex'));
}
(async()=>{
 let bytes=Buffer.from('original JPEG');
 const frame={path:'same.jpg',sha256:native.createHash('sha256').update(bytes).digest('hex')};
 global.fetch=async(path,options)=>{assert.equal(path,'same.jpg');assert.equal(options.cache,'no-store');return {ok:true,arrayBuffer:async()=>Uint8Array.from(bytes).buffer}};
 const url=await reviewFrameUrl(frame);assert.ok(url.startsWith('blob:'));URL.revokeObjectURL(url);
 bytes=Buffer.from('changed JPEG');await assert.rejects(reviewFrameUrl(frame),/Footage changed/);
 await assert.rejects(reviewFrameUrl({path:'same.jpg'}),/Missing review frame identity/);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run([node, '-'], input=helper + checks, text=True, capture_output=True, check=True)
