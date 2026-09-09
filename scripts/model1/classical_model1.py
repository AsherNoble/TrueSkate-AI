"""Resumable local classical Model 1 experiments; test scoring requires explicit freeze."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import cv2
import numpy as np
from trueskate_ai.data.clip_frames import _decode_even_frames
from trueskate_ai.model1.classical.predictor import Config, extract, predict_features, metrics


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)); temp.replace(path)


def source_hash():
    return digest([(str(p.relative_to(ROOT)), hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in [Path(__file__), ROOT / 'src/trueskate_ai/model1/classical/predictor.py']])


def content_hash(path):
    h = hashlib.sha256()
    files = [path / 'meta.json', *sorted(path.glob('frame_*.png'))]
    if not files[1:]: files.append(path / 'frames.mp4')
    for p in files:
        h.update(p.name.encode())
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def prepare(args):
    from trueskate_ai.model1.linear.dataset import BasicLinearClipDataset, split_by_command
    ds = BasicLinearClipDataset(args.data)
    partitions = split_by_command(ds, seed=0)
    entries = []
    for name, indices in zip(('train', 'validation', 'test'), partitions):
        for i in indices:
            p = ds.sample_paths[i]
            entries.append({'path': str(p), 'partition': name, 'command': ds.command_keys[i],
                            'content_sha256': content_hash(p)})
    baseline = json.loads(args.baseline.read_text()) if args.baseline else {}
    original_paths = [args.original_prefix.rstrip('/') + '/' + str(p.relative_to(args.data.resolve()))
                      for p in ds.sample_paths] if args.original_prefix else []
    fp = hashlib.sha256(''.join(p + '\n' for p in original_paths).encode()).hexdigest()
    recovered = f'sha256:{len(ds)}:{fp}'
    verified = bool(original_paths and baseline.get('dataset_fingerprint') == recovered)
    payload = {'entries': entries, 'fingerprint': digest(entries), 'baseline_path_identity_verified': verified,
               'reconstructed_original_fingerprint': recovered, 'baseline': baseline,
               'provenance_limit': 'Historical fingerprint binds path membership, not historical video bytes.',
               'split_sizes': {k:len(v) for k,v in zip(('train','validation','test'),partitions)}}
    if args.out.exists(): raise ValueError('refusing to overwrite a sealed manifest')
    atomic(args.out, payload)
    print(json.dumps({k:v for k,v in payload.items() if k not in ('entries','baseline')}, indent=2))


def evaluate(args):
    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    if digest(manifest['entries']) != manifest['fingerprint']: raise ValueError('manifest changed')
    config = Config(**json.loads(args.config.read_text())) if args.config else Config()
    run_id = digest([manifest['fingerprint'], asdict(config), source_hash(), args.partition, args.limit])[:20]
    args.out.mkdir(parents=True, exist_ok=True)
    frozen_path = args.out / 'frozen.json'
    if args.partition == 'test':
        frozen = json.loads(frozen_path.read_text())
        if frozen != {'config':asdict(config), 'source':source_hash(), 'manifest':manifest['fingerprint']}:
            raise ValueError('test requires the exact frozen source, config and manifest')
        if not manifest['baseline_path_identity_verified']: raise ValueError('original corpus identity unverified')
        if args.limit: raise ValueError('test must use complete split')
        marker = args.out / 'test_started.json'
        if marker.exists() and json.loads(marker.read_text())['run_id'] != run_id:
            raise ValueError('a different final test was already started')
        if not marker.exists(): atomic(marker, {'run_id':run_id})
    entries = [e for e in manifest['entries'] if e['partition'] == args.partition]
    if args.limit:
        entries = sorted(entries, key=lambda e: digest(e['command']))[:args.limit]
    run_dir = args.out / run_id; run_dir.mkdir(exist_ok=True)
    config_id = digest({k:v for k,v in asdict(config).items() if k in
                        ('hue_low','hue_high','saturation','value','difference','min_area')})[:16]
    extractor_id = digest([inspect.getsource(extract), '32-128x288-BGR-elapsed-v1'])[:16]
    cv2.setNumThreads(1)
    def one(e):
        path = Path(e['path']); key = digest(e)[:24]; saved = run_dir / (key+'.json')
        if content_hash(path) != e['content_sha256']: raise ValueError(f'content changed: {path}')
        if saved.exists(): return json.loads(saved.read_text())
        meta = json.loads((path/'meta.json').read_text())
        feature_path = args.out / 'features' / extractor_id / config_id / (key+'.json')
        error = None
        try:
            if feature_path.exists(): features = json.loads(feature_path.read_text())
            else:
                pngs = sorted(path.glob('frame_*.png'))
                if pngs: count = len(pngs)
                else:
                    capture = cv2.VideoCapture(str(path/'frames.mp4'))
                    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); capture.release()
                if count != len(meta['frame_times']): raise ValueError('video/metadata frame-count mismatch')
                frames = _decode_even_frames(path, 32)
                frames = [cv2.resize(f, (128,288), interpolation=cv2.INTER_AREA) for f in frames]
                raw = np.asarray(meta['frame_times'], dtype=float)
                selected = np.linspace(0,len(raw)-1,32).round().astype(int)
                elapsed = raw[selected] - raw[0]  # absolute touch anchor deliberately removed
                features = extract(frames, elapsed, config)
                atomic(feature_path, features)
            prediction = predict_features(features, config)
        except (ValueError, cv2.error) as exc:
            prediction = None; error = str(exc)
        target = [*meta['waypoints'][0], *meta['waypoints'][1], meta['duration']]
        record = {'path':str(path), 'prediction':None if prediction is None else list(map(float,prediction)),
                  'target':target, 'error':error}
        atomic(saved, record); return record
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Bounded chunks: avoid queuing a whole corpus past a deadline.
        for offset in range(0,len(entries),max(2,args.workers*4)):
            if time.monotonic()-started >= args.seconds: break
            rows.extend(pool.map(one, entries[offset:offset+max(2,args.workers*4)]))
            atomic(run_dir/'progress.json', {'completed':len(rows),'total':len(entries), 'elapsed_s':time.monotonic()-started})
    result = {'run_id':run_id, 'partition':args.partition, 'config':asdict(config), 'source':source_hash(),
              'manifest':manifest['fingerprint'], 'complete':len(rows)==len(entries),
              'wall_seconds':time.monotonic()-started, 'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'metrics':metrics([r['prediction'] for r in rows],[r['target'] for r in rows]) if rows else None,
              'hypothesis':args.hypothesis}
    atomic(run_dir/'summary.json', result)
    atomic(args.out/'latest.json', result)
    print(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline',type=Path);p.add_argument('--original-prefix');p.set_defaults(func=prepare)
    p=sub.add_parser('evaluate');p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path);p.add_argument('--partition',choices=['train','validation','test'],default='train')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seconds',type=float,default=900);p.add_argument('--hypothesis',default='baseline');p.set_defaults(func=evaluate)
    p=sub.add_parser('freeze');p.add_argument('--manifest',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    def freeze(a):
        if (a.out/'test_started.json').exists(): raise ValueError('test already started')
        m=json.loads(a.manifest.read_text());c=asdict(Config(**json.loads(a.config.read_text())))
        atomic(a.out/'frozen.json',{'config':c,'source':source_hash(),'manifest':m['fingerprint']})
    p.set_defaults(func=freeze)
    args=ap.parse_args()
    if hasattr(args, 'workers') and not 1 <= args.workers <= 2: ap.error('workers must be 1 or 2')
    if hasattr(args, 'seconds') and args.seconds <= 0: ap.error('seconds must be positive')
    if hasattr(args, 'limit') and args.limit < 0: ap.error('limit must be nonnegative')
    args.func(args)

if __name__=='__main__': main()
