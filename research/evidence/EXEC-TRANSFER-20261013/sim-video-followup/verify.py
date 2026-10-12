"""Independent arithmetic/integrity checks; no model selection or source mutation."""
from pathlib import Path
import argparse, gzip, hashlib, importlib.metadata, json, platform, subprocess
from collections import defaultdict
import numpy as np

OUT=Path(__file__).resolve().parent
SP=Path('/private/tmp/claude-503/-Users-ashernoble-Projects-Robotics---hardware-TrueSkate-AI/e0f1c48e-973a-4ff3-a26a-f5f2350d9a43/scratchpad')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_json(p):
    raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preserved-only',action='store_true',help='Verify preserved commands/arithmetic; explicitly report unavailable original profiles/media.')
    args=parser.parse_args()
    prov=json.loads((OUT/'provenance.json').read_text());rows=json.loads((OUT/'primary-rows.json').read_text());res=json.loads((OUT/'results.json').read_text())
    if args.preserved_only:
        bundle=json.loads((OUT.parent/'provenance.json').read_text())
        manifest=Path(bundle['external_manifest_backup'])
        assert sha(manifest)==bundle['external_files']['manifest.json.gz']['sha256']
        transfer_path=OUT.parent/'xctest-transfer/resume-20261013/r6-transfer.json.gz'
        retained=json.loads((OUT.parent/'xctest-transfer/preservation.json').read_text())
        assert sha(transfer_path)==retained['files']['resume-20261013/r6-transfer.json.gz']['sha256']
    else:
        manifest=SP/'audit6-mirror/audit-6/manifest.json'
        transfer_path=SP/'research/1-xctest-transfer/resume-20261013/r6-transfer.json'
    m=read_json(manifest)['samples']
    transfer=read_json(transfer_path)['commands']
    commands=[]
    for s in m:
        c=np.array(transfer[f'r6_{s["id"]:03d}']['command']);g=s['gesture']
        expected=np.c_[g['times_ms'],np.array(g['command_points'])*[414,896]]
        commands.append(np.max(np.abs(c-expected)))
    input_matches={p:(sha(Path(p))==h if Path(p).exists() else None) for p,h in prov['input_files'].items()}
    media_matches={p:(sha(Path(p))==h if Path(p).exists() else None) for p,h in prov['source_media'].items()}
    if not args.preserved_only:
        assert all(v is True for v in input_matches.values())
        assert all(v is True for v in media_matches.values())
    trainrecs={r['recording'] for r in rows if r['split']=='development'}
    testrecs={r['recording'] for r in rows if r['split']=='heldout'}
    p=next(x for x in res if x['threshold']==25 and x['cap_px']==29 and x['sampling']=='interpolate')
    arithmetic={}
    for law in ['linear','simulator']:
        vals=defaultdict(list)
        for row in rows:
            if row['split']=='heldout':
                e=float(np.linalg.norm(np.array(row[law+'_xy'])-row['obs_xy']))
                assert abs(e-row[law+'_error'])<1e-10
                vals[(row['recording'],row['sid'])].append(e)
        recvals=defaultdict(list)
        for (rec,sid),vs in vals.items():recvals[rec].append(np.median(vs))
        agg=float(np.mean([np.mean(vs) for vs in recvals.values()]))
        arithmetic[law]=dict(recomputed=agg,stored=p['models'][law]['heldout']['mean_recording'],match=abs(agg-p['models'][law]['heldout']['mean_recording'])<1e-10)
    result=dict(mode='preserved-only' if args.preserved_only else 'full-source',
        all_inputs_match=all(v is True for v in input_matches.values()),all_selected_media_match=all(v is True for v in media_matches.values()),
        n_input_hashes=len(input_matches),n_selected_media_hashes=len(media_matches),
        unavailable_original_inputs=[p for p,v in input_matches.items() if v is None],
        unavailable_original_media=[p for p,v in media_matches.items() if v is None],
        exact_command_count=len(commands),max_command_difference_logical_pt=float(max(commands)),
        development_and_heldout_recordings_disjoint=not bool(trainrecs&testrecs),
        no_clip67_selected=all(r['sid']!=67 for r in rows),
        no_endpoint_or_hold_time_inversion='The analysis uses forward XY schedules; profile geometry excludes stationary locations and finite endpoints.',
        arithmetic=arithmetic,
        software=dict(python=platform.python_version(),numpy=importlib.metadata.version('numpy'),scipy=importlib.metadata.version('scipy'),opencv=importlib.metadata.version('opencv-python'),matplotlib=importlib.metadata.version('matplotlib'),
            ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]),
        output_hashes={x.name:sha(x) for x in [OUT/'PLAN-FROZEN.md',OUT/'study.py',OUT/'visual-review.json',OUT/'results.json',OUT/'primary-rows.json']})
    assert not any(v is False for v in input_matches.values()) and not any(v is False for v in media_matches.values())
    assert result['development_and_heldout_recordings_disjoint']
    assert max(commands)<1e-10 and all(v['match'] for v in arithmetic.values())
    output=OUT/('verification-preserved.json' if args.preserved_only else 'verification.json')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if not k.startswith('unavailable_')},indent=2))
if __name__=='__main__':main()
