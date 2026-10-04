"""Validate and unblind a complete operator export; preserve its bytes and counts."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from trueskate_ai.research.linear_length_probe import blind_key
from trueskate_ai.research.linear_speed_probe import verify_manifest
from trueskate_ai.research.review_provenance import verify_bundle,legacy_provenance,verify_execution
from trueskate_ai.research.curve_protocol import digest


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate JSON key: '+key)
        result[key]=value
    return result


def report(export_path, evidence, out, *, allow_legacy=False, bundle_map=None, media_root=None):
    raw=export_path.read_bytes();export=json.loads(raw,object_pairs_hook=unique_object)
    frozen=json.loads((evidence/'manifest.json').read_text());verify_manifest(frozen)
    key=json.loads((evidence/'private-key.json').read_text())
    if key!=blind_key(frozen):raise ValueError('private key does not match frozen workload')
    summary=json.loads((evidence/'summary.json').read_text())
    if export.get('schema')=='blind-linear-length-v1':
        provenance=legacy_provenance(allow_legacy)
    elif export.get('schema')=='blind-linear-length-v2':
        if bundle_map is None:raise ValueError('v2 review requires --bundle-map')
        private=json.loads(Path(bundle_map).read_text(),object_pairs_hook=unique_object)
        identity=verify_bundle(private,export,media_root)
        if digest(identity['provenance']['frozen_manifest'])!=digest(frozen) or identity['mapping']['key']!=key:
            raise ValueError('bundle frozen conditions differ from evidence')
        if [c['id'] for c in identity['clips']] != [item['token'] for item in key['items']]:
            raise ValueError('bundle clip coverage differs from frozen workload')
        proofs=identity['provenance']['executions']
        if set(proofs)!={str(i) for i in range(1,len(frozen['recordings'])+1)}:
            raise ValueError('incomplete bundle execution provenance')
        for i,commands in enumerate(frozen['recordings'],1):
            proof=proofs[str(i)]
            verify_execution(frozen,commands,proof['planned'],proof['execution'],proof['wda_timing'],[c['payload'] for c in commands])
        provenance='v2: frozen commands, successful execution receipts and displayed JPEG bytes bound'
    else:raise ValueError('wrong export schema')
    if export.get('bundle_sha256')!=summary['blinded_bundle_sha256']:raise ValueError('wrong blinded bundle')
    marks=export['assessments'];expected={x['token'] for x in key['items']}
    if set(marks)!=expected:raise ValueError(f"incomplete/unknown labels: missing={len(expected-set(marks))}, unknown={len(set(marks)-expected)}")
    rows=[]
    for clip,item in enumerate(key['items'],1):
        mark=marks[item['token']]
        if set(mark)!={'trace_visible','board_moved','comments','updated_at'}:raise ValueError('unexpected assessment fields')
        if mark.get('trace_visible') not in ('flicker','hold','trace'):raise ValueError('invalid trace label')
        if type(mark.get('board_moved')) is not bool:raise ValueError('board label must be boolean')
        if not isinstance(mark.get('comments'),str):raise ValueError('comments must be text')
        rows.append(dict(clip=clip,**item,**mark))
    cells=Counter((r['length_fraction'],r['duration_ms']) for r in rows)
    if len(cells)!=45 or set(cells.values())!={3}:raise ValueError('unbalanced review cells')
    def counts(items):
        c=Counter(r['trace_visible'] for r in items)
        return dict(n=len(items),trace=c['trace'],hold=c['hold'],flicker=c['flicker'],board_moved=sum(r['board_moved'] for r in items))
    by_duration=[dict(duration_ms=d,**counts([r for r in rows if r['duration_ms']==d])) for d in frozen['durations_ms']]
    by_cell=[dict(duration_ms=d,length_fraction=l,**counts([r for r in rows if r['duration_ms']==d and r['length_fraction']==l])) for d in frozen['durations_ms'] for l in frozen['length_fractions']]
    ascending=sorted(by_duration,key=lambda r:r['duration_ms'])
    board_transitions=[[a['duration_ms'],b['duration_ms']] for a,b in zip(ascending,ascending[1:]) if a['board_moved']==0 and b['board_moved']==b['n']]
    all_trace=[r['duration_ms'] for r in ascending if r['trace']==r['n']]
    mixed_trace=[r['duration_ms'] for r in ascending if 0<r['trace']<r['n']]
    flickers=[r for r in rows if r['trace_visible']=='flicker']
    result=dict(experiment=frozen['experiment'],assessment_sha256=hashlib.sha256(raw).hexdigest(),bundle_sha256=export['bundle_sha256'],source='operator blinded viewing labels',provenance=provenance,coverage=dict(expected=135,reviewed=len(rows),missing=0,unknown=0,balanced_repetitions=3),overall=counts(rows),by_duration=by_duration,by_cell=by_cell,trace_and_board=[dict(label=v,**counts([r for r in rows if r['trace_visible']==v])) for v in ('trace','hold','flicker')],comments=[r for r in rows if r['comments']],interpretation=dict(board_response_transitions_ms=board_transitions,all_trace_at_tested_durations_ms=all_trace,mixed_trace_at_tested_durations_ms=mixed_trace,input_path_collapse='not established by these video labels'),defaults=dict(trace_visible='trace after UI improvement',board_moved=True))
    out.mkdir(parents=True,exist_ok=False)
    (out/'operator-assessments.json').write_bytes(raw)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'unblinded-assessments.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Blinded review results','',f"All 135 labels validated against bundle `{export['bundle_sha256']}`.",'','| Duration (ms) | Trace | Hold | Flicker | Board moved |','|---:|---:|---:|---:|---:|']
    for r in by_duration:text.append(f"| {r['duration_ms']} | {r['trace']}/15 | {r['hold']}/15 | {r['flicker']}/15 | {r['board_moved']}/15 |")
    text+=['','## Trace counts by gesture length','','Lengths are fractions of the original diagonal. Each cell is trace labels out of three. Board movement totals are reported in the duration table above.','','| Duration (ms) | 20% | 40% | 60% | 80% | 100% |','|---:|---:|---:|---:|---:|---:|']
    for d in frozen['durations_ms']:
        a=[r for r in by_cell if r['duration_ms']==d];text.append('| '+str(d)+' | '+' | '.join(f"{r['trace']}/3" for r in a)+' |')
    text+=['',f'Adjacent tested durations separating no board movement from movement in every clip: {board_transitions}. All-trace durations: {all_trace} ms; mixed trace durations: {mixed_trace} ms. Of {len(flickers)} flicker labels, {sum(r["board_moved"] for r in flickers)} still have board movement. Hold labels: {result["overall"]["hold"]}.','','Three repeats per cell describe this session; they do not establish an exact universal threshold or input path collapse. Trace and board defaults were used by the UI, so these are operator-saved assessments, not independent instrument readings.','','## Comments','']
    for r in result['comments']:text.append(f"- Clip {r['clip']}: {r['duration_ms']} ms, {r['length_fraction']:.0%} length — {r['comments']}")
    (out/'results.md').write_text('\n'.join(text)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('export','evidence','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--allow-legacy',action='store_true',help='explicitly import weaker historical v1 provenance')
    p.add_argument('--bundle-map',type=Path)
    p.add_argument('--media-root',type=Path)
    a=p.parse_args();r=report(a.export,a.evidence,a.out,allow_legacy=a.allow_legacy,bundle_map=a.bundle_map,media_root=a.media_root);print(json.dumps(dict(coverage=r['coverage'],overall=r['overall'])))
