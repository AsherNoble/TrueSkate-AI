"""Evaluate the frozen first/last-anchor rule against blind onset labels."""
import argparse
import json
from pathlib import Path


def compare(report, manifest, labels, frame_index):
    records, gestures, marks = report['records'], manifest['gestures'], labels['labels']
    frames = frame_index['frames']
    if len(records) != 11 or len(gestures) != 11 or len(marks) != 11:
        raise ValueError('Expected all eleven touches; do not discard observations')
    if report['dropped_records'] != 0 or labels['frame_count'] != len(frames):
        raise ValueError('Incomplete records or mismatched frame count')
    if [i for i,m in enumerate(marks) if m['kind']=='calibration'] != [0,5,10]:
        raise ValueError('Expected calibration labels at start, middle, end')
    if any(m['kind'] != 'gesture' for i,m in enumerate(marks) if i not in (0,5,10)):
        raise ValueError('Unexpected label type')
    if any(a['frame'] >= b['frame'] for a,b in zip(marks,marks[1:])):
        raise ValueError('Labels must be chronological')
    for i,(r,m) in enumerate(zip(records,marks)):
        if r['sequence'] != i or r['outcome'] != 'success':
            raise ValueError('Unordered or failed timing records')
        if abs(float(frames[m['frame']]['best_effort_timestamp_time'])-m['time_s'])>1e-6:
            raise ValueError('Label time does not match original frame timestamp')
    x=[r['submitted_to_ios']['monotonic_s'] for r in records]
    y=[m['time_s'] for m in marks]
    if any(b<=a for a,b in zip(x,x[1:])): raise ValueError('Nonchronological WDA timestamps')
    rate=(y[-1]-y[0])/(x[-1]-x[0])
    rows=[]
    for i in range(11):
        one=y[0]+x[i]-x[0]; two=y[0]+rate*(x[i]-x[0])
        rows.append(dict(index=i,kind=marks[i]['kind'],frame=marks[i]['frame'],fit_anchor=i in (0,10),
                         observed_s=y[i],single_anchor_error_ms=(y[i]-one)*1000,
                         two_anchor_error_ms=(y[i]-two)*1000))
    held=[r for r in rows if not r['fit_anchor']]
    stats={}
    for method in ('single_anchor','two_anchor'):
        errors=[abs(r[method+'_error_ms']) for r in held]
        stats[method]=dict(evaluated=len(errors),within_one_frame=sum(e<=1000/30 for e in errors),
                           max_absolute_ms=max(errors),mean_absolute_ms=sum(errors)/len(errors))
    span=x[-1]-x[0]
    return dict(rule='First and last calibration only; middle calibration and all gestures held out',
                residual_sign='Observed minus predicted',clock='submitted_to_ios.monotonic_s',
                anchor_span_s=span,minute_admission=span>=55,rate=rate,rows=rows,stats=stats,
                initial_test_pass=span>=55 and stats['two_anchor']['within_one_frame']==9)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('run','viewer','labels','out'): p.add_argument('--'+flag,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise ValueError('Preserve existing results; choose a new output')
    read=lambda p:json.loads(p.read_text())
    labels=read(args.labels); provenance=read(args.viewer/'provenance.json')
    if labels['recording']!=provenance['recording']: raise ValueError('Recording identity mismatch')
    result=compare(read(args.run/'wda-action-timings.json'),read(args.run/'segment_00000.json'),
                   labels,read(args.viewer/'frame_index.json'))
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['stats'],indent=2))

if __name__=='__main__':main()
