"""Compare ABA capture controls at synthesis cadence, retaining coalesced input."""
import collections
import json
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'observer-20261013'
rows = [json.loads(line) for line in (OUT / 'requests.jsonl').read_text().splitlines()]
main = [json.loads(line) for line in (ROOT / 'resume-20261013/requests.jsonl').read_text().splitlines()]

def sequence(row):
    begin = next(e['ts'] for e in row['touches'] if e['ph'] == 'B')
    points = collections.defaultdict(set)
    for event in row['touches']:
        for ts, x, y, phase in event.get('co') or [[event['ts'], event['x'], event['y'], event['ph']]]:
            # Stationary samples are optionally returned when UIKit coalesces callbacks.
            # Compare B/M/E delivery; retain raw S rows in requests.jsonl.
            if phase == 'S':
                continue
            points[round((ts - begin) * 60)].add((x, y, phase))
    return dict(points)

def duration(row):
    return 1000 * (next(e['ts'] for e in row['touches'] if e['ph'] == 'E') - next(e['ts'] for e in row['touches'] if e['ph'] == 'B'))

result = {}
for name in sorted({row['name'] for row in rows}):
    selected = [row for row in main + rows if row['name'] == name]
    reference = sequence(selected[0])
    comparisons = []
    for row in selected:
        observed = sequence(row)
        reference_xy = {frame: {(x, y) for x, y, phase in points} for frame, points in reference.items()}
        observed_xy = {frame: {(x, y) for x, y, phase in points} for frame, points in observed.items()}
        comparisons.append({'group': row.get('group', 'main'), 'repeat': row['repeat'],
                            'all_frames_phases_positions_identical': observed == reference,
                            'all_frame_positions_identical': observed_xy == reference_xy,
                            'different_frames': [frame for frame in sorted(set(reference) | set(observed)) if reference.get(frame) != observed.get(frame)],
                            'duration_ms': duration(row), 'callbacks': len(row['touches']),
                            'unique_frame_phase_positions': sum(map(len, observed.values()))})
    result[name] = {'comparisons': comparisons,
                    'all_20_sequences_identical': all(r['all_frames_phases_positions_identical'] for r in comparisons),
                    'all_20_frame_positions_identical': all(r['all_frame_positions_identical'] for r in comparisons),
                    'duration_range_ms': [min(duration(row) for row in selected), max(duration(row) for row in selected)],
                    'duration_sd_ms': float(np.std([duration(row) for row in selected]))}
(OUT / 'trajectory-validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: {a: b for a, b in v.items() if a != 'comparisons'} for k, v in result.items()}, indent=2))
