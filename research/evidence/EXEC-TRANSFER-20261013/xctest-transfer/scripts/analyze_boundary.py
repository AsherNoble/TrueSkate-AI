"""Recompute boundary summaries and validate all compiled native record points."""
import collections
import csv
import json
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'boundary-20261013'
manifest = json.loads((OUT / 'manifest.json').read_text())
rows = [json.loads(line) for line in (OUT / 'requests.jsonl').read_text().splitlines()]
grouped = collections.defaultdict(list)
metrics = []
matching_points = 0
for row in rows:
    cmd = manifest['probes'][row['name']]['cmd']
    native = row['native_dump']
    expected = [[1, 0.0, cmd[0][1], cmd[0][2]]] + [[2, t / 1000, x, y] for t, x, y in cmd[1:]] + [[3, cmd[-1][0] / 1000, cmd[-1][1], cmd[-1][2]]]
    actual = native['paths'][0]
    assert len(actual) == len(expected), (row['name'], actual, expected)
    for observed, target in zip(actual, expected):
        vector = [observed[key] for key in ['event_type', 'offset_s', 'x', 'y']]
        assert np.allclose(vector, target, rtol=0, atol=1e-7), (row['name'], observed, target)
        matching_points += 1
    begin = next(e['ts'] for e in row['touches'] if e['ph'] == 'B')
    end = next(e['ts'] for e in row['touches'] if e['ph'] == 'E')
    item = {'name': row['name'], 'repeat': row['repeat'], 'command_ms': cmd[-1][0],
            'native_ms': native['maximum_offset_s'] * 1000, 'delivered_ms': (end - begin) * 1000}
    metrics.append(item)
    grouped[row['name']].append(item)
summary = {name: {'n': len(items), 'command_ms': items[0]['command_ms'], 'native_ms': items[0]['native_ms'],
                  'delivered_median_ms': float(np.median([item['delivered_ms'] for item in items])),
                  'delivered_sd_ms': float(np.std([item['delivered_ms'] for item in items])),
                  'all_native_offsets_exact': True} for name, items in grouped.items()}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
(OUT / 'native-record-validation.json').write_text(json.dumps({'requests_checked': len(rows), 'matching_record_points': matching_points, 'all_timestamps_and_positions_match_commands': True, 'event_types': '1 down,2 move,3 up'}, indent=2))
with (OUT / 'metrics.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)
print('Validated', len(rows), 'requests;', matching_points, 'native record points')
