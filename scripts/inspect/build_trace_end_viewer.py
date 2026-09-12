"""Add a last-visible-frame viewer beside an existing onset viewer.

Original onset labels are embedded unchanged. End labels are paired by start
frame and saved under a separate browser key and export filename.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path


def script_json(value):
    return json.dumps(value).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def build(viewer_dir: Path, labels_path: Path) -> Path:
    labels = json.loads(labels_path.read_text())
    index = json.loads((viewer_dir / 'frame_index.json').read_text())
    times = [float(f['best_effort_timestamp_time']) for f in index['frames']]
    if labels.get('frame_count') != len(times) or labels.get('frame_numbering') != 'zero-based':
        raise ValueError('Labels and recording frame index disagree')
    for a in labels['labels']:
        f = a['frame']
        if not isinstance(f, int) or not 0 <= f < len(times) or abs(a['time_s'] - times[f]) > 1e-6:
            raise ValueError('Start label timestamp/frame mismatch')
    if not any(a['kind'] == 'gesture' for a in labels['labels']):
        raise ValueError('No gesture starts to annotate')
    provenance = json.loads((viewer_dir / 'provenance.json').read_text())
    if provenance['recording'] != labels['recording']:
        raise ValueError('Wrong recording identity')
    out = viewer_dir / 'trace-ends.html'
    if out.exists():
        raise ValueError('End viewer already exists; preserve the existing review')
    template = Path(__file__).with_name('templates') / 'trace_ends.html'
    text = template.read_text()
    digest = hashlib.sha256(script_json(labels).encode()).hexdigest()
    for key, value in {'__TIMES__': script_json(times), '__STARTS__': script_json(labels),
                       '__KEY__': script_json('trueskate-trace-ends-' + digest),
                       '__TITLE__': html.escape(labels['recording'])}.items():
        text = text.replace(key, value)
    out.write_text(text)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer-dir', type=Path, required=True)
    p.add_argument('--starts', type=Path, required=True)
    a = p.parse_args()
    print(build(a.viewer_dir, a.starts))
