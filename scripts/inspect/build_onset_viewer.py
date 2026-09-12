"""Build a local, frame-exact onset annotation viewer from an original recording.

Requires ffmpeg and ffprobe. Does not contact a device or alter training data.
Output must be empty: existing annotations and frames are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess


def script_json(value):
    return json.dumps(value).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def render_viewer(times: list[float], recording: str, title: str, status: str,
                  export_name: str) -> str:
    template = Path(__file__).with_name('templates') / 'gesture_onsets.html'
    content = template.read_text()
    replacements = {
        '__TIMES__': script_json(times), '__RECORDING__': script_json(recording),
        '__TITLE__': html.escape(title), '__STATUS__': html.escape(status),
        '__STORAGE_KEY__': script_json('trueskate-onsets-' + hashlib.sha256(recording.encode()).hexdigest()),
        '__EXPORT_NAME__': script_json(export_name),
    }
    for old, new in replacements.items():
        content = content.replace(old, new)
    return content


def build(video: Path, out: Path, recording: str, title: str, status: str,
          export_name: str) -> int:
    video = video.resolve(strict=True)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty; preserve existing frames and annotations.')
    index = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
        '-show_frames', '-show_entries',
        'stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time',
        '-of', 'json', str(video)]))
    times = [float(f['best_effort_timestamp_time']) for f in index['frames']]
    if not times or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('Recording needs nonempty chronological frame timestamps.')
    frames = out / 'frames'
    frames.mkdir(parents=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-map', '0:v:0',
                    '-fps_mode', 'passthrough', '-q:v', '2', '-start_number', '0',
                    str(frames / 'frame_%06d.jpg')], check=True)
    if len(list(frames.glob('frame_*.jpg'))) != len(times):
        raise ValueError('Extracted frame count disagrees with ffprobe; viewer not published.')
    (out / 'frame_index.json').write_text(json.dumps(index))
    with video.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    (out / 'provenance.json').write_text(json.dumps({
        'recording': recording, 'original_video': str(video), 'sha256': digest,
        'frame_count': len(times), 'status': status,
        'extraction': 'Every decoded frame, original resolution, JPEG quality 2, actual presentation timestamps',
    }, indent=2))
    (out / 'index.html').write_text(render_viewer(times, recording, title, status, export_name))
    return len(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--recording-id', required=True, help='Stable session/filename identity for exported labels')
    parser.add_argument('--title', default='Gesture timing annotation')
    parser.add_argument('--status', default='Diagnostic recording; training-set membership unverified.')
    parser.add_argument('--export-name', default='timing-labels.json')
    args = parser.parse_args()
    count = build(args.video, args.out, args.recording_id, args.title, args.status, args.export_name)
    print(f'Wrote {count} frames and {args.out / "index.html"}')


if __name__ == '__main__':
    main()
