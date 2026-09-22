"""Synthetic recording checks; research labels are not test fixtures."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('onset_viewer', ROOT / 'scripts/inspect/build_onset_viewer.py')
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def test_viewer_escapes_identity_and_keeps_timestamp_origin():
    content = viewer.render_viewer([.065, .099], '</script>', '<title>', '<status>', 'labels.json')
    assert 'const TIMES=[0.065, 0.099]' in content
    assert '\\u003c/script\\u003e' in content
    assert '&lt;title&gt;' in content
    assert '__RECORDING__' not in content


def test_all_frames_extracted_and_existing_labels_protected(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg tools unavailable')
    video = tmp_path / 'original.mov'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=32x48:rate=10',
                    '-frames:v', '5', '-c:v', 'libx264', str(video)], check=True)
    out = tmp_path / 'review'
    assert viewer.build(video, out, 'synthetic/original.mov', 'Synthetic', 'Test', 'labels.json') == 5
    assert len(json.loads((out / 'frame_index.json').read_text())['frames']) == 5
    assert (out / 'frames/frame_000004.jpg').exists()
    before = (out / 'index.html').read_bytes()
    with pytest.raises(ValueError, match='empty'):
        viewer.build(video, out, 'synthetic/original.mov', 'Synthetic', 'Test', 'labels.json')
    assert (out / 'index.html').read_bytes() == before
    if shutil.which('node'):
        code = (out / 'index.html').read_text().split('<script>')[1].split('</script>')[0]
        js = tmp_path / 'viewer.js'
        js.write_text(code)
        subprocess.run(['node', '--check', str(js)], check=True)
