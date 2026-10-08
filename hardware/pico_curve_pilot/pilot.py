#!/usr/bin/env python3
"""Prepare and review the bounded preloaded USB curve pilot (no live execution)."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import struct
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))

from trueskate_ai.research import pico_curve_pilot as pilot
from trueskate_ai.research import pico_curve_review as review

def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def prepare(program, out):
    out.mkdir(parents=True, exist_ok=False)
    write(out/'program.json', program)
    sketch = out/'pico_curve_pilot'
    sketch.mkdir()
    # Reuse the proven mouse descriptor, latch, neutral-on-host-loss and sector layout verbatim.
    source = ROOT/'hardware/pico_hover_rate/pico_hover_rate.ino'
    shutil.copyfile(source, sketch/'pico_curve_pilot.ino')
    (sketch/'schedule.h').write_text(pilot.firmware_header(program))
    write(out/'source.json', dict(program_sha256=program['sha256'],
        firmware_source_sha256=review.file_sha256(source),
        header_sha256=review.file_sha256(sketch/'schedule.h')))
    print(f'{program["kind"]}: {program["n_events"]} events, {program["duration_us"]/1e6:.3f} s, {program["sha256"]}')


def validate_uf2(path):
    """RP2040 firmware only; never overwrite the last-sector timing/latch evidence."""
    data = path.read_bytes()
    if not data or len(data) % 512:
        raise ValueError('invalid UF2 block size')
    count = len(data)//512
    for i in range(count):
        block = data[i*512:(i+1)*512]
        m0, m1, flags, address, size, number, total, family = struct.unpack_from('<8I', block)
        if ((m0, m1) != (0x0a324655, 0x9e5d5157) or struct.unpack_from('<I', block, 508)[0] != 0x0ab16f30 or
            flags != 0x2000 or family != 0xe48bff56 or (number, total) != (i, count) or
            not 0 < size <= 476 or not 0x10000000 <= address < address+size <= 0x101ff000):
            raise ValueError('UF2 format, RP2040 family or EEPROM boundary invalid')


def package(prepared, uf2, out):
    program = read(prepared/'program.json')
    pilot.validate_program(program)
    source = read(prepared/'source.json')
    sketch = prepared/'pico_curve_pilot'
    if (source['program_sha256'] != program['sha256'] or
        source['firmware_source_sha256'] != review.file_sha256(sketch/'pico_curve_pilot.ino') or
        source['header_sha256'] != review.file_sha256(sketch/'schedule.h') or
        (sketch/'schedule.h').read_text() != pilot.firmware_header(program)):
        raise ValueError('prepared firmware sources changed')
    validate_uf2(uf2)
    # Include the compiler's saved inputs so a UF2 from a different program cannot
    # be packaged under the same filename. The map also verifies the stats address.
    compile_dir = prepared/'compile'
    compiled_header = (compile_dir/'sketch/schedule.h').read_text()
    if compiled_header.startswith('#line 1 '):
        compiled_header = compiled_header.split('\n', 1)[1]  # Arduino adds its source-location directive.
    if (compiled_header != (sketch/'schedule.h').read_text() or not re.search(
        r'0x0*101ff000\s+PROVIDE\s+\(_EEPROM_start', (compile_dir/'pico_curve_pilot.ino.map').read_text())):
        raise ValueError('compiled schedule or EEPROM map differs')
    if review.file_sha256(uf2) != review.file_sha256(compile_dir/'pico_curve_pilot.ino.uf2'):
        raise ValueError('UF2 differs from the checked compiler output')
    out.mkdir(parents=True, exist_ok=False)
    files = ['validate_ios_ipv4_recording.py', 'ios_ipv4_tunnel.mjs', 'ios_ipv4_node_tunnel.mjs',
             'ios_wifi_tunnel.py', 'ios_wifi_attachments.py', 'run_xr2_wifi_diagnostic.sh']
    for file in files:
        shutil.copyfile(ROOT/'scripts/ops'/file, out/file)
    (out/'run_xr2_wifi_diagnostic.sh').chmod(0o755)
    shutil.copyfile(ROOT/'src/trueskate_ai/research/pico_curve_pilot.py', out/'pico_curve_pilot.py')
    shutil.copyfile(ROOT/'src/trueskate_ai/control/hid_pointer.py', out/'pico_curve_hid_reference.py')
    shutil.copyfile(ROOT/'src/trueskate_ai/data/control_hitboxes.py', out/'pico_curve_controls_reference.py')
    shutil.copyfile(prepared/'program.json', out/'pico-program.json')
    (out/'fw').mkdir()
    shutil.copyfile(uf2, out/'fw/pico_curve_pilot.ino.uf2')
    shutil.copyfile(prepared/'source.json', out/'fw/source.json')
    write(out/'package-manifest.json', dict(schema='pico-curve-stage-v1', kind=program['kind'],
        program_sha256=program['sha256'], eeprom_start='0x101ff000', eeprom_end='0x10200000',
        files={str(path.relative_to(out)):review.file_sha256(path) for path in sorted(out.rglob('*')) if path.is_file()}))
    print(out/'package-manifest.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    gain = sub.add_parser('gain')
    gain.add_argument('--out', type=Path, required=True)
    curves = sub.add_parser('curves')
    curves.add_argument('--profile', type=Path, required=True)
    curves.add_argument('--batch', type=int, choices=(1, 2), required=True)
    curves.add_argument('--out', type=Path, required=True)
    stats = sub.add_parser('read-stats')
    stats.add_argument('--program', type=Path, required=True)
    stats.add_argument('--stats', type=Path, required=True)
    stats.add_argument('--out', type=Path, required=True)
    viewer = sub.add_parser('review')
    for flag in ('program', 'receipts', 'movie', 'capture-report', 'out'):
        viewer.add_argument('--'+flag, type=Path, required=True)
    viewer.add_argument('--play-origin-s', type=float)
    viewer.add_argument('--controls-bundle', type=Path)
    viewer.add_argument('--controls-export', type=Path)
    viewer.add_argument('--controls-media', type=Path)
    measure = sub.add_parser('measure-gain')
    for flag in ('program', 'receipts', 'movie', 'capture-report', 'out'):
        measure.add_argument('--'+flag, type=Path, required=True)
    measure.add_argument('--play-origin-s', type=float, required=True)
    for name in ('fit-gain', 'fit-controls', 'score'):
        cmd = sub.add_parser(name)
        for flag in ('bundle', 'export', 'media-root', 'out'):
            cmd.add_argument('--'+flag, type=Path, required=True)
    aggregate = sub.add_parser('aggregate')
    for flag in ('first', 'second', 'out'):
        aggregate.add_argument('--'+flag, type=Path, required=True)
    pack = sub.add_parser('package')
    for flag in ('prepared', 'uf2', 'out'):
        pack.add_argument('--'+flag, type=Path, required=True)
    a = p.parse_args()
    for value in vars(a).values():
        if isinstance(value, Path) and not value.is_absolute():
            p.error('all filesystem paths must be absolute')
    if 'tmp' not in a.out.parts:
        p.error('pilot output must be isolated under tmp')
    if a.command == 'gain':
        prepare(pilot.gain_program(), a.out)
    elif a.command == 'curves':
        prepare(pilot.curve_program(read(a.profile), a.batch), a.out)
    elif a.command == 'read-stats':
        write(a.out, pilot.receipts(a.stats.read_bytes(), read(a.program)))
    elif a.command == 'review':
        controls = (a.controls_bundle, a.controls_export, a.controls_media)
        if any(controls) and not all(controls):
            p.error('all three controls review paths are required together')
        control_review = (read(controls[0]), read(controls[1]), controls[2]) if all(controls) else None
        print(review.build_review(read(a.program), read(a.receipts), a.movie, read(a.capture_report), a.out,
            ROOT/'scripts/inspect/templates', origin_s=a.play_origin_s, control_review=control_review))
    elif a.command == 'aggregate':
        write(a.out, review.aggregate(read(a.first), read(a.second)))
    elif a.command == 'measure-gain':
        profile = review.measure_gain(read(a.program), read(a.receipts), a.movie, read(a.capture_report), a.out, a.play_origin_s)
        print(f'measured USB profile {profile["sha256"]}: {a.out/"profile.json"}')
    elif a.command == 'package':
        package(a.prepared, a.uf2, a.out)
    else:
        fn = {'fit-gain': review.gain_fit, 'fit-controls': review.control_fit, 'score': review.score_review}[a.command]
        result = fn(read(a.bundle), read(a.export), a.media_root)
        if a.command == 'fit-controls':
            result = result[1]
        write(a.out, result)


if __name__ == '__main__':
    main()
