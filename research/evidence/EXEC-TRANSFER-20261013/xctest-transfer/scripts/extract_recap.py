"""Read only: extract controls and targeted methods from installed simulator Recap."""
import hashlib
import json
import pathlib
import struct
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'resume-20261013'
runtime = pathlib.Path('/Library/Developer/CoreSimulator/Volumes/iOS_22G86/Library/Developer/CoreSimulator/Profiles/Runtimes/iOS 18.6.simruntime/Contents/Resources/RuntimeRoot/System/Library/PrivateFrameworks/Recap.framework/Recap')
data = runtime.read_bytes()
assert struct.unpack_from('>I', data)[0] == 0xcafebabe
narch = struct.unpack_from('>I', data, 4)[0]
base = next(struct.unpack_from('>IIIII', data, 8 + 20 * i)[2] for i in range(narch) if struct.unpack_from('>I', data, 8 + 20 * i)[0] == 0x0100000c)
assert struct.unpack_from('<I', data, base)[0] == 0xfeedfacf
ncommands = struct.unpack_from('<I', data, base + 16)[0]
cursor = base + 32
address = 0x24090
for _ in range(ncommands):
    command, size = struct.unpack_from('<II', data, cursor)
    if command == 0x19:
        vmaddr, vmsize, fileoff, filesize = struct.unpack_from('<QQQQ', data, cursor + 24)
        if vmaddr <= address < vmaddr + filesize:
            offset = base + fileoff + address - vmaddr
            break
    cursor += size
else:
    raise AssertionError('Control address not in an arm64 file segment')
controls = struct.unpack_from('<8f', data, offset)
proof = {'runtime_path': str(runtime), 'sha256': hashlib.sha256(data).hexdigest(), 'arm64_slice_offset': base,
         'constant_virtual_address': hex(address), 'touch_control_points': list(controls[:4]),
         'other_input_control_points': list(controls[4:]),
         'interpretation': 'RCPMediaTimingFunctionForInputType input0 flag0 selects first four; touchCurveFunction passes input0, flag0.'}
(OUT / 'recap-control-points.json').write_text(json.dumps(proof, indent=2))
disassembly = subprocess.check_output(['otool', '-arch', 'arm64', '-tvV', str(runtime)], text=True)
(OUT / 'recap-arm64-disassembly.txt').write_text(disassembly)
wanted = ['_RCPMediaTimingFunctionForInputType:', '-[RCPSyntheticEventStream touchCurveFunction]:',
          '-[RCPSyntheticEventStream _nextTouchLocationFrom:to:at:]:',
          '-[CAMediaTimingFunction(RCPEasingFunction) rcp_solveForStart:end:at:]:',
          '-[RCPSyntheticEventStream moveToPoints:touchCount:pressure:duration:radius:]:']
lines = disassembly.splitlines()
found = []
for title in wanted:
    starts = [i for i, line in enumerate(lines) if line == title]
    assert len(starts) == 1, (title, starts)
    start = starts[0]
    stop = start + 1
    while stop < len(lines) and (not lines[stop] or lines[stop][0].isdigit()):
        stop += 1
    found.append('\n'.join(lines[start:stop]))
(OUT / 'recap-targeted-methods-reproduced.txt').write_text('\n\n'.join(found) + '\n')
print(json.dumps(proof, indent=2))
