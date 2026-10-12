"""Rebuild exact round-6 payloads from manifest; verify payload_sha256."""
import json, sys
from trueskate_ai.model1.curved.contract import Gesture, digest
man, out = sys.argv[1], sys.argv[2]
m = json.load(open(man))
res = []
for s in m['samples']:
    g = Gesture.from_dict(s['gesture'])  # raises on fingerprint mismatch
    p = g.payload()
    assert digest(p) == s['gesture']['payload_sha256']
    t = list(g.times_ms); pts = s['gesture']['command_points']
    holds = [i for i in range(len(t)-1) if pts[i] == pts[i+1]]
    res.append(dict(id=s['id'], park=s['park'], times_ms=t, payload=p, profile=g.profile,
                    sampler=g.sampler, hold_segments=holds, nseg=len(t)-1,
                    onset_s=s.get('onset_s'), release_s=s.get('release_s')))
json.dump(res, open(out, 'w'))
print(len(res), 'payloads verified;', sum(1 for r in res if r['hold_segments']), 'with holds')
