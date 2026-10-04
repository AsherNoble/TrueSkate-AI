"""Reuse Claude's 29 masked adjacent-frame measurements; write read-only audit.

Motion window is deliberately fixed at 4.0–6.8 s: continuous gameplay after
the first push, shared across the twenty short replay experiments. Full-screen
lossy equality is not a valid duplicate criterion. Retain the original orange
trail/button/home masks, discard HUD tile rows, and tolerate twenty changed
tiles per pair (far more than a 20-point cursor affects).
"""
from pathlib import Path
import csv
import json
import numpy as np

SOURCE = Path('/Users/ashernoble/.claude/jobs/4e7efedf/tmp/agent-fps')
DEST = Path('/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/claude-review-resume-20261004/fps')
DEST.mkdir(parents=True, exist_ok=True)
probe = json.loads((SOURCE / 'probe.json').read_text())
summary = {}
for group, rows in (
    ('60-game', [x for x in probe if 'variance' in x['f']]),
    ('30-game', [x for x in probe if 'pointer/' in x['f'] or 'pointer-v3/' in x['f']]),
    ('60-hover', [x for x in probe if 'hover_rate' in x['f']]),
):
    summary[group] = dict(
        videos=len(rows), duration_s=sum(x['dur'] for x in rows),
        frames=sum(x['n'] for x in rows), bytes=sum(round(x['mb'] * 1e6) for x in rows),
        mb_min=sum(x['mb'] for x in rows) * 60 / sum(x['dur'] for x in rows),
        range={key:[min(x[key] for x in rows), max(x[key] for x in rows)]
               for key in ['dur', 'mbmin', 'eff', 'dmax', 'ngap']},
    )

details = []
gaps = []
all_motion = []
all_static = []
for path in sorted(SOURCE.glob('dup_*.npz')):
    z = np.load(path)
    assert z['tiles'].shape[0] + 1 == len(z['pts']) == len(z['ptype'])
    pts = z['pts']
    # Exclude top 96 points and left HUD column; bottom 96 points contains
    # speed display and home indicator. Cursor/trails are sparse at this scale.
    tiles = z['tiles'][:, 3:25, 2:23].reshape(len(pts)-1, -1)
    valid = np.isfinite(tiles).sum(1)
    changed = (tiles > 3).sum(1)
    robust_frac = np.maximum(changed - 20, 0) / np.maximum(valid - 20, 1)
    raw_frac = changed / valid
    t = pts[1:]
    is_motion = 'variance' in path.name
    if is_motion:
        chosen = (t >= 4.0) & (t <= 6.8)
        all_motion.extend(robust_frac[chosen])
    elif 'hover' in path.name:
        chosen = (t >= .5) & (t <= pts[-1] - .1)
        all_static.extend(raw_frac[chosen])
    else:
        chosen = (t >= 4.0) & (t <= 6.8)
    rate = 60 if 'variance' in path.name or 'hover' in path.name else 30
    diffs = np.diff(pts)
    chosen_gaps = np.flatnonzero(diffs > 1.35 / rate)
    missed_grid_slots = 0
    for i in chosen_gaps:
        # At 30 fps, observed 50 ms gaps are 1.5 nominal intervals: do not
        # round them into a claimed whole missed-frame count.
        missed = max(0, round(diffs[i] * rate)-1) if rate == 60 else None
        missed_grid_slots += missed or 0
        gaps.append(dict(file=path.name, fps_requested=rate, before_s=float(pts[i]),
                         after_s=float(pts[i+1]), delta_ms=float(diffs[i]*1000),
                         missed_60hz_slots=missed))
    details.append(dict(
        file=path.name, frame_count=len(pts), requested_fps=rate,
        pairs_chosen=int(chosen.sum()),
        robust_fraction_quantiles=np.percentile(robust_frac[chosen], [0,10,50,90,100]).tolist(),
        raw_fraction_quantiles=np.percentile(raw_frac[chosen], [0,10,50,90,100]).tolist(),
        duplicate_candidates_lt_0_1=int((robust_frac[chosen]<.1).sum()),
        long_pts_gaps=len(chosen_gaps), missed_60hz_slots=missed_grid_slots if rate==60 else None,
    ))

summary['motion_audit'] = dict(
    fixed_window_s=[4.0, 6.8], pairs=len(all_motion),
    robust_changed_fraction_quantiles=np.percentile(all_motion,[0,10,50,90,100]).tolist(),
    candidates_lt_0_1=int((np.asarray(all_motion)<.1).sum()),
    hover_raw_fraction_quantiles=np.percentile(all_static,[0,90,99,100]).tolist(),
)
replay_gaps = [g for g in gaps if 'variance' in g['file']]
summary['replay_gaps'] = dict(
    count=len(replay_gaps), missed_60hz_slots=sum(g['missed_60hz_slots'] for g in replay_gaps),
    bins={label:sum(lo <= g['after_s'] < hi for g in replay_gaps)
          for label,lo,hi in [('0-1s',0,1),('1-3.5s',1,3.5),('3.5s-end',3.5,100)]},
    gameplay=[g for g in replay_gaps if g['after_s']>=3.5],
)
(DEST/'measurement-summary.json').write_text(json.dumps(summary,indent=2))
(DEST/'recording-audit.json').write_text(json.dumps(details,indent=2))
with (DEST/'pts-gaps.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f, fieldnames=list(gaps[0]))
    writer.writeheader(); writer.writerows(gaps)
print(json.dumps(summary,indent=2))
