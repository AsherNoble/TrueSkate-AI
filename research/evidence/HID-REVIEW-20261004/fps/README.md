# Offline recording audit evidence

The completed Claude analysis (`dup.py`) measured 29 recordings, including 20 60 fps gameplay replays, 6 30 fps replays and 3 60 fps hover controls. The resumed review reused those arrays, checked moving-scene changes against static/compression noise and quantified capture PTS gaps. This is diagnostic evidence from existing recordings, not a training holdout.

- `source-probe.json`: original per-file sizes, durations and source PTS statistics.
- `measurement-summary.json`: grouped sizes/rates, masked-motion audit and gap summary.
- `recording-audit.json`: per-file metadata and PTS count/decoded-array lengths.
- `pts-gaps.csv`: source timestamp discontinuities.
- `decode-benchmark.json`: bounded local decode/downscale timings and environment.
- `dup.py`, `look.py`: original masked tile-change calculation and inspection.
- `summarise.py`, `benchmark.py`: resumed aggregation and bounded decode/encode benchmarks. These scripts preserve the historical calculations. Before rerunning, copy them into an isolated `tmp/` directory and adapt their input/output paths; do not create temporary videos in this evidence directory.

External originals and intermediate arrays are not backed up by Git:

- `/Users/ashernoble/.claude/jobs/4e7efedf/tmp/agent-fps/rec/`: local copies of the original gameplay movies.
- `/Users/ashernoble/.claude/jobs/4e7efedf/tmp/agent-fps/dup_*.npz`: previously completed frame-change arrays.
- `/Users/ashernoble/.claude/jobs/4e7efedf/tmp/cal/rate/hover_rate_*.mov`: hover controls.
- `/Users/training-server/trueskate-ai-runtime/tmp/demo-replay-20261004-variance/`: rig originals of the 20 60 fps replays.
- `/Users/training-server/trueskate-ai-runtime/tmp/demo-replay-20261004-pointer/` and `demo-replay-20261004-pointer-v3/`: rig 30 fps originals.

Full source PTS remain authoritative. The change threshold is diagnostic; broad scene motion separated the examined gameplay window from static encoding noise. No duplicate candidate occurred among 3,364 adjacent moving pairs, which does not certify all parks/sessions.
