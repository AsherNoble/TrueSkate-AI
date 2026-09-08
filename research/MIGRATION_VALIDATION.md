# BC migration validation

Updated 2026-09-08. Implementation is staged, **not merged or deployed**.

## Source and preservation

- Promotion candidate: `946639a737884b74fc1602d1ec5365730e5cae36`,
  [PR #14](https://github.com/AsherNoble/TrueSkate-AI/pull/14).
- Tested cleanup candidate: `f39201dfbf700965655984c49e6c9aa05da038cb`,
  [PR #15](https://github.com/AsherNoble/TrueSkate-AI/pull/15).
- Original sources, journals and installed service definitions are preserved
  through [ARCHIVE.md](ARCHIVE.md). Archive tags have active update/deletion
  protection. Main's proposed required-check rules are not yet enabled.
- The original rig checkout remains at
  `463316d34b81129986a171920369dd6067e91f7b`, with its ten dirty tracked
  source files intact. It has not been reset, switched or replaced.

## Completed checks

- Reconciled pre-prune baseline: 248 offline tests passed.
- Cleanup candidate: 258 local tests passed; archive integrity passed against
  `origin/main`.
- Staged rig: 255 tests passed, three skipped in 107.29 seconds, using its
  original OpenCV 4.13.0 and Torch 2.2.2 runtime. Pytest support packages were
  isolated under `tmp/`; the live virtual environment was not upgraded.
  Skips: one synthetic MP4-writer test (that OpenCV build lacks the writer)
  and two optional visual-regression tests (fixtures absent in the worktree).
  Production video decoding and FFmpeg-generated alignment fixtures passed.
- Clean Linux installation and both CI jobs passed at the tested cleanup SHA:
  [run 34172665739](https://github.com/AsherNoble/TrueSkate-AI/actions/runs/34172665739).
- 600 gesture parameter cases matched the pre-prune implementation across
  slot counts, spin layouts, bounds, clamping and unpacking.
- Editable installation, retained entrypoint help checks and a synthetic
  heatmap training smoke passed. No paid training or research holdout evaluation.
- Staged dashboard served `/`, `/data` and `/deployment.json` with HTTP 200
  against the existing rig corpus/log paths. Loaded and disk revisions both
  matched the tested SHA; source was clean and restart was not pending.
  This was an HTTP check, not a visual browser review. The temporary loopback
  server on port 8401 was stopped; the live dashboard on 8400 was untouched.

## Physical collection gate: not passed

The original live XR1 collector recorded one bounded segment: 74.4 MB,
12 gestures and two menu skips/relaunches. Only one calibration tap was detected;
the existing admission gate rejected it, preserved the recording and admitted
zero samples. This is not a successful baseline. Recording is retained at:

`/Users/training-server/trueskate-ai/tmp/migration-baseline-20260908/iPhone_XR_20260907_072027/segment_00000.mov`

The staged XR2 attempt could not establish a WDA session and produced no footage.
Subsequent checks found no USB device IDs and connection refusals on both
WDA ports (8100 and 8103). The root remotexpc tunnel daemon was running.
No healthy WDA service was manually restarted and no calibration, menu or
frame-count gate was weakened.

After the user confirmed both phones powered on, both USB IDs appeared and
XR2 WDA became ready. A staged one-minute XR2 retry recorded 64.62 MB but
skipped all 12 attempts as menu/replay frames: zero gestures, zero detected
calibration taps, zero admitted samples. A read-only screenshot confirmed
the game's bottom navigation and tutorial overlay were visible. The rejected
recording remains under
`/Users/training-server/trueskate-ai/tmp/migration-candidate-output-20260908/iPhone_XR2_20260907_172619/`.
XR1 WDA still refused connections. Power and USB visibility alone have not
restored collection readiness; normal gameplay and XR1 WDA remain prerequisites.

## Remaining acceptance and rollout

1. Restore stable powered, unlocked USB access to both XRs on training-server.
2. Pass bounded one-minute collection on each XR from the staged candidate;
   verify calibration, decoded frame counts and strict loader admission.
   Keep validation output isolated with honest park provenance.
3. Record final runtime results here. Merge #14 normally, retarget #15 to main,
   require green checks and merge normally. Enable main's required-check rules
   without imposing linear history or bypassing validation.
4. Deploy committed source at a safe boundary, preserving the original dirty
   checkout and service definitions. Record actual loaded service revisions;
   compatibility launchers alone do not change which checkout a service uses.

Follow [DEPLOYMENT.md](../DEPLOYMENT.md) for rollout and recovery. Neither
successful offline tests nor phone power-on alone waives the physical gate.
