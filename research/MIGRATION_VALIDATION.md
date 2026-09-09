# BC migration validation

Closed out 2026-09-09. Both-phone candidate acceptance, BC promotion, cleanup
merge and live source rollout are complete. The checks below are historical
candidate evidence, not a claim that both phones are currently ready.
XR1's later signing issue is separate operational recovery.

## Source and preservation

- Promotion candidate: `946639a737884b74fc1602d1ec5365730e5cae36`,
  [PR #14](https://github.com/AsherNoble/TrueSkate-AI/pull/14).
- Tested cleanup candidate: `f39201dfbf700965655984c49e6c9aa05da038cb`,
  [PR #15](https://github.com/AsherNoble/TrueSkate-AI/pull/15).
- Original sources, journals and installed service definitions are preserved
  through [ARCHIVE.md](ARCHIVE.md). Archive tags have active update/deletion
  protection. Main's required-check rules are active (ruleset 22508756):
  PRs, strict `archive-integrity`/`offline-tests`, no force-push or deletion.
- BC promotion #14 merged normally as
  `4eb9cc3c6ad58f4258d7b41431729e694fa20cf8`. Cleanup #15 merged normally as
  `c8c384ffff6cc1bfdf75b3ec43c226886f234b85`, without rebasing or squashing history.
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

## Physical collection gate: passed with explicit idle-navigation allowance

Both XRs passed at `292838de27cbf24ee8f39d378d7884ddc3044827`.
Earlier rejected attempts below are retained as diagnostic history, not successes.

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

### Operator clarification and idle-navigation correction

The operator subsequently confirmed both screens were usable and that XR2's
bottom bar appears during idle gameplay. The previous screenshot diagnosis was
too strong: the neutral navigation signature alone does not establish blocked
gameplay. Live measurements isolated this signature (four neutral cells,
red/teal fractions both zero, editor detection false). Its score fluctuated
around the threshold; a brief passing observation was not sufficient validation.

Commit `292838de27cbf24ee8f39d378d7884ddc3044827` adds explicit
`--allow-idle-navigation` collection, recorded in the segment manifest. Replay
and editor checks remain active, and dataset filtering and calibration are
unchanged. Local tests: 259 passed; both Linux CI runs passed. Rig guard tests:
five passed, two optional visual-fixture skips. See the operating guidance in
[DEPLOYMENT.md](../DEPLOYMENT.md#idle-navigation-versus-blocking-menus).

Both WDA endpoints recovered through the existing supervisor. XR1's original
collector resumed and was left undisturbed during the XR2 retry.

XR2 then passed the bounded candidate run with the explicit allowance:
79.07 MB, ten gestures, zero UI skips; calibration accepted two detected taps
(MAD 0.0 s). All ten samples aligned. Strict linear admission accepted seven
drags and rejected the three calibration taps as non-linear. Each accepted clip
decoded to exactly 32 frames matching metadata, and the loader produced a
`(32, 3, 288, 128)` tensor. Output:
`/Users/training-server/trueskate-ai/tmp/migration-candidate-output-20260908/iPhone_XR2_20260907_213035/`.
The collector automatically deleted the source MOV after successful alignment;
sample clips and manifests remain. A duplicate disconnect emitted an already-
terminated-session warning after successful completion; it did not invalidate
the recording or alignment.

XR1 passed the same isolated check: 54.82 MB, 12 gestures, zero UI skips;
calibration accepted all three taps (MAD 0.0 s). All 12 samples aligned;
strict linear admission accepted nine drags and excluded three calibration taps.
All nine accepted clips decoded to exactly 32 frames matching metadata, and
the loader returned `(32, 3, 288, 128)`. Output:
`/Users/training-server/trueskate-ai/tmp/migration-candidate-output-20260908/iPhone_XR_20260907_213520/`.
The source MOV was automatically deleted after successful alignment. The original
collector was gracefully interrupted after suspending its restart wrapper;
the wrapper resumed in a `finally` block after candidate exit 0. Process inspection
confirmed a new original collector running. WDA was never manually restarted.

## Acceptance and rollout — complete

Both-phone physical acceptance passed; isolated outputs remain preserved.
Cleanup #15 merged and its source was deployed, preserving the dirty checkout
and service definitions. See [initial rollout](RIG_ROLLOUT_20260908.md).
The later autofixer-removal PR #17 also merged and was deployed as
`9ff97c5fb29e799ac7cf027fde100cbce9b64600`; the live dashboard reported that
exact loaded/disk revision, clean source and no pending restart.

There is no outstanding merge/refactor acceptance gate. XR1's later reboot
exposed an Xcode account/provisioning failure; its bounded recovery check is
deferred to operator signing work. Collection is intentionally off, including
after any future recovery probe. See [autostart investigation](COLLECTION_AUTOSTART_20260909.md)
and [current deployment guidance](../DEPLOYMENT.md).
