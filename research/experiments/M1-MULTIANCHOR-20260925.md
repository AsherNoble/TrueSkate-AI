# M1-MULTIANCHOR-20260925 — Multi-anchor consensus calibration (smoke)

Status: offline replay plus a bounded device smoke in the production
single-touch configuration. Output went only to an isolated rig tmp directory,
not the corpus. **No launcher was changed.**

## Idea

A single falsely early start or end detection bends the two-anchor fit
(M1-AUDIT, M1-ONSET-VALIDATION). With three or more single centre controls,
a consensus line fit can reject the bad anchor instead (exhaustive pairwise
RANSAC, 1.5-frame tolerance, ≥3 inliers over ≥30 s; `fit_multi_anchor_timeline`).
The production detector is unchanged, and no multi-touch is needed.
Background and offline replay: [M1-DIE5-COMPARE](M1-DIE5-COMPARE-20260925.md)
(44/44 fits accepted; only the known bad XR2 start and 9 post-swipe mid false
triggers were rejected; max rate deviation 756 ppm).

## Smoke

Commit `ffdfdd2`, staged under `code-multi-ffdfdd2` and run with release
`1b8497e`'s venv. The operator-loaded SLS 2015 Los Angeles, 2 segments per XR.
The command was the production linear launcher flags plus:

```
--retain-mov --mid-markers 2 --mid-marker-every 3 --mid-marker-kinds single
--start-settle-threshold 1.5 --start-settle-max-s 15 --start-settle-required
--end-settle-threshold 1.5 --end-settle-max-s 3.5
```

[Summary](../evidence/M1-MULTIANCHOR-20260925/smoke-summary.json):

- **Calibration:** 4/4 segments declared `wda_submitted_multi_anchor` and were
  accepted by the aligner. Rates were 1.000009–1.000417.
- **Clips:** 37 emitted (10/9/9/9). The strict loader admitted 37/37, all with
  32 frames.
- **Anchor agreement:** in 3 segments all four anchors agree within 0.51
  frames. In one XR2 segment a mid control was detected **17 frames early**;
  the fit rejected it and the remaining three agree within 0.09 frames.
- **Settle:** start waits settled 4/4 in 7.1–7.5 s. End waits settled 3/4;
  one timed out at 4.1 s and the end control fired anyway, as designed.
- **Session warning:** each run logged `NoSuchDriverError` after alignment,
  when the Appium session expired during the foreground aligner wait (seen
  before in M1-LA-ORIGINALS). The collector exited 0 each time.

## Limits and next step

Four segments in one park show the path runs end to end and can reject a real
bad anchor. They do not measure error rates. Mid controls after swipes are
themselves noisy (about 6% false early triggers), so consensus needs ≥4
anchors to tolerate one bad anchor with margin.

The cost is two extra controls per segment, ~1–1.5 payload slots. The
confirmatory design could reuse
[M1-SETTLE-CONFIRM-PROTOCOL](M1-SETTLE-CONFIRM-PROTOCOL-20260925.md) with
human-labelled anchor error as the outcome. Operator approval is needed.

## Red-team review and corrections (2026-09-25)

An experiment-red-team review (read-only) returned **FIXABLE**: the code works
and fails closed. Nothing yet shows multi-anchor clips are more *accurate*,
only that their anchors are self-consistent.

**Corrected claims**

- The "known-bad" XR2 start and the 9 mid false triggers were identified by
  the same label-free consistency proxy the fit uses. The offline replay shows
  the fit is *consistent with that proxy*, not validated against ground truth.
  The 1.5-frame tolerance was chosen on Phase 2 data, which is inside the
  replay.
- A falsely early anchor of up to ~1.5 frames can be accepted as an inlier.
  Timeline bending is *bounded by about the tolerance*, not prevented.
- ALIGN-20260913's 28/28-within-one-frame result validated two-anchor
  interpolation between anchors. It does not cover this method or
  extrapolation beyond the inliers.
- "37/37 strictly admitted" is not accuracy evidence. The rejected 17-frame
  mid (score 11.5) has not been human-checked.

**Code fixes** (tests added; 362 pass)

- The fit refits on inliers until the set is stable. Inlier and outlier sets
  now always partition the anchors, and every inlier is within tolerance of
  the final line.
- The timing screen no longer passes multi-anchor clips blindly. It requires
  `accepted`, ≥3 inliers, and a gesture WDA time inside the inlier range, so
  extrapolated clips are excluded.
- The collector declares `wda_submitted_multi_anchor` only when a mid control
  actually fired in the segment.
- The docstring now names the pairwise-consensus method.

**Open items**

- Mid controls go through the W3C pointer path (`_execute_marker`) while
  start/end use `_execute(tap)`. Any fixed latency difference would bias mids;
  the smoke agreement (≤0.51 frames in 3/4 segments) suggests it is small.
- Four anchors with `min_inliers=3` gives thin redundancy: two bad anchors
  fail closed.

**Next check:** once Phase 3 labels exist, score inlier/outlier decisions
against human first-touch frames. The phase 3 set contains die-five
start/end/mid markers, so anchor correctness can be scored there, but not
production single-control mids. A labelled production-path run is still
needed.
