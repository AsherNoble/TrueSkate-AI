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
