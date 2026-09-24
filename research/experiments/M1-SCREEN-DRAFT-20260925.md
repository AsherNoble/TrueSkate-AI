# M1-SCREEN-DRAFT-20260925 — Draft timing screen for the replacement corpus

Status: read-only analysis and a prepared blind check. **No clip was excluded,
moved or relabelled, and no manifest was built.** The screen choice and any
recollection are operator decisions.

## Why a screen is needed

M1-AUDIT and M1-ONSET-VALIDATION showed that a falsely early calibration
detection bends the two-anchor fit. Assuming the true clock rate is ~1.0, the
predicted onset error of a clip is `(rate − 1) × (end_wda − gesture_wda)`
(start anchor implicated) or `(1 − rate) × (gesture_wda − start_wda)` (end
anchor implicated). At extreme rates this matched 22/24 blind labels exactly
and 24/24 within one displayed frame.

## Noise floor

Across parks without the fault, the fitted-rate spread is about ±600 ppm
(5th–95th percentile; e.g. Workshop −535/+579 ppm), consistent with ±1 native
frame of anchor quantisation over a ~55 s span. XR1 Los Angeles is shifted:
median +735 ppm, 95th percentile +13,461 ppm. So Los Angeles start anchors are
probably mildly early even inside the current 0.002 screen.

## Screen options

Script: [calibration_clip_screen.py](../../scripts/inspect/calibration_clip_screen.py),
run read-only on the rig over the 13,592-clip replacement pool plus the
310-clip exact-PTS baseline (13,902 clips). "Clip ≤1 frame" keeps clips in
flagged segments whose predicted onset error is at most one native frame.

| Park | Target | All | Segment-exclude 0.002 | Clip ≤1 frame @0.002 | Clip ≤1 frame @0.0008 |
|---|---:|---:|---:|---:|---:|
| The Workshop | 4,029 | 4,052 | 4,052 | 4,052 | 4,041 |
| SLS 2013 Kansas City | 3,784 | 3,809 | 3,770 | 3,772 | 3,760 |
| Skateboard GB 2024 | 2,017 | 2,020 | 1,973 | 1,975 | 1,965 |
| SLS 2015 Super Crown | 2,009 | 2,018 | 1,977 | 1,979 | 1,964 |
| SLS 2015 Los Angeles | 1,261 | 2,003 | 1,067 | 1,152 | 1,089 |
| **Total** | **13,100** | **13,902** | **12,839** | **12,930** | **12,819** |

Evidence: [0.002](../evidence/M1-SCREEN-DRAFT-20260925/clip-screen.json),
[0.001](../evidence/M1-SCREEN-DRAFT-20260925/clip-screen-0.001.json),
[0.0008](../evidence/M1-SCREEN-DRAFT-20260925/clip-screen-0.0008.json).

Clip-level salvage is small (91 clips at 0.002) because high-rate segments
are usually far outside one frame except at their last gesture. Under any
screen, the historical mixture cannot be met without recollection. With the
strictest option (clip ≤1 frame at 0.0008), the shortfalls are about Los
Angeles −172, Skateboard GB −52, Super Crown −45 and Kansas City −24.

## Open question and prepared check

The error formula is unvalidated between 0.0008 and 0.002, and that is where
the threshold choice matters. A blind 24-clip check is prepared and was **not
yet labelled**. It uses [select_midband_onset_validation.py](../../scripts/inspect/select_midband_onset_validation.py)
with seed 2509250317 and excludes all audit and earlier validation sessions.

- Mid-high (`1.0012 < rate ≤ 1.002`): all 6 available recordings, all XR1 Los
  Angeles. The early clip is predicted at displayed frame 9–10; the late clip
  at 8.
- Mid-low (`0.998 ≤ rate < 0.9988`): the only available recording (XR1 Los
  Angeles), with the reverse pattern.
- Ordinary (`abs(rate − 1) < 0.0003`): 5 recordings in other parks; all
  predicted at frame 8.

A first draw with the bands at 0.0008–0.002 was discarded before any viewing
or labelling. Its predictions (≈1.2–1.6 native frames) could not be told apart
from anchor quantisation. The quotas then became "all available" because the
narrower bands hold only 6 and 1 recordings.

The predictions and selection are sealed. Their SHA-256 hashes are in
[midband-sealed-sha256.txt](../evidence/M1-SCREEN-DRAFT-20260925/midband-sealed-sha256.txt),
and the files stay on the rig until labels are exported.

- Viewer: `http://100.113.165.56:8772/`, served from the `viewer/` directory
  only. `selection.json` returns 404.
- Known blinding limit, shared with the earlier viewer: item IDs in the page
  source contain the corpus path, which names the park. They are not shown on
  screen.
- The export downloads as `model1-onset-validation-20260924.json` because the
  builder was reused unchanged; the file's seed field reads 2509250317.

Scoring: compare the human first-trace frame with the sealed predicted frame,
using the ONSET-VALIDATION convention of exact stored frame times.

- Mid-band screen supported: most mid-high early clips land at their predicted
  late frame, and ordinary clips stay at 8.
- Only extremes matter: mid-high early clips sit at 8 like the ordinary ones.
  Then the 0.002 segment screen is adequate.

## Cohort-builder integration (commit `0e73bd6`)

`build_model1_scaling_manifests.py cohort` has an opt-in timing screen:
`--timing-screen-rate R --timing-screen-max-frames F`. Its parameters and
per-park exclusion counts are sealed into the manifest (`timing_screen`). The
default behaviour and fingerprints of unscreened cohorts are unchanged.

Dry run over `model1_linear_replacement_20260922` only (13,592 clips; the
310-clip baseline lives in a separate root). Draft manifests were written only
to the rig tmp directory
`/Users/training-server/trueskate-ai-runtime/tmp/die5-compare-phase2-20260925/draft-manifests/`:

| Screen | Clips | Excluded | Los Angeles | Kansas City | Super Crown | Skateboard GB | Workshop |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 13,592 | 0 | 2,003 | 3,809 | 2,018 | 1,710 | 4,052 |
| 0.002, ≤1 frame | 12,628 | 964 | 1,152 | 3,772 | 1,979 | 1,673 | 4,052 |
| 0.0008, ≤1 frame | 12,517 | 1,075 | 1,089 | 3,760 | 1,964 | 1,663 | 4,041 |

The Los Angeles counts match the standalone screen exactly. None of these
manifests is a training decision.
